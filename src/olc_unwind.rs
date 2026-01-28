#![allow(unsafe_op_in_unsafe_fn)]

//! OLC Retry Strategy: Stack Unwinding via Panic
//!
//! This module implements Optimistic Lock Coupling retry via `panic_any(VersionMismatch)`
//! and `catch_unwind`. When validation fails, we panic with a sentinel type, which unwinds
//! the stack back to the `catch_unwind` handler in the public API.
//!
//! ## Trade-offs
//!
//! **Pros:**
//! - Zero branches on the happy path - no `if err` checks at each stack frame
//! - The validation failure is truly exceptional, so unwinding overhead is acceptable
//! - CPU pipeline stays clean without branch misprediction overhead
//!
//! **Cons:**
//! - Relies on DWARF exception tables (on most platforms) for unwinding
//! - Unwinding is EXPENSIVE (~100-1000x slower than a normal return)
//! - Requires `panic = "unwind"` (won't work with `panic = "abort"`)
//! - Less idiomatic for Rust control flow
//!
//! ## Exception Safety
//!
//! This is safe because:
//! 1. OLC readers don't hold any locks - they just read with version validation
//! 2. Writers use RAII guards (NodeWriteGuard) that release locks on drop
//! 3. The panic only happens in the read path before any writes
//! 4. No mutexes are poisoned because we don't hold std::sync primitives
//!
//! ## CPU Mechanism: Why This Can Be Faster
//!
//! The happy path involves no conditional branches for error checking. Modern CPUs:
//! - Have limited branch prediction resources
//! - Suffer pipeline stalls on mispredicted branches
//! - The Result approach adds N branches for N stack frames
//! - The Unwind approach has 0 branches until unwinding begins
//!
//! However, when unwinding DOES happen, it's extremely slow because it must:
//! - Walk the DWARF exception tables to find handlers
//! - Run destructors for all stack frames
//! - Perform personality routine lookups
//!
//! Net effect: Faster happy path, MUCH slower retry path.

use std::cell::UnsafeCell;
use std::panic::{AssertUnwindSafe, catch_unwind, panic_any, resume_unwind};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::any::Any;

/// Sentinel type for OLC validation failure.
/// We use a distinct type so we can distinguish it from real panics.
#[derive(Debug)]
pub struct VersionMismatch;

pub const DEFAULT_CAP: usize = 11;
pub const DEFAULT_CHILDREN_CAP: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(u32);

#[derive(Debug)]
pub struct Node<K, V, const CAP: usize = DEFAULT_CAP, const CHILDREN_CAP: usize = DEFAULT_CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    version: AtomicU64,
    keys: UnsafeCell<[K; CAP]>,
    values: UnsafeCell<[V; CAP]>,
    children: UnsafeCell<[Option<NodeId>; CHILDREN_CAP]>,
    len: UnsafeCell<usize>,
    is_leaf: UnsafeCell<bool>,
}

unsafe impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> Sync for Node<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{}

impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> Node<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    pub fn new(is_leaf: bool) -> Self {
        Self {
            version: AtomicU64::new(0),
            len: UnsafeCell::new(0),
            is_leaf: UnsafeCell::new(is_leaf),
            keys: UnsafeCell::new([K::default(); CAP]),
            values: UnsafeCell::new([V::default(); CAP]),
            children: UnsafeCell::new([None; CHILDREN_CAP]),
        }
    }

    fn new_uninit() -> Self {
        Self {
            version: AtomicU64::new(u64::MAX),
            len: UnsafeCell::new(0),
            is_leaf: UnsafeCell::new(true),
            keys: UnsafeCell::new([K::default(); CAP]),
            values: UnsafeCell::new([V::default(); CAP]),
            children: UnsafeCell::new([None; CHILDREN_CAP]),
        }
    }

    #[inline]
    pub fn read_version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Validates version. PANICS with VersionMismatch on failure.
    /// This is the key difference: no Result, just panic.
    #[inline]
    pub fn validate_or_panic(&self, start_version: u64) {
        std::sync::atomic::fence(Ordering::Acquire);
        let current = self.version.load(Ordering::Relaxed);
        if current != start_version || current % 2 != 0 {
            panic_any(VersionMismatch);
        }
        // No branch on the happy path after this point!
    }

    /// Check if version is even (unlocked). PANICS if locked.
    #[inline]
    pub fn check_unlocked_or_panic(&self, version: u64) {
        if version % 2 != 0 {
            panic_any(VersionMismatch);
        }
    }

    pub fn write_lock(&self) {
        let mut backoff = 0;
        loop {
            let v = self.version.load(Ordering::Relaxed);
            if v % 2 == 0
                && self
                    .version
                    .compare_exchange_weak(v, v + 1, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
            {
                return;
            }
            for _ in 0..(1 << backoff) {
                std::hint::spin_loop();
            }
            if backoff < 6 {
                backoff += 1;
            }
        }
    }

    pub fn write_unlock(&self) {
        self.version.fetch_add(1, Ordering::Release);
    }

    #[inline]
    unsafe fn read_len(&self) -> usize {
        std::ptr::read_volatile(self.len.get())
    }

    #[inline]
    unsafe fn read_is_leaf(&self) -> bool {
        std::ptr::read_volatile(self.is_leaf.get())
    }

    #[inline]
    unsafe fn read_key(&self, idx: usize) -> K {
        let ptr = self.keys.get() as *const K;
        std::ptr::read_volatile(ptr.add(idx))
    }

    #[inline]
    unsafe fn read_value(&self, idx: usize) -> V {
        let ptr = self.values.get() as *const V;
        std::ptr::read_volatile(ptr.add(idx))
    }

    #[inline]
    unsafe fn read_child(&self, idx: usize) -> Option<NodeId> {
        let ptr = self.children.get() as *const Option<NodeId>;
        std::ptr::read_volatile(ptr.add(idx))
    }

    #[inline]
    unsafe fn write_key(&self, idx: usize, key: K) {
        let ptr = self.keys.get() as *mut K;
        std::ptr::write(ptr.add(idx), key);
    }

    #[inline]
    unsafe fn write_value(&self, idx: usize, value: V) {
        let ptr = self.values.get() as *mut V;
        std::ptr::write(ptr.add(idx), value);
    }

    #[inline]
    unsafe fn write_child(&self, idx: usize, child: Option<NodeId>) {
        let ptr = self.children.get() as *mut Option<NodeId>;
        std::ptr::write(ptr.add(idx), child);
    }

    #[inline]
    unsafe fn write_len(&self, len: usize) {
        std::ptr::write(self.len.get(), len);
    }

    #[inline]
    unsafe fn shift_keys_right(&self, start: usize, count: usize) {
        if count == 0 { return; }
        let ptr = self.keys.get() as *mut K;
        std::ptr::copy(ptr.add(start), ptr.add(start + 1), count);
    }

    #[inline]
    unsafe fn shift_values_right(&self, start: usize, count: usize) {
        if count == 0 { return; }
        let ptr = self.values.get() as *mut V;
        std::ptr::copy(ptr.add(start), ptr.add(start + 1), count);
    }

    #[inline]
    unsafe fn shift_children_right(&self, start: usize, count: usize) {
        if count == 0 { return; }
        let ptr = self.children.get() as *mut Option<NodeId>;
        std::ptr::copy(ptr.add(start), ptr.add(start + 1), count);
    }

    #[inline]
    unsafe fn copy_keys_from(&self, dst_start: usize, src: &Self, src_start: usize, count: usize) {
        if count == 0 { return; }
        let src_ptr = src.keys.get() as *const K;
        let dst_ptr = self.keys.get() as *mut K;
        std::ptr::copy_nonoverlapping(src_ptr.add(src_start), dst_ptr.add(dst_start), count);
    }

    #[inline]
    unsafe fn copy_values_from(&self, dst_start: usize, src: &Self, src_start: usize, count: usize) {
        if count == 0 { return; }
        let src_ptr = src.values.get() as *const V;
        let dst_ptr = self.values.get() as *mut V;
        std::ptr::copy_nonoverlapping(src_ptr.add(src_start), dst_ptr.add(dst_start), count);
    }

    #[inline]
    unsafe fn copy_children_from(&self, dst_start: usize, src: &Self, src_start: usize, count: usize) {
        if count == 0 { return; }
        let src_ptr = src.children.get() as *const Option<NodeId>;
        let dst_ptr = self.children.get() as *mut Option<NodeId>;
        std::ptr::copy_nonoverlapping(src_ptr.add(src_start), dst_ptr.add(dst_start), count);
    }

    #[inline]
    unsafe fn binary_search_raw(&self, key: &K, len: usize) -> Result<usize, usize> {
        let mut left = 0;
        let mut right = len;
        while left < right {
            let mid = left + (right - left) / 2;
            let mid_key = self.read_key(mid);
            match mid_key.cmp(key) {
                std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Greater => right = mid,
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        Err(left)
    }
}

pub struct NodeWriteGuard<'a, K, V, const CAP: usize = DEFAULT_CAP, const CHILDREN_CAP: usize = DEFAULT_CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    node: &'a Node<K, V, CAP, CHILDREN_CAP>,
    released: bool,
}

impl<'a, K, V, const CAP: usize, const CHILDREN_CAP: usize> NodeWriteGuard<'a, K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    #[inline]
    fn new(node: &'a Node<K, V, CAP, CHILDREN_CAP>) -> Self {
        node.write_lock();
        Self { node, released: false }
    }

    #[allow(dead_code)]
    #[inline]
    fn release(mut self) {
        if !self.released {
            self.node.write_unlock();
            self.released = true;
        }
    }
}

impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> Drop for NodeWriteGuard<'_, K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    fn drop(&mut self) {
        if !self.released {
            self.node.write_unlock();
        }
    }
}

impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> std::ops::Deref for NodeWriteGuard<'_, K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    type Target = Node<K, V, CAP, CHILDREN_CAP>;

    fn deref(&self) -> &Self::Target {
        self.node
    }
}

/// Check if a panic payload is our VersionMismatch sentinel.
fn is_version_mismatch(payload: &Box<dyn Any + Send>) -> bool {
    payload.is::<VersionMismatch>()
}

/// B-Tree using OLC with panic-based retry (stack unwinding).
pub struct BTreeUnwind<K, V, const CAP: usize = DEFAULT_CAP, const CHILDREN_CAP: usize = DEFAULT_CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    pages: UnsafeCell<Vec<Node<K, V, CAP, CHILDREN_CAP>>>,
    next_free_idx: AtomicUsize,
    root_id: AtomicU32,
    node_capacity: usize,
    entry_count: AtomicUsize,
}

unsafe impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> Sync for BTreeUnwind<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{}

unsafe impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> Send for BTreeUnwind<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{}

impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> Default for BTreeUnwind<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> BTreeUnwind<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    const DEFAULT_NODE_CAPACITY: usize = 10_000;

    pub fn new() -> Self {
        Self::with_node_capacity(Self::DEFAULT_NODE_CAPACITY)
    }

    pub fn with_capacity(max_nodes: usize) -> Self {
        Self::with_node_capacity(max_nodes)
    }

    pub fn with_node_capacity(max_nodes: usize) -> Self {
        assert!(max_nodes >= 1, "Arena must have at least 1 node");

        let mut pages = Vec::with_capacity(max_nodes);
        pages.push(Node::new(true));
        for _ in 1..max_nodes {
            let node = Node::new_uninit();
            unsafe {
                std::ptr::write_volatile(node.keys.get() as *mut u8, 0);
            }
            pages.push(node);
        }

        BTreeUnwind {
            pages: UnsafeCell::new(pages),
            next_free_idx: AtomicUsize::new(1),
            root_id: AtomicU32::new(0),
            node_capacity: max_nodes,
            entry_count: AtomicUsize::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.entry_count.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn node(&self, id: NodeId) -> &Node<K, V, CAP, CHILDREN_CAP> {
        unsafe {
            let ptr = self.pages.get();
            let slice = &*ptr;
            &slice[id.0 as usize]
        }
    }

    fn new_node(&self, is_leaf: bool) -> NodeId {
        let idx = self.next_free_idx.fetch_add(1, Ordering::AcqRel);

        if idx >= self.node_capacity {
            panic!("Arena OOM: Tree exceeded capacity of {} nodes.", self.node_capacity);
        }

        let n = self.node(NodeId(idx as u32));

        unsafe {
            std::ptr::write(n.keys.get(), [K::default(); CAP]);
            std::ptr::write(n.values.get(), [V::default(); CAP]);
            std::ptr::write(n.children.get(), [None; CHILDREN_CAP]);
            std::ptr::write(n.len.get(), 0);
            std::ptr::write(n.is_leaf.get(), is_leaf);
            n.version.store(0, Ordering::Release);
        }
        NodeId(idx as u32)
    }

    /// Public search API using catch_unwind for retry.
    ///
    /// The catch_unwind boundary is the ONLY place where we handle retries.
    /// This is the key architectural difference from the Result approach.
    pub fn search(&self, key: K) -> Option<V> {
        loop {
            // AssertUnwindSafe is needed because &self might not be UnwindSafe
            // (due to UnsafeCell). However, we know this is safe because:
            // 1. We only read data, no mutations in the optimistic path
            // 2. If we panic, no invariants are violated
            let result = catch_unwind(AssertUnwindSafe(|| {
                self.search_inner(key)
            }));

            match result {
                Ok(value) => return value,
                Err(payload) => {
                    if is_version_mismatch(&payload) {
                        // Our sentinel - retry
                        continue;
                    } else {
                        // Real panic - re-raise it
                        resume_unwind(payload);
                    }
                }
            }
        }
    }

    /// Inner search that panics on version mismatch.
    ///
    /// Notice: NO error handling code in this function!
    /// The validate_or_panic calls either succeed silently or unwind.
    /// This means the CPU sees a straight-line code path with no branches
    /// for error checking.
    #[inline]
    fn search_inner(&self, key: K) -> Option<V> {
        let mut current_id = NodeId(self.root_id.load(Ordering::Acquire));

        loop {
            let node = self.node(current_id);

            let start_version = node.read_version();
            node.check_unlocked_or_panic(start_version); // Panics if locked

            let read_result = unsafe {
                let len = node.read_len();
                let is_leaf = node.read_is_leaf();

                match node.binary_search_raw(&key, len) {
                    Ok(idx) => Ok(Some(node.read_value(idx))),
                    Err(idx) => {
                        if is_leaf {
                            Ok(None)
                        } else {
                            Err(node.read_child(idx))
                        }
                    }
                }
            };

            node.validate_or_panic(start_version); // Panics if version changed

            match read_result {
                Ok(result) => return result,
                Err(child_opt) => {
                    match child_opt {
                        Some(child_id) => current_id = child_id,
                        None => panic_any(VersionMismatch), // Child was None
                    }
                }
            }
        }
    }

    /// Public insert API using catch_unwind.
    pub fn insert(&self, key: K, value: V) {
        loop {
            let result = catch_unwind(AssertUnwindSafe(|| {
                self.insert_optimistic_inner(key, value)
            }));

            match result {
                Ok(success) => {
                    if success {
                        return;
                    }
                    // success == false means we need pessimistic path
                    self.insert_pessimistic(key, value);
                    return;
                }
                Err(payload) => {
                    if is_version_mismatch(&payload) {
                        continue; // Retry
                    } else {
                        resume_unwind(payload);
                    }
                }
            }
        }
    }

    /// Optimistic insert that panics on version mismatch.
    /// Returns true if insert succeeded, false if we need pessimistic fallback.
    #[inline]
    fn insert_optimistic_inner(&self, key: K, value: V) -> bool {
        let mut current_id = NodeId(self.root_id.load(Ordering::Acquire));

        loop {
            let node = self.node(current_id);
            let start_version = node.read_version();

            let next_step = unsafe {
                let len = node.read_len();
                let is_leaf = node.read_is_leaf();

                if is_leaf {
                    Err(())
                } else {
                    let idx = match node.binary_search_raw(&key, len) {
                        Ok(i) => i + 1,
                        Err(i) => i,
                    };
                    match node.read_child(idx) {
                        Some(id) => Ok(id),
                        None => panic_any(VersionMismatch),
                    }
                }
            };

            node.validate_or_panic(start_version);

            match next_step {
                Ok(child_id) => current_id = child_id,
                Err(()) => {
                    let guard = NodeWriteGuard::new(node);

                    let current_version = guard.version.load(Ordering::Relaxed);
                    if current_version != start_version + 1 {
                        panic_any(VersionMismatch);
                    }

                    let is_leaf = unsafe { guard.read_is_leaf() };
                    if !is_leaf {
                        panic_any(VersionMismatch);
                    }

                    let is_full = unsafe { guard.read_len() >= CAP };
                    if is_full {
                        // Need pessimistic path - guard will unlock on drop
                        return false;
                    }

                    let is_new = unsafe {
                        let len = guard.read_len();

                        match guard.binary_search_raw(&key, len) {
                            Ok(idx) => {
                                guard.write_value(idx, value);
                                false
                            }
                            Err(idx) => {
                                guard.shift_keys_right(idx, len - idx);
                                guard.shift_values_right(idx, len - idx);
                                guard.write_key(idx, key);
                                guard.write_value(idx, value);
                                guard.write_len(len + 1);
                                true
                            }
                        }
                    };

                    drop(guard);
                    if is_new {
                        self.entry_count.fetch_add(1, Ordering::Relaxed);
                    }
                    return true;
                }
            }
        }
    }

    fn insert_pessimistic(&self, key: K, value: V) {
        let root_id = loop {
            let root_id = NodeId(self.root_id.load(Ordering::Acquire));
            let root = self.node(root_id);
            root.write_lock();
            let current_root = NodeId(self.root_id.load(Ordering::Acquire));
            if current_root == root_id {
                break root_id;
            }
            root.write_unlock();
        };
        let root = self.node(root_id);
        let root_full = unsafe { root.read_len() == CAP };

        let current_root_id = if root_full {
            let new_root_id = self.new_node(false);
            let new_root = self.node(new_root_id);

            unsafe {
                new_root.write_child(0, Some(root_id));
            }

            self.split_child(new_root_id, 0);
            self.root_id.store(new_root_id.0, Ordering::Release);
            new_root_id
        } else {
            root.write_unlock();
            root_id
        };

        self.insert_pessimistic_non_full(current_root_id, key, value);
    }

    fn insert_pessimistic_non_full(&self, node_id: NodeId, key: K, value: V) {
        let node = self.node(node_id);

        node.write_lock();
        let is_leaf = unsafe { node.read_is_leaf() };

        if is_leaf {
            let is_new = unsafe {
                let len = node.read_len();

                match node.binary_search_raw(&key, len) {
                    Ok(idx) => {
                        node.write_value(idx, value);
                        false
                    }
                    Err(idx) => {
                        node.shift_keys_right(idx, len - idx);
                        node.shift_values_right(idx, len - idx);
                        node.write_key(idx, key);
                        node.write_value(idx, value);
                        node.write_len(len + 1);
                        true
                    }
                }
            };
            node.write_unlock();
            if is_new {
                self.entry_count.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            let (idx, child_id) = unsafe {
                let len = node.read_len();
                let idx = match node.binary_search_raw(&key, len) {
                    Ok(i) => i + 1,
                    Err(i) => i,
                };
                let child_id = node.read_child(idx).expect("Structure broken");
                (idx, child_id)
            };

            let child = self.node(child_id);
            child.write_lock();
            let child_full = unsafe { child.read_len() == CAP };

            if child_full {
                self.split_child(node_id, idx);

                let go_right = unsafe { key > node.read_key(idx) };

                if go_right {
                    let right_id = unsafe { node.read_child(idx + 1).expect("right child must exist") };
                    node.write_unlock();
                    self.insert_pessimistic_non_full(right_id, key, value);
                } else {
                    node.write_unlock();
                    self.insert_pessimistic_non_full(child_id, key, value);
                }
            } else {
                child.write_unlock();
                node.write_unlock();
                self.insert_pessimistic_non_full(child_id, key, value);
            }
        }
    }

    fn split_child(&self, parent_id: NodeId, child_idx: usize) {
        let parent = self.node(parent_id);
        let child_id = unsafe { parent.read_child(child_idx).expect("child must exist") };
        let child = self.node(child_id);

        let (mid_key, mid_val, right_id) = self.allocate_and_distribute(child_id);

        unsafe {
            let p_len = parent.read_len();

            parent.shift_keys_right(child_idx, p_len - child_idx);
            parent.shift_values_right(child_idx, p_len - child_idx);
            parent.shift_children_right(child_idx + 1, p_len - child_idx);

            parent.write_key(child_idx, mid_key);
            parent.write_value(child_idx, mid_val);
            parent.write_child(child_idx + 1, Some(right_id));
            parent.write_len(p_len + 1);
        }

        child.write_unlock();
    }

    fn allocate_and_distribute(&self, left_id: NodeId) -> (K, V, NodeId) {
        let left = self.node(left_id);
        let is_leaf = unsafe { left.read_is_leaf() };

        let right_id = self.new_node(is_leaf);
        let right = self.node(right_id);

        let (mid_key, mid_val) = unsafe {
            let left_len = left.read_len();
            let mid = left_len / 2;
            let count = left_len - 1 - mid;

            right.copy_keys_from(0, left, mid + 1, count);
            right.copy_values_from(0, left, mid + 1, count);

            if !is_leaf {
                right.copy_children_from(0, left, mid + 1, count + 1);
            }

            let mk = left.read_key(mid);
            let mv = left.read_value(mid);

            left.write_len(mid);
            right.write_len(count);

            (mk, mv)
        };

        (mid_key, mid_val, right_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let tree = BTreeUnwind::<u64, u64>::new();

        tree.insert(10, 100);
        tree.insert(20, 200);
        tree.insert(5, 50);

        assert_eq!(tree.search(10), Some(100));
        assert_eq!(tree.search(5), Some(50));
        assert_eq!(tree.search(20), Some(200));
        assert_eq!(tree.search(999), None);
    }

    #[test]
    fn test_many_inserts() {
        let tree = BTreeUnwind::<u64, u64>::new();
        for i in 0..1000u64 {
            tree.insert(i, i * 10);
        }
        for i in 0..1000u64 {
            assert_eq!(tree.search(i), Some(i * 10));
        }
    }

    #[test]
    fn test_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let tree = Arc::new(BTreeUnwind::<u64, u64>::new());

        let mut handles = vec![];
        for t in 0..4 {
            let tree_ref = tree.clone();
            handles.push(thread::spawn(move || {
                for i in (t * 250)..((t + 1) * 250) {
                    tree_ref.insert(i as u64, (i * 10) as u64);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(tree.len(), 1000);
    }

    #[test]
    fn test_exception_safety_no_lock_poisoning() {
        // Verify that version mismatch panics don't poison any state
        use std::sync::Arc;
        use std::thread;
        use std::sync::atomic::AtomicUsize;

        let tree = Arc::new(BTreeUnwind::<u64, u64>::new());
        let success_count = Arc::new(AtomicUsize::new(0));

        // Pre-populate
        for i in 0..100u64 {
            tree.insert(i, i);
        }

        let mut handles = vec![];

        // Readers that will cause many version mismatches
        for _ in 0..4 {
            let tree_ref = tree.clone();
            let count_ref = success_count.clone();
            handles.push(thread::spawn(move || {
                for i in 0..1000u64 {
                    let _ = tree_ref.search(i % 100);
                    count_ref.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        // Writers that cause the mismatches
        for t in 0..2 {
            let tree_ref = tree.clone();
            handles.push(thread::spawn(move || {
                for i in 0..500u64 {
                    tree_ref.insert((i + t * 500) % 100, i);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // All operations completed without deadlock or poisoned state
        assert!(success_count.load(Ordering::Relaxed) > 0);
    }
}
