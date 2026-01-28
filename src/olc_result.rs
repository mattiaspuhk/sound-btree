#![allow(unsafe_op_in_unsafe_fn)]

//! OLC Retry Strategy: Result Propagation
//!
//! This module implements Optimistic Lock Coupling retry via `Result<T, VersionMismatch>`.
//! When validation fails, we return `Err(VersionMismatch)` and propagate it up the call
//! stack using the `?` operator. The public API catches this at the top level and loops.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

#[derive(Debug, Clone, Copy)]
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

    #[inline]
    pub fn validate(&self, start_version: u64) -> Result<(), VersionMismatch> {
        std::sync::atomic::fence(Ordering::Acquire);
        let current = self.version.load(Ordering::Relaxed);
        if current == start_version && current % 2 == 0 {
            Ok(())
        } else {
            Err(VersionMismatch)
        }
    }

    #[inline]
    pub fn check_unlocked(&self, version: u64) -> Result<(), VersionMismatch> {
        if version % 2 == 0 {
            Ok(())
        } else {
            Err(VersionMismatch)
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

pub struct BTreeResult<K, V, const CAP: usize = DEFAULT_CAP, const CHILDREN_CAP: usize = DEFAULT_CHILDREN_CAP>
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

unsafe impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> Sync for BTreeResult<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{}

unsafe impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> Send for BTreeResult<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{}

impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> Default for BTreeResult<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> BTreeResult<K, V, CAP, CHILDREN_CAP>
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

        BTreeResult {
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

    /// Public search API - loops until success.
    /// The retry loop is ONLY here at the top level.
    pub fn search(&self, key: K) -> Option<V> {
        loop {
            match self.search_inner(key) {
                Ok(result) => return result,
                Err(VersionMismatch) => continue, // Retry
            }
        }
    }

    /// Inner search that propagates `VersionMismatch` via `?`.
    ///
    /// Each `?` operator compiles to approximately:
    /// ```ignore
    /// match result {
    ///     Ok(v) => v,
    ///     Err(e) => return Err(e),
    /// }
    /// ```
    ///
    /// This is a branch that the CPU must predict on every call frame.
    #[inline]
    fn search_inner(&self, key: K) -> Result<Option<V>, VersionMismatch> {
        let mut current_id = NodeId(self.root_id.load(Ordering::Acquire));

        loop {
            let node = self.node(current_id);

            let start_version = node.read_version();
            node.check_unlocked(start_version)?; // Branch 1: is it locked?

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

            node.validate(start_version)?; // Branch 2: did version change?

            match read_result {
                Ok(result) => return Ok(result),
                Err(child_opt) => {
                    match child_opt {
                        Some(child_id) => current_id = child_id,
                        None => return Err(VersionMismatch), // Child was None during read
                    }
                }
            }
        }
    }

    /// Public insert API.
    pub fn insert(&self, key: K, value: V) {
        loop {
            match self.insert_optimistic_inner(key, value) {
                Ok(true) => return,                    // Success
                Ok(false) => continue,                 // Retry optimistic
                Err(VersionMismatch) => continue,      // Version mismatch, retry
            }
        }
    }

    /// Optimistic insert that propagates errors via Result.
    #[inline]
    fn insert_optimistic_inner(&self, key: K, value: V) -> Result<bool, VersionMismatch> {
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
                    Ok(node.read_child(idx).ok_or(VersionMismatch)?)
                }
            };

            node.validate(start_version)?;

            match next_step {
                Ok(child_id) => current_id = child_id,
                Err(()) => {
                    let guard = NodeWriteGuard::new(node);

                    let current_version = guard.version.load(Ordering::Relaxed);
                    if current_version != start_version + 1 {
                        return Err(VersionMismatch);
                    }

                    let is_leaf = unsafe { guard.read_is_leaf() };
                    if !is_leaf {
                        return Err(VersionMismatch);
                    }

                    let is_full = unsafe { guard.read_len() >= CAP };
                    if is_full {
                        drop(guard);
                        self.insert_pessimistic(key, value);
                        return Ok(true);
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
                    return Ok(true);
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
        let tree = BTreeResult::<u64, u64>::new();

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
        let tree = BTreeResult::<u64, u64>::new();
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

        let tree = Arc::new(BTreeResult::<u64, u64>::new());

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
}
