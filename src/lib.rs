#![allow(clippy::manual_is_multiple_of)]

use std::cell::UnsafeCell;
use std::marker::Sync;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

const B: usize = 6;
const CAPACITY: usize = 2 * B - 1;
const MIN_KEYS: usize = B - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(u32);

#[derive(Debug)]
pub struct Node<K, V>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    version: AtomicU64,

    keys: UnsafeCell<[K; CAPACITY]>,
    values: UnsafeCell<[V; CAPACITY]>,
    children: UnsafeCell<[Option<NodeId>; CAPACITY + 1]>,
    len: UnsafeCell<usize>,
    is_leaf: UnsafeCell<bool>,
}

/// SAFETY: Node<K, V> is Sync because concurrent access is mediated by the seqlock protocol:
///
/// 1. **Writers** must hold the write lock (version is odd) before accessing any `UnsafeCell`
///    field. The lock is acquired via `write_lock()` and released via `write_unlock()`.
///
/// 2. **Readers** use Optimistic Lock Coupling (OLC):
///    - Snapshot the version (must be even, meaning unlocked)
///    - Read data from `UnsafeCell` fields
///    - Validate that version hasn't changed
///    - Retry the entire operation if validation fails
///
/// 3. **Initialization**: New nodes are fully initialized (including `is_leaf`) before
///    their `NodeId` is made visible to other threads. The `Release` ordering on
///    `version.store(0)` in `new_node` synchronizes with the `Acquire` on version reads.
///
/// 4. **Generic bounds**: K and V are required to be `Copy`, ensuring all data lives inline
///    in the arrays (no heap pointers that could be invalidated by concurrent access).
///
/// These invariants ensure that data races cannot occur: either a reader sees a consistent
/// snapshot (validation succeeds) or it detects concurrent modification (validation fails
/// and triggers retry).
///
/// # Known Limitation (Seqlock UB)
///
/// This implementation uses the seqlock pattern, which technically violates Rust's strict
/// aliasing rules: readers create `&T` references while a writer may concurrently hold
/// `&mut T`. This is a documented open problem in Rust's memory model for seqlocks
/// (see rust-lang RFCs and discussions on `UnsafeCell` semantics).
///
/// The implementation is **empirically sound** on all major architectures (x86, ARM) because:
/// - K and V are `Copy` types with no internal pointers to invalidate
/// - Word-sized reads/writes are atomic on modern CPUs (no torn reads)
/// - The validation step detects any inconsistency from concurrent modification
/// - Miri will flag this as UB, but real hardware behaves correctly
///
/// A fully sound implementation would require `AtomicCell<T>` or similar primitives
/// that don't yet exist in stable Rust.
unsafe impl<K, V> Sync for Node<K, V>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{}

impl<K, V> Node<K, V>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    pub fn new(is_leaf: bool) -> Self {
        Self {
            version: AtomicU64::new(0),
            len: UnsafeCell::new(0),
            is_leaf: UnsafeCell::new(is_leaf),
            keys: UnsafeCell::new([K::default(); CAPACITY]),
            values: UnsafeCell::new([V::default(); CAPACITY]),
            children: UnsafeCell::new([None; CAPACITY + 1]),
        }
    }

    fn new_uninit() -> Self {
        Self {
            version: AtomicU64::new(u64::MAX),
            len: UnsafeCell::new(0),
            is_leaf: UnsafeCell::new(true),
            keys: UnsafeCell::new([K::default(); CAPACITY]),
            values: UnsafeCell::new([V::default(); CAPACITY]),
            children: UnsafeCell::new([None; CAPACITY + 1]),
        }
    }

    pub fn read_version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    pub fn validate(&self, start_version: u64) -> bool {
        let current = self.version.load(Ordering::Acquire);
        current == start_version && current % 2 == 0
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
}

pub struct BTree<K, V>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    pages: UnsafeCell<Vec<Node<K, V>>>,
    next_free_idx: AtomicUsize,
    root_id: AtomicU32,
    capacity: usize,
    entry_count: AtomicUsize,
}

/// SAFETY: BTree<K, V> is Sync because:
///
/// 1. **`pages` (UnsafeCell<Vec<Node<K, V>>>)**: The Vec itself is never resized after construction
///    (pre-allocated arena). Individual Node access is synchronized via each Node's seqlock.
///
/// 2. **`next_free_idx` (AtomicUsize)**: Atomic, inherently thread-safe. Used for allocating
///    new nodes with `fetch_add`.
///
/// 3. **`root_id` (AtomicU32)**: Atomic, inherently thread-safe. Root changes are protected
///    by locking the old root before updating this field.
///
/// The combination of per-node seqlocks and atomic root/allocation indices ensures that
/// concurrent operations are correctly synchronized.
unsafe impl<K, V> Sync for BTree<K, V>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{}

/// SAFETY: BTree<K, V> is Send because all its fields can be safely transferred between threads:
/// - `pages`: Vec<Node<K, V>> where Node<K, V> is Sync (seqlock-protected)
/// - `next_free_idx`: AtomicUsize (inherently Send)
/// - `root_id`: AtomicU32 (inherently Send)
unsafe impl<K, V> Send for BTree<K, V>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{}

impl<K, V> Default for BTree<K, V>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> BTree<K, V>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    const DEFAULT_CAPACITY: usize = 10_000;

    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    pub fn with_capacity(max_nodes: usize) -> Self {
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

        BTree {
            pages: UnsafeCell::new(pages),
            next_free_idx: AtomicUsize::new(1),
            root_id: AtomicU32::new(0),
            capacity: max_nodes,
            entry_count: AtomicUsize::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.entry_count.load(Ordering::Relaxed)
    }
    
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains_key(&self, key: K) -> bool {
        self.search(key).is_some()
    }

    pub fn clear(&self) {
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

        let node0 = self.node(NodeId(0));
        if root_id.0 != 0 {
            node0.write_lock();
        }

        unsafe {
            *node0.keys.get() = [K::default(); CAPACITY];
            *node0.values.get() = [V::default(); CAPACITY];
            *node0.children.get() = [None; CAPACITY + 1];
            *node0.len.get() = 0;
            *node0.is_leaf.get() = true;
        }

        self.entry_count.store(0, Ordering::Relaxed);
        self.next_free_idx.store(1, Ordering::Relaxed);
        self.root_id.store(0, Ordering::Release);

        if root_id.0 != 0 {
            node0.write_unlock();
        }
        self.node(root_id).write_unlock();
    }

    fn node(&self, id: NodeId) -> &Node<K, V> {
        unsafe {
            let ptr = self.pages.get();
            let slice = &*ptr;
            &slice[id.0 as usize]
        }
    }

    fn new_node(&self, is_leaf: bool) -> NodeId {
        let idx = self.next_free_idx.fetch_add(1, Ordering::Relaxed);

        if idx >= self.capacity {
            panic!("Arena OOM: Tree exceeded capacity of {} nodes. Use BTree::with_capacity() for larger trees.", self.capacity);
        }

        if idx + 1 < self.capacity {
            let next_node = self.node(NodeId((idx + 1) as u32));
            Self::prefetch_node(next_node);
        }

        let n = self.node(NodeId(idx as u32));

        unsafe {
            *n.keys.get() = [K::default(); CAPACITY];
            *n.values.get() = [V::default(); CAPACITY];
            *n.children.get() = [None; CAPACITY + 1];
            *n.len.get() = 0;
            *n.is_leaf.get() = is_leaf;
            n.version.store(0, Ordering::Release);
        }
        NodeId(idx as u32)
    }

    #[inline]
    fn prefetch_node(node: &Node<K, V>) {
        let ptr = node.keys.get() as *const i8;

        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_T0);
        }

        #[cfg(not(target_arch = "x86_64"))]
        unsafe {
            std::ptr::read_volatile(ptr);
        }
    }

    pub fn search(&self, key: K) -> Option<V> {
        'restart: loop {
            let mut current_id = NodeId(self.root_id.load(Ordering::Acquire));

            loop {
                let node = self.node(current_id);

                let start_version = node.read_version();
                if start_version % 2 != 0 {
                    continue 'restart;
                }

                let read_result = unsafe {
                    let len = *node.len.get();
                    let keys = &*node.keys.get();

                    match keys[0..len].binary_search(&key) {
                        Ok(idx) => {
                            let values = &*node.values.get();
                            Ok(Some(values[idx]))
                        }
                        Err(idx) => {
                            if *node.is_leaf.get() {
                                Ok(None)
                            } else {
                                let children = &*node.children.get();
                                Err(children[idx])
                            }
                        }
                    }
                };

                if !node.validate(start_version) {
                    continue 'restart;
                }

                match read_result {
                    Ok(result) => return result,
                    Err(child_opt) => {
                        match child_opt {
                            Some(child_id) => current_id = child_id,
                            None => continue 'restart,
                        }
                    }
                }
            }
        }
    }

    pub fn insert(&self, key: K, value: V) {
        if let Some(true) = self.insert_optimistic(key, value) {
            return;
        }

        self.insert_pessimistic(key, value);
    }

    fn insert_optimistic(&self, key: K, value: V) -> Option<bool> {
        let mut current_id = NodeId(self.root_id.load(Ordering::Acquire));

        loop {
            let node = self.node(current_id);
            let start_version = node.read_version();

            let next_step = unsafe {
                let len = *node.len.get();
                let keys = &*node.keys.get();

                if *node.is_leaf.get() {
                    Err(())
                } else {
                    let idx = match keys[0..len].binary_search(&key) {
                        Ok(i) => i + 1,
                        Err(i) => i,
                    };
                    Ok((*node.children.get())[idx].unwrap())
                }
            };

            if !node.validate(start_version) {
                return Some(false);
            }

            match next_step {
                Ok(child_id) => current_id = child_id,
                Err(_) => {
                    node.write_lock();

                    let is_full = unsafe { *node.len.get() >= CAPACITY };

                    if is_full {
                        node.write_unlock();
                        return None;
                    }

                    let is_new = unsafe {
                        let len = *node.len.get();
                        let keys = &mut *node.keys.get();
                        let vals = &mut *node.values.get();

                        match keys[0..len].binary_search(&key) {
                            Ok(idx) => {
                                vals[idx] = value;
                                false
                            }
                            Err(idx) => {
                                keys.copy_within(idx..len, idx + 1);
                                vals.copy_within(idx..len, idx + 1);
                                keys[idx] = key;
                                vals[idx] = value;
                                *node.len.get() += 1;
                                true
                            }
                        }
                    };

                    node.write_unlock();
                    if is_new {
                        self.entry_count.fetch_add(1, Ordering::Relaxed);
                    }
                    return Some(true);
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
        let root_full = unsafe { *root.len.get() == CAPACITY };

        let current_root_id = if root_full {
            let new_root_id = self.new_node(false);
            let new_root = self.node(new_root_id);

            unsafe {
                (*new_root.children.get())[0] = Some(root_id);
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
        let is_leaf = unsafe { *node.is_leaf.get() };

        if is_leaf {
            let is_new = unsafe {
                let len = *node.len.get();
                let keys = &mut *node.keys.get();
                let vals = &mut *node.values.get();

                match keys[0..len].binary_search(&key) {
                    Ok(idx) => {
                        vals[idx] = value;
                        false
                    }
                    Err(idx) => {
                        keys.copy_within(idx..len, idx + 1);
                        vals.copy_within(idx..len, idx + 1);
                        keys[idx] = key;
                        vals[idx] = value;
                        *node.len.get() += 1;
                        true
                    }
                }
            };
            node.write_unlock();
            if is_new {
                self.entry_count.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            let keys_ptr = node.keys.get();
            let children_ptr = node.children.get();

            let idx = unsafe {
                let len = *node.len.get();
                let keys = &*keys_ptr;
                match keys[0..len].binary_search(&key) {
                    Ok(i) => i + 1,
                    Err(i) => i,
                }
            };

            let child_id = unsafe { (*children_ptr)[idx].expect("Structure broken") };

            let child = self.node(child_id);
            child.write_lock();
            let child_full = unsafe { *child.len.get() == CAPACITY };

            if child_full {
                self.split_child(node_id, idx);

                let go_right = unsafe { key > (*keys_ptr)[idx] };

                if go_right {
                    let right_id = unsafe { (*children_ptr)[idx + 1].unwrap() };
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

        let child_id = unsafe { (*parent.children.get())[child_idx].unwrap() };
        let child = self.node(child_id);

        let (mid_key, mid_val, right_id) = self.allocate_and_distribute(child_id);

        unsafe {
            let p_len = *parent.len.get();
            let keys = &mut *parent.keys.get();
            let vals = &mut *parent.values.get();
            let children = &mut *parent.children.get();

            keys.copy_within(child_idx..p_len, child_idx + 1);
            vals.copy_within(child_idx..p_len, child_idx + 1);
            children.copy_within(child_idx + 1..p_len + 1, child_idx + 2);

            keys[child_idx] = mid_key;
            vals[child_idx] = mid_val;
            children[child_idx + 1] = Some(right_id);
            *parent.len.get() += 1;
        }

        child.write_unlock();
    }

    fn allocate_and_distribute(&self, left_id: NodeId) -> (K, V, NodeId) {
        let left = self.node(left_id);
        let is_leaf = unsafe { *left.is_leaf.get() };

        let right_id = self.new_node(is_leaf);
        let right = self.node(right_id);

        let (mid_key, mid_val) = unsafe {
            let left_len = *left.len.get();
            let mid = left_len / 2;
            let count = left_len - 1 - mid;

            let l_keys = &mut *left.keys.get();
            let l_vals = &mut *left.values.get();
            let l_children = &mut *left.children.get();

            let r_keys = &mut *right.keys.get();
            let r_vals = &mut *right.values.get();
            let r_children = &mut *right.children.get();

            r_keys[0..count].copy_from_slice(&l_keys[mid + 1..mid + 1 + count]);
            r_vals[0..count].copy_from_slice(&l_vals[mid + 1..mid + 1 + count]);

            if !is_leaf {
                r_children[0..=count].copy_from_slice(&l_children[mid + 1..mid + 2 + count]);
            }

            let mk = l_keys[mid];
            let mv = l_vals[mid];

            *left.len.get() = mid;
            *right.len.get() = count;

            (mk, mv)
        };

        (mid_key, mid_val, right_id)
    }

    pub fn delete(&self, key: K) -> Option<V> {
        if let Some(result) = self.delete_optimistic(key) {
            if result.is_some() {
                self.entry_count.fetch_sub(1, Ordering::Relaxed);
            }
            return result;
        }

        let result = self.delete_pessimistic(key);
        if result.is_some() {
            self.entry_count.fetch_sub(1, Ordering::Relaxed);
        }
        result
    }

    fn delete_optimistic(&self, key: K) -> Option<Option<V>> {
        let mut current_id = NodeId(self.root_id.load(Ordering::Acquire));

        loop {
            let node = self.node(current_id);
            let start_version = node.read_version();

            if start_version % 2 != 0 {
                return None;
            }

            let next_step = unsafe {
                let len = *node.len.get();
                let keys = &*node.keys.get();
                let is_leaf = *node.is_leaf.get();

                if is_leaf {
                    match keys[0..len].binary_search(&key) {
                        Ok(idx) => Err(Some(idx)),
                        Err(_) => Err(None),
                    }
                } else {
                    match keys[0..len].binary_search(&key) {
                        Ok(_) => {
                            return None;
                        }
                        Err(idx) => Ok((*node.children.get())[idx]),
                    }
                }
            };

            if !node.validate(start_version) {
                return None;
            }

            match next_step {
                Ok(Some(child_id)) => current_id = child_id,
                Ok(None) => return None,
                Err(Some(idx)) => {
                    node.write_lock();

                    let can_delete = unsafe {
                        let len = *node.len.get();
                        let is_leaf = *node.is_leaf.get();
                        let keys = &*node.keys.get();
                        let is_root = current_id.0 == self.root_id.load(Ordering::Relaxed);

                        is_leaf
                            && (len > MIN_KEYS || is_root)
                            && matches!(keys[0..len].binary_search(&key), Ok(found_idx) if found_idx == idx)
                    };

                    if !can_delete {
                        node.write_unlock();
                        return None;
                    }

                    let value = self.remove_from_leaf(current_id, idx);
                    node.write_unlock();
                    return Some(Some(value));
                }
                Err(None) => {
                    return Some(None);
                }
            }
        }
    }

    fn delete_pessimistic(&self, key: K) -> Option<V> {
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

        let result = self.delete_from_subtree(root_id, key);

        let root = self.node(root_id);
        let (should_shrink, new_root) = unsafe {
            let len = *root.len.get();
            let is_leaf = *root.is_leaf.get();
            if len == 0 && !is_leaf {
                (true, (*root.children.get())[0].unwrap())
            } else {
                (false, root_id)
            }
        };

        if should_shrink {
            self.root_id.store(new_root.0, Ordering::Release);
        }

        root.write_unlock();
        result
    }

    fn delete_from_subtree(&self, node_id: NodeId, key: K) -> Option<V> {
        let node = self.node(node_id);

        let (is_leaf, len) = unsafe { (*node.is_leaf.get(), *node.len.get()) };

        let keys_ptr = node.keys.get();
        let search_result = unsafe {
            let keys = &*keys_ptr;
            keys[0..len].binary_search(&key)
        };

        if is_leaf {
            match search_result {
                Ok(idx) => Some(self.remove_from_leaf(node_id, idx)),
                Err(_) => None,
            }
        } else {
            match search_result {
                Ok(idx) => {
                    Some(self.remove_from_internal(node_id, idx, key))
                }
                Err(idx) => {
                    let child_id = self.ensure_child_can_lose_key(node_id, idx);

                    let child = self.node(child_id);
                    child.write_lock();

                    let result = self.delete_from_subtree(child_id, key);

                    child.write_unlock();
                    result
                }
            }
        }
    }

    fn ensure_child_can_lose_key(&self, parent_id: NodeId, child_idx: usize) -> NodeId {
        let parent = self.node(parent_id);

        let child_id = unsafe { (*parent.children.get())[child_idx].unwrap() };
        let child = self.node(child_id);
        child.write_lock();

        let child_len = unsafe { *child.len.get() };

        if child_len > MIN_KEYS {
            child.write_unlock();
            return child_id;
        }

        let parent_len = unsafe { *parent.len.get() };

        if child_idx > 0 {
            let left_id = unsafe { (*parent.children.get())[child_idx - 1].unwrap() };
            let left = self.node(left_id);
            left.write_lock();

            let left_len = unsafe { *left.len.get() };
            if left_len > MIN_KEYS {
                self.borrow_from_left(parent_id, child_idx);
                left.write_unlock();
                child.write_unlock();
                return child_id;
            }

            let merged_id = self.merge_with_left(parent_id, child_idx);
            left.write_unlock();
            child.write_unlock();
            return merged_id;
        }

        if child_idx < parent_len {
            let right_id = unsafe { (*parent.children.get())[child_idx + 1].unwrap() };
            let right = self.node(right_id);
            right.write_lock();

            let right_len = unsafe { *right.len.get() };
            if right_len > MIN_KEYS {
                self.borrow_from_right(parent_id, child_idx);
                right.write_unlock();
                child.write_unlock();
                return child_id;
            }

            let merged_id = self.merge_with_right(parent_id, child_idx);
            right.write_unlock();
            child.write_unlock();
            return merged_id;
        }

        child.write_unlock();
        child_id
    }

    fn borrow_from_left(&self, parent_id: NodeId, child_idx: usize) {
        let parent = self.node(parent_id);
        let child_id = unsafe { (*parent.children.get())[child_idx].unwrap() };
        let left_id = unsafe { (*parent.children.get())[child_idx - 1].unwrap() };

        let child = self.node(child_id);
        let left = self.node(left_id);

        unsafe {
            let p_keys = &mut *parent.keys.get();
            let p_vals = &mut *parent.values.get();

            let c_keys = &mut *child.keys.get();
            let c_vals = &mut *child.values.get();
            let c_children = &mut *child.children.get();
            let c_len = *child.len.get();

            let l_keys = &mut *left.keys.get();
            let l_vals = &mut *left.values.get();
            let l_children = &mut *left.children.get();
            let l_len = *left.len.get();

            c_keys.copy_within(0..c_len, 1);
            c_vals.copy_within(0..c_len, 1);
            if !*child.is_leaf.get() {
                c_children.copy_within(0..=c_len, 1);
            }

            c_keys[0] = p_keys[child_idx - 1];
            c_vals[0] = p_vals[child_idx - 1];

            p_keys[child_idx - 1] = l_keys[l_len - 1];
            p_vals[child_idx - 1] = l_vals[l_len - 1];

            if !*child.is_leaf.get() {
                c_children[0] = l_children[l_len];
            }

            *child.len.get() += 1;
            *left.len.get() -= 1;
        }
    }

    fn borrow_from_right(&self, parent_id: NodeId, child_idx: usize) {
        let parent = self.node(parent_id);
        let child_id = unsafe { (*parent.children.get())[child_idx].unwrap() };
        let right_id = unsafe { (*parent.children.get())[child_idx + 1].unwrap() };

        let child = self.node(child_id);
        let right = self.node(right_id);

        unsafe {
            let p_keys = &mut *parent.keys.get();
            let p_vals = &mut *parent.values.get();

            let c_keys = &mut *child.keys.get();
            let c_vals = &mut *child.values.get();
            let c_children = &mut *child.children.get();
            let c_len = *child.len.get();

            let r_keys = &mut *right.keys.get();
            let r_vals = &mut *right.values.get();
            let r_children = &mut *right.children.get();
            let r_len = *right.len.get();

            c_keys[c_len] = p_keys[child_idx];
            c_vals[c_len] = p_vals[child_idx];

            if !*child.is_leaf.get() {
                c_children[c_len + 1] = r_children[0];
            }

            p_keys[child_idx] = r_keys[0];
            p_vals[child_idx] = r_vals[0];

            r_keys.copy_within(1..r_len, 0);
            r_vals.copy_within(1..r_len, 0);
            if !*child.is_leaf.get() {
                r_children.copy_within(1..=r_len, 0);
            }

            *child.len.get() += 1;
            *right.len.get() -= 1;
        }
    }

    fn merge_with_left(&self, parent_id: NodeId, child_idx: usize) -> NodeId {
        let parent = self.node(parent_id);
        let child_id = unsafe { (*parent.children.get())[child_idx].unwrap() };
        let left_id = unsafe { (*parent.children.get())[child_idx - 1].unwrap() };

        let child = self.node(child_id);
        let left = self.node(left_id);

        unsafe {
            let p_keys = &mut *parent.keys.get();
            let p_vals = &mut *parent.values.get();
            let p_children = &mut *parent.children.get();
            let p_len = *parent.len.get();

            let c_keys = &*child.keys.get();
            let c_vals = &*child.values.get();
            let c_children = &*child.children.get();
            let c_len = *child.len.get();

            let l_keys = &mut *left.keys.get();
            let l_vals = &mut *left.values.get();
            let l_children = &mut *left.children.get();
            let l_len = *left.len.get();

            l_keys[l_len] = p_keys[child_idx - 1];
            l_vals[l_len] = p_vals[child_idx - 1];

            l_keys[l_len + 1..l_len + 1 + c_len].copy_from_slice(&c_keys[0..c_len]);
            l_vals[l_len + 1..l_len + 1 + c_len].copy_from_slice(&c_vals[0..c_len]);

            if !*child.is_leaf.get() {
                l_children[l_len + 1..l_len + 2 + c_len].copy_from_slice(&c_children[0..=c_len]);
            }

            *left.len.get() = l_len + 1 + c_len;

            p_keys.copy_within(child_idx..p_len, child_idx - 1);
            p_vals.copy_within(child_idx..p_len, child_idx - 1);
            p_children.copy_within(child_idx + 1..=p_len, child_idx);
            *parent.len.get() -= 1;
        }

        left_id
    }

    fn merge_with_right(&self, parent_id: NodeId, child_idx: usize) -> NodeId {
        let parent = self.node(parent_id);
        let child_id = unsafe { (*parent.children.get())[child_idx].unwrap() };
        let right_id = unsafe { (*parent.children.get())[child_idx + 1].unwrap() };

        let child = self.node(child_id);
        let right = self.node(right_id);

        unsafe {
            let p_keys = &mut *parent.keys.get();
            let p_vals = &mut *parent.values.get();
            let p_children = &mut *parent.children.get();
            let p_len = *parent.len.get();

            let c_keys = &mut *child.keys.get();
            let c_vals = &mut *child.values.get();
            let c_children = &mut *child.children.get();
            let c_len = *child.len.get();

            let r_keys = &*right.keys.get();
            let r_vals = &*right.values.get();
            let r_children = &*right.children.get();
            let r_len = *right.len.get();

            c_keys[c_len] = p_keys[child_idx];
            c_vals[c_len] = p_vals[child_idx];

            c_keys[c_len + 1..c_len + 1 + r_len].copy_from_slice(&r_keys[0..r_len]);
            c_vals[c_len + 1..c_len + 1 + r_len].copy_from_slice(&r_vals[0..r_len]);

            if !*child.is_leaf.get() {
                c_children[c_len + 1..c_len + 2 + r_len].copy_from_slice(&r_children[0..=r_len]);
            }

            *child.len.get() = c_len + 1 + r_len;

            p_keys.copy_within(child_idx + 1..p_len, child_idx);
            p_vals.copy_within(child_idx + 1..p_len, child_idx);
            p_children.copy_within(child_idx + 2..=p_len, child_idx + 1);
            *parent.len.get() -= 1;
        }

        child_id
    }

    fn remove_from_internal(&self, node_id: NodeId, idx: usize, key: K) -> V {
        let node = self.node(node_id);

        let old_value = unsafe { (*node.values.get())[idx] };

        let left_child_id = self.ensure_child_can_lose_key(node_id, idx);

        let current_len = unsafe { *node.len.get() };

        let key_still_here = unsafe {
            let keys = &*node.keys.get();
            idx < current_len && keys[idx] == key
        };

        if key_still_here {
            let (pred_key, pred_val) = self.delete_max_from_subtree(left_child_id);

            unsafe {
                (*node.keys.get())[idx] = pred_key;
                (*node.values.get())[idx] = pred_val;
            }
        } else {
            let child = self.node(left_child_id);
            child.write_lock();
            let _ = self.delete_from_subtree(left_child_id, key);
            child.write_unlock();
        }

        old_value
    }

    fn delete_max_from_subtree(&self, node_id: NodeId) -> (K, V) {
        let node = self.node(node_id);
        node.write_lock();

        let is_leaf = unsafe { *node.is_leaf.get() };

        if is_leaf {
            let len = unsafe { *node.len.get() };
            let result = unsafe {
                let keys = &*node.keys.get();
                let vals = &*node.values.get();
                (keys[len - 1], vals[len - 1])
            };
            unsafe { *node.len.get() -= 1 };
            node.write_unlock();
            result
        } else {
            let len = unsafe { *node.len.get() };
            let rightmost_idx = len;

            let child_id = self.ensure_child_can_lose_key(node_id, rightmost_idx);

            let result = self.delete_max_from_subtree(child_id);

            node.write_unlock();
            result
        }
    }

    fn remove_from_leaf(&self, node_id: NodeId, idx: usize) -> V {
        let node = self.node(node_id);

        unsafe {
            let len = *node.len.get();
            let keys = &mut *node.keys.get();
            let vals = &mut *node.values.get();

            let value = vals[idx];

            keys.copy_within(idx + 1..len, idx);
            vals.copy_within(idx + 1..len, idx);

            *node.len.get() -= 1;

            value
        }
    }
}

impl<K, V> BTree<K, V>
where
    K: Copy + Ord + Default + std::fmt::Debug,
    V: Copy + Default,
{
    pub fn print(&self) {
        println!("=== B-Tree Structure (Arena + OLC) ===");
        self.print_subtree(NodeId(self.root_id.load(Ordering::Acquire)), 0);
        println!("======================================");
    }

    fn print_subtree(&self, node_id: NodeId, depth: usize) {
        let node = self.node(node_id);
        let indent = "  ".repeat(depth);
        unsafe {
            let keys_slice = &*node.keys.get();
            let len = *node.len.get();
            let is_leaf = *node.is_leaf.get();
            println!(
                "{}Node[{}] (Leaf: {}) Keys: {:?}",
                indent,
                node_id.0,
                is_leaf,
                &keys_slice[0..len]
            );
            if !is_leaf {
                for i in 0..=*node.len.get() {
                    if let Some(child_id) = (*node.children.get())[i] {
                        self.print_subtree(child_id, depth + 1);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_safe_insert_search() {
        let tree = BTree::<u64, u64>::new();

        tree.insert(10, 100);
        tree.insert(20, 200);
        tree.insert(5, 50);

        assert_eq!(tree.search(10), Some(100));
        assert_eq!(tree.search(5), Some(50));
        assert_eq!(tree.search(20), Some(200));
        assert_eq!(tree.search(999), None);
    }

    #[test]
    fn test_splitting_logic() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..20u64 {
            tree.insert(i, i * 10);
        }
        for i in 0..20u64 {
            assert_eq!(tree.search(i), Some(i * 10));
        }
    }

    #[test]
    fn test_generic_with_i32() {
        let tree = BTree::<i32, i32>::new();
        tree.insert(-10, 100);
        tree.insert(0, 0);
        tree.insert(10, -100);

        assert_eq!(tree.search(-10), Some(100));
        assert_eq!(tree.search(0), Some(0));
        assert_eq!(tree.search(10), Some(-100));
        assert_eq!(tree.search(5), None);
    }

    #[test]
    fn test_len() {
        let tree = BTree::<u64, u64>::new();
        assert_eq!(tree.len(), 0);

        tree.insert(1, 10);
        assert_eq!(tree.len(), 1);

        tree.insert(2, 20);
        tree.insert(3, 30);
        assert_eq!(tree.len(), 3);

        tree.insert(2, 200);
        assert_eq!(tree.len(), 3);

        for i in 10..50u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 43);
    }

    #[test]
    fn test_is_empty() {
        let tree = BTree::<u64, u64>::new();
        assert!(tree.is_empty());

        tree.insert(1, 10);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_contains_key() {
        let tree = BTree::<u64, u64>::new();
        assert!(!tree.contains_key(1));

        tree.insert(1, 10);
        tree.insert(5, 50);
        tree.insert(10, 100);

        assert!(tree.contains_key(1));
        assert!(tree.contains_key(5));
        assert!(tree.contains_key(10));
        assert!(!tree.contains_key(2));
        assert!(!tree.contains_key(100));
    }

    #[test]
    fn test_clear() {
        let tree = BTree::<u64, u64>::new();

        for i in 0..50u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 50);
        assert!(!tree.is_empty());

        tree.clear();

        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
        assert!(!tree.contains_key(0));
        assert!(!tree.contains_key(25));

        tree.insert(100, 1000);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.search(100), Some(1000));
    }

    #[test]
    fn test_delete_single_key() {
        let tree = BTree::<u64, u64>::new();
        tree.insert(10, 100);
        assert_eq!(tree.delete(10), Some(100));
        assert_eq!(tree.search(10), None);
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_delete_nonexistent_key() {
        let tree = BTree::<u64, u64>::new();
        tree.insert(10, 100);
        assert_eq!(tree.delete(20), None);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.search(10), Some(100));
    }

    #[test]
    fn test_delete_from_empty_tree() {
        let tree = BTree::<u64, u64>::new();
        assert_eq!(tree.delete(10), None);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_delete_multiple_keys() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..10u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 10);

        assert_eq!(tree.delete(5), Some(50));
        assert_eq!(tree.len(), 9);
        assert_eq!(tree.search(5), None);

        assert_eq!(tree.delete(0), Some(0));
        assert_eq!(tree.len(), 8);

        assert_eq!(tree.delete(9), Some(90));
        assert_eq!(tree.len(), 7);

        for i in [1, 2, 3, 4, 6, 7, 8] {
            assert_eq!(tree.search(i), Some(i * 10));
        }
    }

    #[test]
    fn test_delete_all_keys() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..50u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 50);

        for i in 0..50u64 {
            assert_eq!(tree.delete(i), Some(i * 10), "Failed to delete key {}", i);
        }
        assert!(tree.is_empty());

        tree.insert(100, 1000);
        assert_eq!(tree.search(100), Some(1000));
    }

    #[test]
    fn test_delete_causes_underflow() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..30u64 {
            tree.insert(i, i * 10);
        }

        for i in 0..30u64 {
            let result = tree.delete(i);
            assert_eq!(result, Some(i * 10), "Failed to delete key {}", i);
            assert_eq!(tree.search(i), None);
        }
        assert!(tree.is_empty());
    }

    #[test]
    fn test_delete_reverse_order() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..50u64 {
            tree.insert(i, i * 10);
        }

        for i in (0..50u64).rev() {
            assert_eq!(tree.delete(i), Some(i * 10));
        }
        assert!(tree.is_empty());
    }

    #[test]
    fn test_delete_alternating() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..100u64 {
            tree.insert(i, i * 10);
        }

        for i in (0..100u64).step_by(2) {
            assert_eq!(tree.delete(i), Some(i * 10));
        }
        assert_eq!(tree.len(), 50);

        for i in (1..100u64).step_by(2) {
            assert_eq!(tree.search(i), Some(i * 10));
        }

        for i in (1..100u64).step_by(2) {
            assert_eq!(tree.delete(i), Some(i * 10));
        }
        assert!(tree.is_empty());
    }

    #[test]
    fn test_delete_then_insert() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..20u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 20);

        assert_eq!(tree.delete(5), Some(50));
        assert_eq!(tree.search(5), None, "Key 5 should be deleted");
        assert_eq!(tree.len(), 19);

        assert_eq!(tree.delete(10), Some(100));
        assert_eq!(tree.search(10), None, "Key 10 should be deleted");
        assert_eq!(tree.len(), 18);

        assert_eq!(tree.delete(15), Some(150));
        assert_eq!(tree.search(15), None, "Key 15 should be deleted");
        assert_eq!(tree.len(), 17);

        tree.insert(5, 500);
        assert_eq!(tree.len(), 18);

        tree.insert(10, 1000);
        assert_eq!(tree.len(), 19);

        tree.insert(15, 1500);
        assert_eq!(tree.len(), 20);

        assert_eq!(tree.search(5), Some(500));
        assert_eq!(tree.search(10), Some(1000));
        assert_eq!(tree.search(15), Some(1500));
    }

    #[test]
    fn test_delete_large_tree() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..1000u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 1000);

        for i in (0..1000u64).step_by(2) {
            assert_eq!(tree.delete(i), Some(i * 10));
        }
        assert_eq!(tree.len(), 500);

        for i in (1..1000u64).step_by(2) {
            assert_eq!(tree.search(i), Some(i * 10));
        }
    }

    #[test]
    fn test_concurrent_delete() {
        use std::sync::Arc;
        use std::thread;

        let tree = Arc::new(BTree::<u64, u64>::new());

        for i in 0..1000u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 1000);

        let mut handles = vec![];
        for t in 0..4 {
            let tree_ref = tree.clone();
            handles.push(thread::spawn(move || {
                for i in (t * 250)..((t + 1) * 250) {
                    tree_ref.delete(i as u64);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert!(tree.is_empty());
    }

    #[test]
    fn test_concurrent_insert_delete() {
        use std::sync::Arc;
        use std::thread;

        let tree = Arc::new(BTree::<u64, u64>::new());

        for i in 0..500u64 {
            tree.insert(i, i * 10);
        }

        let mut handles = vec![];

        for t in 0..2 {
            let tree_ref = tree.clone();
            handles.push(thread::spawn(move || {
                for i in (500 + t * 250)..(500 + (t + 1) * 250) {
                    tree_ref.insert(i as u64, (i * 10) as u64);
                }
            }));
        }

        for t in 0..2 {
            let tree_ref = tree.clone();
            handles.push(thread::spawn(move || {
                for i in (t * 250)..((t + 1) * 250) {
                    tree_ref.delete(i as u64);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(tree.len(), 500);

        for i in 500..1000u64 {
            assert_eq!(tree.search(i), Some(i * 10));
        }

        for i in 0..500u64 {
            assert_eq!(tree.search(i), None);
        }
    }

    #[test]
    fn test_concurrent_search_delete() {
        use std::sync::Arc;
        use std::thread;
        use std::sync::atomic::{AtomicBool, Ordering};

        let tree = Arc::new(BTree::<u64, u64>::new());
        let done = Arc::new(AtomicBool::new(false));

        for i in 0..1000u64 {
            tree.insert(i, i * 10);
        }

        let mut handles = vec![];

        for _ in 0..2 {
            let tree_ref = tree.clone();
            let done_ref = done.clone();
            handles.push(thread::spawn(move || {
                while !done_ref.load(Ordering::Relaxed) {
                    for i in 0..1000u64 {
                        let _ = tree_ref.search(i);
                    }
                }
            }));
        }

        let tree_ref = tree.clone();
        let done_ref = done.clone();
        handles.push(thread::spawn(move || {
            for i in 0..1000u64 {
                tree_ref.delete(i);
            }
            done_ref.store(true, Ordering::Relaxed);
        }));

        for h in handles {
            h.join().unwrap();
        }

        assert!(tree.is_empty());
    }
}