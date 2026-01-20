use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicUsize, Ordering};
use std::marker::Sync;

const B: usize = 6;
const CAPACITY: usize = 2 * B - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(u32);

#[derive(Debug)]
pub struct Node<K, V>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    pub version: AtomicU64,

    pub keys: UnsafeCell<[K; CAPACITY]>,
    pub values: UnsafeCell<[V; CAPACITY]>,
    pub children: UnsafeCell<[Option<NodeId>; CAPACITY + 1]>,
    pub len: UnsafeCell<usize>,
    pub is_leaf: UnsafeCell<bool>,
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
        current == start_version && (current % 2 == 0)
    }

    pub fn write_lock(&self) {
        let mut backoff = 0;
        loop {
            let v = self.version.load(Ordering::Relaxed);
            if v % 2 == 0 {
                if self.version.compare_exchange_weak(v, v + 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                    return;
                }
            }
            for _ in 0..(1 << backoff) { std::hint::spin_loop(); }
            if backoff < 6 { backoff += 1; }
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
            pages.push(Node::new_uninit());
        }

        BTree {
            pages: UnsafeCell::new(pages),
            next_free_idx: AtomicUsize::new(1),
            root_id: AtomicU32::new(0),
            capacity: max_nodes,
        }
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

    pub fn search(&self, key: K) -> Option<V> {
        'restart: loop {
            let mut current_id = NodeId(self.root_id.load(Ordering::Acquire));

            loop {
                let node = self.node(current_id);

                let start_version = node.read_version();

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
        if let Some(success) = self.insert_optimistic(key, value) {
            if success { return; }
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

                    unsafe {
                        let len = *node.len.get();
                        let keys = &mut *node.keys.get();
                        let vals = &mut *node.values.get();

                        match keys[0..len].binary_search(&key) {
                            Ok(idx) => vals[idx] = value,
                            Err(idx) => {
                                keys.copy_within(idx..len, idx + 1);
                                vals.copy_within(idx..len, idx + 1);
                                keys[idx] = key;
                                vals[idx] = value;
                                *node.len.get() += 1;
                            }
                        }
                    }

                    node.write_unlock();
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
            unsafe {
                let len = *node.len.get();
                let keys = &mut *node.keys.get();
                let vals = &mut *node.values.get();

                match keys[0..len].binary_search(&key) {
                    Ok(idx) => vals[idx] = value,
                    Err(idx) => {
                        keys.copy_within(idx..len, idx + 1);
                        vals.copy_within(idx..len, idx + 1);
                        keys[idx] = key;
                        vals[idx] = value;
                        *node.len.get() += 1;
                    }
                }
            }
            node.write_unlock();
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
}