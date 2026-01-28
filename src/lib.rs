#![allow(clippy::manual_is_multiple_of)]

pub mod olc_result;
pub mod olc_unwind;

use std::cell::UnsafeCell;
use std::marker::Sync;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

pub const DEFAULT_CAP: usize = 11;

pub const DEFAULT_CHILDREN_CAP: usize = 12;

/// Identifies a node within the B-tree's arena allocator.
///
/// Node IDs are indices into the pre-allocated node vector. They remain valid
/// for the lifetime of the BTree. Deleted nodes are not recycled (see Memory
/// Reclamation section in BTree docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(u32);

/// A node in the concurrent B-tree using Optimistic Lock Coupling (OLC).
///
/// # Thread Safety
///
/// `Node` is `Sync` despite containing `UnsafeCell` because thread-safe access
/// is guaranteed by the OLC protocol:
///
/// 1. **Writers** must acquire the write lock (making version odd) before
///    mutating any data. All writes use raw pointer operations to avoid
///    creating Rust references that could conflict with concurrent readers.
///
/// 2. **Readers** perform optimistic reads without locks:
///    - Read the version (must be even, meaning unlocked)
///    - Read data using `read_volatile` via raw pointers
///    - Validate that version hasn't changed
///    - Restart if validation fails
///
/// 3. **Synchronization** is provided by atomic operations on `version`:
///    - Writers use `Release` ordering when unlocking
///    - Readers use `Acquire` ordering when reading version
///    - An `Acquire` fence before validation ensures all reads complete
///
/// # Memory Model
///
/// - Even version: Node is unlocked and data is consistent
/// - Odd version: Node is locked; a write is in progress
/// - u64::MAX (odd): Uninitialized node; will be skipped by readers
#[derive(Debug)]
pub struct Node<K, V, const CAP: usize = DEFAULT_CAP, const CHILDREN_CAP: usize = DEFAULT_CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    /// Version counter for OLC. Even = unlocked, odd = locked.
    version: AtomicU64,

    keys: UnsafeCell<[K; CAP]>,
    values: UnsafeCell<[V; CAP]>,
    children: UnsafeCell<[Option<NodeId>; CHILDREN_CAP]>,
    len: UnsafeCell<usize>,
    is_leaf: UnsafeCell<bool>,
}

// SAFETY: Node is Sync because the OLC protocol ensures safe concurrent access:
// - Writers hold an exclusive lock (odd version) before mutating
// - Readers use raw pointers (no references) and validate version after reading
// - Atomic operations on `version` provide the necessary synchronization
// - All data access through UnsafeCell uses raw pointers, never Rust references
//   during concurrent access, avoiding Stacked Borrows violations
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

    /// Reads the current version with Acquire ordering.
    ///
    /// Used at the start of an optimistic read to establish a synchronization
    /// point with any prior Release operations (i.e., write unlocks).
    pub fn read_version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    /// Validates that the node hasn't been modified since `start_version`.
    ///
    /// Returns `true` if the version is unchanged AND the node is unlocked (even).
    /// The Acquire fence ensures all prior reads in this thread complete before
    /// we check the version, preventing the compiler from reordering reads past
    /// the validation.
    ///
    /// # OLC Protocol
    ///
    /// Typical usage:
    /// 1. `let v = node.read_version()` (Acquire)
    /// 2. Check `v % 2 == 0` (skip if locked)
    /// 3. Read data via raw pointers
    /// 4. `node.validate(v)` → restart if false
    pub fn validate(&self, start_version: u64) -> bool {
        std::sync::atomic::fence(Ordering::Acquire);
        let current = self.version.load(Ordering::Relaxed);
        current == start_version && current % 2 == 0
    }

    /// Acquires the write lock using compare-and-swap with exponential backoff.
    ///
    /// Spins until the version is even (unlocked) and we successfully increment
    /// it to odd. Uses Acquire ordering on success to synchronize with prior
    /// Release operations.
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

    /// Releases the write lock by incrementing version from odd to even.
    ///
    /// Uses Release ordering to ensure all prior writes in this thread are
    /// visible to subsequent Acquire operations (readers checking the version).
    pub fn write_unlock(&self) {
        self.version.fetch_add(1, Ordering::Release);
    }

    #[inline]
    unsafe fn read_len(&self) -> usize {
        unsafe { std::ptr::read_volatile(self.len.get()) }
    }

    #[inline]
    unsafe fn read_is_leaf(&self) -> bool {
        unsafe { std::ptr::read_volatile(self.is_leaf.get()) }
    }

    #[inline]
    unsafe fn read_key(&self, idx: usize) -> K {
        unsafe {
            let ptr = self.keys.get() as *const K;
            std::ptr::read_volatile(ptr.add(idx))
        }
    }

    #[inline]
    unsafe fn read_value(&self, idx: usize) -> V {
        unsafe {
            let ptr = self.values.get() as *const V;
            std::ptr::read_volatile(ptr.add(idx))
        }
    }

    #[inline]
    unsafe fn read_child(&self, idx: usize) -> Option<NodeId> {
        unsafe {
            let ptr = self.children.get() as *const Option<NodeId>;
            std::ptr::read_volatile(ptr.add(idx))
        }
    }

    // === Raw pointer write helpers ===
    // These avoid creating Rust references, preventing Stacked Borrows conflicts
    // with concurrent optimistic readers using raw pointer reads.

    /// Writes a key at the given index using raw pointers.
    ///
    /// # Safety
    /// Caller must hold the write lock and ensure idx < CAP.
    #[inline]
    unsafe fn write_key(&self, idx: usize, key: K) {
        unsafe {
            let ptr = self.keys.get() as *mut K;
            std::ptr::write(ptr.add(idx), key);
        }
    }

    /// Writes a value at the given index using raw pointers.
    ///
    /// # Safety
    /// Caller must hold the write lock and ensure idx < CAP.
    #[inline]
    unsafe fn write_value(&self, idx: usize, value: V) {
        unsafe {
            let ptr = self.values.get() as *mut V;
            std::ptr::write(ptr.add(idx), value);
        }
    }

    /// Writes a child pointer at the given index using raw pointers.
    ///
    /// # Safety
    /// Caller must hold the write lock and ensure idx < CHILDREN_CAP.
    #[inline]
    unsafe fn write_child(&self, idx: usize, child: Option<NodeId>) {
        unsafe {
            let ptr = self.children.get() as *mut Option<NodeId>;
            std::ptr::write(ptr.add(idx), child);
        }
    }

    /// Writes the length using raw pointers.
    ///
    /// # Safety
    /// Caller must hold the write lock.
    #[inline]
    unsafe fn write_len(&self, len: usize) {
        unsafe {
            std::ptr::write(self.len.get(), len);
        }
    }

    /// Shifts keys right by one position starting at `start` for `count` elements.
    ///
    /// # Safety
    /// Caller must hold the write lock and ensure start + count < CAP.
    #[inline]
    unsafe fn shift_keys_right(&self, start: usize, count: usize) {
        if count == 0 {
            return;
        }
        unsafe {
            let ptr = self.keys.get() as *mut K;
            std::ptr::copy(ptr.add(start), ptr.add(start + 1), count);
        }
    }

    /// Shifts values right by one position starting at `start` for `count` elements.
    ///
    /// # Safety
    /// Caller must hold the write lock and ensure start + count < CAP.
    #[inline]
    unsafe fn shift_values_right(&self, start: usize, count: usize) {
        if count == 0 {
            return;
        }
        unsafe {
            let ptr = self.values.get() as *mut V;
            std::ptr::copy(ptr.add(start), ptr.add(start + 1), count);
        }
    }

    /// Shifts children right by one position starting at `start` for `count` elements.
    ///
    /// # Safety
    /// Caller must hold the write lock and ensure start + count < CHILDREN_CAP.
    #[inline]
    unsafe fn shift_children_right(&self, start: usize, count: usize) {
        if count == 0 {
            return;
        }
        unsafe {
            let ptr = self.children.get() as *mut Option<NodeId>;
            std::ptr::copy(ptr.add(start), ptr.add(start + 1), count);
        }
    }

    /// Shifts keys left by one position, overwriting the element at `start`.
    ///
    /// # Safety
    /// Caller must hold the write lock and ensure start + count <= CAP.
    #[inline]
    unsafe fn shift_keys_left(&self, start: usize, count: usize) {
        if count == 0 {
            return;
        }
        unsafe {
            let ptr = self.keys.get() as *mut K;
            std::ptr::copy(ptr.add(start + 1), ptr.add(start), count);
        }
    }

    /// Shifts values left by one position, overwriting the element at `start`.
    ///
    /// # Safety
    /// Caller must hold the write lock and ensure start + count <= CAP.
    #[inline]
    unsafe fn shift_values_left(&self, start: usize, count: usize) {
        if count == 0 {
            return;
        }
        unsafe {
            let ptr = self.values.get() as *mut V;
            std::ptr::copy(ptr.add(start + 1), ptr.add(start), count);
        }
    }

    /// Shifts children left by one position, overwriting the element at `start`.
    ///
    /// # Safety
    /// Caller must hold the write lock and ensure start + count <= CHILDREN_CAP.
    #[inline]
    unsafe fn shift_children_left(&self, start: usize, count: usize) {
        if count == 0 {
            return;
        }
        unsafe {
            let ptr = self.children.get() as *mut Option<NodeId>;
            std::ptr::copy(ptr.add(start + 1), ptr.add(start), count);
        }
    }

    /// Copies keys from another node using raw pointers.
    ///
    /// # Safety
    /// Caller must hold write locks on both nodes and ensure ranges are valid.
    #[inline]
    unsafe fn copy_keys_from(&self, dst_start: usize, src: &Self, src_start: usize, count: usize) {
        if count == 0 {
            return;
        }
        unsafe {
            let src_ptr = src.keys.get() as *const K;
            let dst_ptr = self.keys.get() as *mut K;
            std::ptr::copy_nonoverlapping(src_ptr.add(src_start), dst_ptr.add(dst_start), count);
        }
    }

    /// Copies values from another node using raw pointers.
    ///
    /// # Safety
    /// Caller must hold write locks on both nodes and ensure ranges are valid.
    #[inline]
    unsafe fn copy_values_from(&self, dst_start: usize, src: &Self, src_start: usize, count: usize) {
        if count == 0 {
            return;
        }
        unsafe {
            let src_ptr = src.values.get() as *const V;
            let dst_ptr = self.values.get() as *mut V;
            std::ptr::copy_nonoverlapping(src_ptr.add(src_start), dst_ptr.add(dst_start), count);
        }
    }

    /// Copies children from another node using raw pointers.
    ///
    /// # Safety
    /// Caller must hold write locks on both nodes and ensure ranges are valid.
    #[inline]
    unsafe fn copy_children_from(&self, dst_start: usize, src: &Self, src_start: usize, count: usize) {
        if count == 0 {
            return;
        }
        unsafe {
            let src_ptr = src.children.get() as *const Option<NodeId>;
            let dst_ptr = self.children.get() as *mut Option<NodeId>;
            std::ptr::copy_nonoverlapping(src_ptr.add(src_start), dst_ptr.add(dst_start), count);
        }
    }

    /// Binary search using raw pointers. Returns Ok(idx) if found, Err(idx) for insertion point.
    ///
    /// # Safety
    /// Caller must ensure len is valid and either hold write lock or be in OLC read phase.
    #[inline]
    unsafe fn binary_search_raw(&self, key: &K, len: usize) -> Result<usize, usize> {
        let mut left = 0;
        let mut right = len;
        while left < right {
            let mid = left + (right - left) / 2;
            let mid_key = unsafe { self.read_key(mid) };
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

#[allow(dead_code)]
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

    #[inline]
    fn node(&self) -> &'a Node<K, V, CAP, CHILDREN_CAP> {
        self.node
    }

    #[inline]
    fn into_inner(mut self) -> &'a Node<K, V, CAP, CHILDREN_CAP> {
        self.released = true;
        self.node
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

/// A concurrent B-tree using Optimistic Lock Coupling (OLC).
///
/// This B-tree provides thread-safe concurrent reads and writes without global
/// locks. Reads are optimistic (lock-free under low contention), while writes
/// use fine-grained per-node locking.
///
/// # Performance Characteristics
///
/// - **Reads**: Lock-free traversal with version validation. Restarts on conflict.
/// - **Writes**: Per-node spin locks with exponential backoff.
/// - **Memory**: Pre-allocated arena; no dynamic allocation during operations.
///
/// # Memory Reclamation
///
/// **This implementation does NOT recycle deleted nodes.** The arena grows
/// monotonically until `clear()` is called or the tree is dropped. This is a
/// deliberate safety constraint: recycling nodes without epoch-based reclamation
/// (EBR) or hazard pointers would cause data races with concurrent readers.
///
/// For applications requiring memory reclamation, consider wrapping with
/// `crossbeam-epoch` or implementing a quiescence protocol.
///
/// # Type Constraints
///
/// Keys and values must be `Copy` because the OLC protocol requires reading data
/// without holding locks. Non-Copy types would require additional synchronization
/// to prevent use-after-free during concurrent access.
pub struct BTree<K, V, const CAP: usize = DEFAULT_CAP, const CHILDREN_CAP: usize = DEFAULT_CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    /// Pre-allocated arena of nodes. Never resized after construction.
    pages: UnsafeCell<Vec<Node<K, V, CAP, CHILDREN_CAP>>>,
    /// Next free index for allocation. Monotonically increasing.
    next_free_idx: AtomicUsize,
    /// Current root node ID.
    root_id: AtomicU32,
    /// Maximum number of nodes (fixed at construction).
    node_capacity: usize,
    /// Number of key-value pairs in the tree.
    entry_count: AtomicUsize,
}

// SAFETY: BTree is Sync because:
// 1. The `pages` Vec is never resized after construction (pre-allocated arena)
// 2. Individual nodes are Sync due to the OLC protocol
// 3. Atomic fields (`next_free_idx`, `root_id`, `entry_count`) are inherently Sync
// 4. Access to `pages` only yields `&Node` references, which are safe to share
unsafe impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> Sync for BTree<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{}

// SAFETY: BTree is Send because:
// 1. All fields are either Sync (atomics) or owned (Vec behind UnsafeCell)
// 2. The UnsafeCell<Vec> can be sent between threads since we maintain
//    exclusive ownership semantics via the OLC protocol
unsafe impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> Send for BTree<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{}

impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> Default for BTree<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> BTree<K, V, CAP, CHILDREN_CAP>
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

        BTree {
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

    pub fn contains_key(&self, key: K) -> bool {
        self.search(key).is_some()
    }

    /// Clears all entries from the tree, resetting it to an empty state.
    ///
    /// # Memory Reclamation
    ///
    /// This implementation does NOT recycle node indices. Previously allocated
    /// nodes become unreachable but their memory is not reused. This is a
    /// deliberate safety constraint: recycling indices without epoch-based
    /// reclamation (EBR) or hazard pointers would cause data races with
    /// concurrent readers still traversing the old tree structure.
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

        // SAFETY: We hold write locks on both the current root and node0.
        unsafe {
            let keys_ptr = node0.keys.get();
            let vals_ptr = node0.values.get();
            let children_ptr = node0.children.get();

            std::ptr::write(keys_ptr, [K::default(); CAP]);
            std::ptr::write(vals_ptr, [V::default(); CAP]);
            std::ptr::write(children_ptr, [None; CHILDREN_CAP]);
            std::ptr::write(node0.len.get(), 0);
            std::ptr::write(node0.is_leaf.get(), true);
        }

        self.entry_count.store(0, Ordering::Relaxed);
        self.root_id.store(0, Ordering::Release);

        if root_id.0 != 0 {
            node0.write_unlock();
        }
        self.node(root_id).write_unlock();
    }

    /// Returns a reference to the node at the given ID.
    ///
    /// # Safety Justification
    ///
    /// This is safe because:
    /// 1. The `pages` Vec is pre-allocated and never resized (stable addresses)
    /// 2. `NodeId` values are only created by `new_node()` with valid indices
    /// 3. Creating `&Node` only borrows the Node struct, not the UnsafeCell contents
    /// 4. Multiple `&Node` references can coexist safely; interior mutability is
    ///    handled by the OLC protocol within each node
    fn node(&self, id: NodeId) -> &Node<K, V, CAP, CHILDREN_CAP> {
        // SAFETY: pages is never resized after construction, and id.0 is always
        // a valid index (allocated by new_node or 0 for the initial root).
        unsafe {
            let ptr = self.pages.get();
            let slice = &*ptr;
            &slice[id.0 as usize]
        }
    }

    fn new_node(&self, is_leaf: bool) -> NodeId {
        let idx = self.next_free_idx.fetch_add(1, Ordering::AcqRel);

        if idx >= self.node_capacity {
            panic!("Arena OOM: Tree exceeded capacity of {} nodes. Use BTree::with_capacity() for larger trees.", self.node_capacity);
        }

        let n = self.node(NodeId(idx as u32));

        // SAFETY: This node index was just atomically claimed by us. The node starts
        // with version = u64::MAX (odd), so optimistic readers will skip it. We
        // initialize the data and then set version to 0 with Release ordering to
        // ensure the initialization is visible before any reader could see this node.
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

    pub fn iter(&self) -> Iter<'_, K, V, CAP, CHILDREN_CAP> {
        Iter::new(self)
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
                let len = node.read_len();
                let is_leaf = node.read_is_leaf();

                if is_leaf {
                    Err(())
                } else {
                    let idx = match node.binary_search_raw(&key, len) {
                        Ok(i) => i + 1,
                        Err(i) => i,
                    };
                    Ok(node.read_child(idx).expect("child must exist at search index"))
                }
            };

            if !node.validate(start_version) {
                return Some(false);
            }

            match next_step {
                Ok(child_id) => current_id = child_id,
                Err(_) => {
                    let guard = NodeWriteGuard::new(node);

                    let current_version = guard.version.load(Ordering::Relaxed);
                    if current_version != start_version + 1 {
                        return Some(false);
                    }

                    // SAFETY: We hold the write lock (guard).
                    let is_leaf = unsafe { guard.read_is_leaf() };
                    if !is_leaf {
                        return Some(false);
                    }

                    // SAFETY: We hold the write lock (guard).
                    let is_full = unsafe { guard.read_len() >= CAP };
                    if is_full {
                        return None;
                    }

                    // SAFETY: We hold the write lock (guard).
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
        // SAFETY: We hold the write lock on the root node.
        let root_full = unsafe { root.read_len() == CAP };

        let current_root_id = if root_full {
            let new_root_id = self.new_node(false);
            let new_root = self.node(new_root_id);

            // SAFETY: new_root was just allocated and is not yet visible to other threads.
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
        // SAFETY: We hold the write lock on this node.
        let is_leaf = unsafe { node.read_is_leaf() };

        if is_leaf {
            // SAFETY: We hold the write lock.
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
            // SAFETY: We hold the write lock on this node.
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

                // SAFETY: After split, parent has a new key at idx.
                let go_right = unsafe { key > node.read_key(idx) };

                if go_right {
                    let right_id = unsafe { node.read_child(idx + 1).expect("right child must exist after split") };
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

        // SAFETY: Caller holds write lock on parent.
        let child_id = unsafe { parent.read_child(child_idx).expect("child must exist at split index") };
        let child = self.node(child_id);

        let (mid_key, mid_val, right_id) = self.allocate_and_distribute(child_id);

        // SAFETY: We hold write lock on parent.
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
        // SAFETY: Caller holds write lock on left node.
        let is_leaf = unsafe { left.read_is_leaf() };

        let right_id = self.new_node(is_leaf);
        let right = self.node(right_id);

        // SAFETY: We hold write lock on left, and right is newly allocated (not yet
        // visible to other threads).
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
                let len = node.read_len();
                let is_leaf = node.read_is_leaf();

                if is_leaf {
                    match node.binary_search_raw(&key, len) {
                        Ok(idx) => Err(Some(idx)),
                        Err(_) => Err(None),
                    }
                } else {
                    match node.binary_search_raw(&key, len) {
                        Ok(_) => {
                            return None;
                        }
                        Err(idx) => Ok(node.read_child(idx)),
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
                    let guard = NodeWriteGuard::new(node);

                    let current_version = guard.version.load(Ordering::Relaxed);
                    if current_version != start_version + 1 {
                        return None;
                    }

                    // SAFETY: We hold the write lock (guard). Using raw pointer reads.
                    let can_delete = unsafe {
                        let len = guard.read_len();
                        let is_leaf = guard.read_is_leaf();
                        let is_root = current_id.0 == self.root_id.load(Ordering::Relaxed);

                        let key_matches = matches!(
                            guard.binary_search_raw(&key, len),
                            Ok(found_idx) if found_idx == idx
                        );

                        is_leaf && (len > CAP / 2 || is_root) && key_matches
                    };

                    if !can_delete {
                        return None;
                    }

                    let value = self.remove_from_leaf(current_id, idx);
                    drop(guard);
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
        // SAFETY: We hold the write lock on root.
        let (should_shrink, new_root) = unsafe {
            let len = root.read_len();
            let is_leaf = root.read_is_leaf();
            if len == 0 && !is_leaf {
                (true, root.read_child(0).expect("internal node must have child"))
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

        // SAFETY: Caller holds write lock on this node.
        let (is_leaf, len) = unsafe { (node.read_is_leaf(), node.read_len()) };

        let search_result = unsafe { node.binary_search_raw(&key, len) };

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

        // SAFETY: Caller holds write lock on parent.
        let child_id = unsafe { parent.read_child(child_idx).expect("child must exist") };
        let child = self.node(child_id);
        child.write_lock();

        let child_len = unsafe { child.read_len() };

        if child_len > CAP / 2 {
            child.write_unlock();
            return child_id;
        }

        let parent_len = unsafe { parent.read_len() };

        if child_idx > 0 {
            let left_id = unsafe { parent.read_child(child_idx - 1).expect("left sibling must exist") };
            let left = self.node(left_id);
            left.write_lock();

            let left_len = unsafe { left.read_len() };
            if left_len > CAP / 2 {
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
            let right_id = unsafe { parent.read_child(child_idx + 1).expect("right sibling must exist") };
            let right = self.node(right_id);
            right.write_lock();

            let right_len = unsafe { right.read_len() };
            if right_len > CAP / 2 {
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
        // SAFETY: Caller holds write locks on parent, child, and left sibling.
        let child_id = unsafe { parent.read_child(child_idx).expect("child must exist") };
        let left_id = unsafe { parent.read_child(child_idx - 1).expect("left sibling must exist") };

        let child = self.node(child_id);
        let left = self.node(left_id);

        // SAFETY: We hold write locks on all three nodes.
        unsafe {
            let c_len = child.read_len();
            let l_len = left.read_len();
            let is_internal = !child.read_is_leaf();

            child.shift_keys_right(0, c_len);
            child.shift_values_right(0, c_len);
            if is_internal {
                child.shift_children_right(0, c_len + 1);
            }

            child.write_key(0, parent.read_key(child_idx - 1));
            child.write_value(0, parent.read_value(child_idx - 1));

            parent.write_key(child_idx - 1, left.read_key(l_len - 1));
            parent.write_value(child_idx - 1, left.read_value(l_len - 1));

            if is_internal {
                child.write_child(0, left.read_child(l_len));
            }

            child.write_len(c_len + 1);
            left.write_len(l_len - 1);
        }
    }

    fn borrow_from_right(&self, parent_id: NodeId, child_idx: usize) {
        let parent = self.node(parent_id);
        // SAFETY: Caller holds write locks on parent, child, and right sibling.
        let child_id = unsafe { parent.read_child(child_idx).expect("child must exist") };
        let right_id = unsafe { parent.read_child(child_idx + 1).expect("right sibling must exist") };

        let child = self.node(child_id);
        let right = self.node(right_id);

        // SAFETY: We hold write locks on all three nodes.
        unsafe {
            let c_len = child.read_len();
            let r_len = right.read_len();
            let is_internal = !child.read_is_leaf();

            child.write_key(c_len, parent.read_key(child_idx));
            child.write_value(c_len, parent.read_value(child_idx));

            if is_internal {
                child.write_child(c_len + 1, right.read_child(0));
            }

            parent.write_key(child_idx, right.read_key(0));
            parent.write_value(child_idx, right.read_value(0));

            right.shift_keys_left(0, r_len - 1);
            right.shift_values_left(0, r_len - 1);
            if is_internal {
                right.shift_children_left(0, r_len);
            }

            child.write_len(c_len + 1);
            right.write_len(r_len - 1);
        }
    }

    fn merge_with_left(&self, parent_id: NodeId, child_idx: usize) -> NodeId {
        let parent = self.node(parent_id);
        // SAFETY: Caller holds write locks on parent, child, and left sibling.
        let child_id = unsafe { parent.read_child(child_idx).expect("child must exist") };
        let left_id = unsafe { parent.read_child(child_idx - 1).expect("left sibling must exist") };

        let child = self.node(child_id);
        let left = self.node(left_id);

        // SAFETY: We hold write locks on all three nodes.
        unsafe {
            let p_len = parent.read_len();
            let c_len = child.read_len();
            let l_len = left.read_len();
            let is_internal = !child.read_is_leaf();

            left.write_key(l_len, parent.read_key(child_idx - 1));
            left.write_value(l_len, parent.read_value(child_idx - 1));

            left.copy_keys_from(l_len + 1, child, 0, c_len);
            left.copy_values_from(l_len + 1, child, 0, c_len);

            if is_internal {
                left.copy_children_from(l_len + 1, child, 0, c_len + 1);
            }

            left.write_len(l_len + 1 + c_len);

            parent.shift_keys_left(child_idx - 1, p_len - child_idx);
            parent.shift_values_left(child_idx - 1, p_len - child_idx);
            parent.shift_children_left(child_idx, p_len - child_idx);
            parent.write_len(p_len - 1);
        }

        left_id
    }

    fn merge_with_right(&self, parent_id: NodeId, child_idx: usize) -> NodeId {
        let parent = self.node(parent_id);
        // SAFETY: Caller holds write locks on parent, child, and right sibling.
        let child_id = unsafe { parent.read_child(child_idx).expect("child must exist") };
        let right_id = unsafe { parent.read_child(child_idx + 1).expect("right sibling must exist") };

        let child = self.node(child_id);
        let right = self.node(right_id);

        // SAFETY: We hold write locks on all three nodes.
        unsafe {
            let p_len = parent.read_len();
            let c_len = child.read_len();
            let r_len = right.read_len();
            let is_internal = !child.read_is_leaf();

            child.write_key(c_len, parent.read_key(child_idx));
            child.write_value(c_len, parent.read_value(child_idx));

            child.copy_keys_from(c_len + 1, right, 0, r_len);
            child.copy_values_from(c_len + 1, right, 0, r_len);

            if is_internal {
                child.copy_children_from(c_len + 1, right, 0, r_len + 1);
            }

            child.write_len(c_len + 1 + r_len);

            parent.shift_keys_left(child_idx, p_len - child_idx - 1);
            parent.shift_values_left(child_idx, p_len - child_idx - 1);
            parent.shift_children_left(child_idx + 1, p_len - child_idx - 1);
            parent.write_len(p_len - 1);
        }

        child_id
    }

    fn remove_from_internal(&self, node_id: NodeId, idx: usize, key: K) -> V {
        let node = self.node(node_id);

        // SAFETY: Caller holds write lock on node.
        let old_value = unsafe { node.read_value(idx) };

        let left_child_id = self.ensure_child_can_lose_key(node_id, idx);

        let current_len = unsafe { node.read_len() };

        let key_still_here = unsafe {
            idx < current_len && node.read_key(idx) == key
        };

        if key_still_here {
            let (pred_key, pred_val) = self.delete_max_from_subtree(left_child_id);

            // SAFETY: We hold write lock on node.
            unsafe {
                node.write_key(idx, pred_key);
                node.write_value(idx, pred_val);
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

        // SAFETY: We hold the write lock.
        let is_leaf = unsafe { node.read_is_leaf() };

        if is_leaf {
            let len = unsafe { node.read_len() };
            let result = unsafe {
                (node.read_key(len - 1), node.read_value(len - 1))
            };
            unsafe { node.write_len(len - 1) };
            node.write_unlock();
            result
        } else {
            let len = unsafe { node.read_len() };
            let rightmost_idx = len;

            let child_id = self.ensure_child_can_lose_key(node_id, rightmost_idx);

            let result = self.delete_max_from_subtree(child_id);

            node.write_unlock();
            result
        }
    }

    fn remove_from_leaf(&self, node_id: NodeId, idx: usize) -> V {
        let node = self.node(node_id);

        // SAFETY: Caller holds write lock on node.
        unsafe {
            let len = node.read_len();
            let value = node.read_value(idx);

            node.shift_keys_left(idx, len - idx - 1);
            node.shift_values_left(idx, len - idx - 1);

            node.write_len(len - 1);

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
        println!("=== B-Tree Structure ===");
        self.print_subtree(NodeId(self.root_id.load(Ordering::Acquire)), 0);
        println!("======================================");
    }

    fn print_subtree(&self, node_id: NodeId, depth: usize) {
        let node = self.node(node_id);
        let indent = "  ".repeat(depth);

        // SAFETY: This is a debug function that performs unsynchronized reads.
        // It should only be used when no concurrent mutations are occurring,
        // or when the caller accepts that the output may be inconsistent.
        // We use raw pointer reads to avoid creating references that could
        // conflict with concurrent writers under Stacked Borrows.
        unsafe {
            let len = node.read_len();
            let is_leaf = node.read_is_leaf();

            let keys: Vec<K> = (0..len).map(|i| node.read_key(i)).collect();

            println!(
                "{}Node[{}] (Leaf: {}) Keys: {:?}",
                indent, node_id.0, is_leaf, keys
            );

            if !is_leaf {
                for i in 0..=len {
                    if let Some(child_id) = node.read_child(i) {
                        self.print_subtree(child_id, depth + 1);
                    }
                }
            }
        }
    }
}

pub struct Iter<'a, K, V, const CAP: usize = DEFAULT_CAP, const CHILDREN_CAP: usize = DEFAULT_CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    tree: &'a BTree<K, V, CAP, CHILDREN_CAP>,
    stack: Vec<(NodeId, usize)>,
    last_key: Option<K>,
    started: bool,
}

impl<'a, K, V, const CAP: usize, const CHILDREN_CAP: usize> Iter<'a, K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    fn new(tree: &'a BTree<K, V, CAP, CHILDREN_CAP>) -> Self {
        Self {
            tree,
            stack: Vec::with_capacity(16),
            last_key: None,
            started: false,
        }
    }

    fn descend_left(&mut self, start: NodeId) -> bool {
        let mut current = start;
        loop {
            let node = self.tree.node(current);
            let version = node.read_version();
            if version % 2 != 0 {
                return false;
            }

            let is_leaf = unsafe { node.read_is_leaf() };

            if !node.validate(version) {
                return false;
            }

            self.stack.push((current, 0));

            if is_leaf {
                return true;
            }

            let version = node.read_version();
            if version % 2 != 0 {
                return false;
            }

            let child = unsafe { node.read_child(0) };

            if !node.validate(version) {
                return false;
            }

            match child {
                Some(id) => current = id,
                None => return true,
            }
        }
    }

    fn seek_after(&mut self, key: K) -> bool {
        self.stack.clear();
        let root_id = NodeId(self.tree.root_id.load(Ordering::Acquire));
        let mut current = root_id;

        loop {
            let node = self.tree.node(current);
            let version = node.read_version();
            if version % 2 != 0 {
                return false;
            }

            let len = unsafe { node.read_len() };
            let is_leaf = unsafe { node.read_is_leaf() };

            let idx = unsafe {
                match node.binary_search_raw(&key, len) {
                    Ok(i) => i + 1,
                    Err(i) => i,
                }
            };

            if !node.validate(version) {
                return false;
            }

            self.stack.push((current, idx));

            if is_leaf {
                return true;
            }

            let version = node.read_version();
            if version % 2 != 0 {
                return false;
            }

            let child = unsafe { node.read_child(idx) };

            if !node.validate(version) {
                return false;
            }

            match child {
                Some(id) => current = id,
                None => return true,
            }
        }
    }

    fn restart_from_last_key(&mut self) {
        self.stack.clear();
    }
}

impl<'a, K, V, const CAP: usize, const CHILDREN_CAP: usize> Iterator for Iter<'a, K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        'restart: loop {
            if !self.started || self.stack.is_empty() {
                self.started = true;

                if let Some(last) = self.last_key {

                    if !self.seek_after(last) {
                        continue 'restart;
                    }
                } else {
                    let root_id = NodeId(self.tree.root_id.load(Ordering::Acquire));
                    if !self.descend_left(root_id) {
                        continue 'restart;
                    }
                }
            }

            while let Some((node_id, idx)) = self.stack.last_mut() {
                let node = self.tree.node(*node_id);
                let version = node.read_version();
                if version % 2 != 0 {
                    self.restart_from_last_key();
                    continue 'restart;
                }

                let len = unsafe { node.read_len() };
                let is_leaf = unsafe { node.read_is_leaf() };

                if !node.validate(version) {
                    self.restart_from_last_key();
                    continue 'restart;
                }

                if *idx < len {
                    let version = node.read_version();
                    if version % 2 != 0 {
                        self.restart_from_last_key();
                        continue 'restart;
                    }

                    let k = unsafe { node.read_key(*idx) };
                    let v = unsafe { node.read_value(*idx) };

                    if !node.validate(version) {
                        self.restart_from_last_key();
                        continue 'restart;
                    }

                    if let Some(last) = self.last_key {
                        if k <= last {
                            self.restart_from_last_key();
                            continue 'restart;
                        }
                    }

                    *idx += 1;
                    self.last_key = Some(k);

                    if !is_leaf && *idx <= len {
                        let version = node.read_version();
                        if version % 2 != 0 {
                            self.restart_from_last_key();
                            continue 'restart;
                        }

                        let child = unsafe { node.read_child(*idx) };

                        if !node.validate(version) {
                            self.restart_from_last_key();
                            continue 'restart;
                        }

                        if let Some(child_id) = child {
                            if !self.descend_left(child_id) {
                                self.restart_from_last_key();
                                continue 'restart;
                            }
                        }
                    }

                    return Some((k, v));
                } else {
                    self.stack.pop();
                }
            }

            return None;
        }
    }
}

impl<'a, K, V, const CAP: usize, const CHILDREN_CAP: usize> IntoIterator for &'a BTree<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    type Item = (K, V);
    type IntoIter = Iter<'a, K, V, CAP, CHILDREN_CAP>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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

    #[test]
    fn test_iter_empty() {
        let tree = BTree::<u64, u64>::new();
        assert_eq!(tree.iter().count(), 0);
    }

    #[test]
    fn test_iter_single() {
        let tree = BTree::<u64, u64>::new();
        tree.insert(42, 420);

        let items: Vec<_> = tree.iter().collect();
        assert_eq!(items, vec![(42, 420)]);
    }

    #[test]
    fn test_iter_ordered() {
        let tree = BTree::<u64, u64>::new();

        tree.insert(50, 500);
        tree.insert(20, 200);
        tree.insert(80, 800);
        tree.insert(10, 100);
        tree.insert(30, 300);

        let items: Vec<_> = tree.iter().collect();
        assert_eq!(items, vec![
            (10, 100),
            (20, 200),
            (30, 300),
            (50, 500),
            (80, 800),
        ]);
    }

    #[test]
    fn test_iter_large() {
        let tree = BTree::<u64, u64>::new();

        for i in (0..1000u64).rev() {
            tree.insert(i, i * 10);
        }

        let items: Vec<_> = tree.iter().collect();
        assert_eq!(items.len(), 1000);

        for (idx, (k, v)) in items.iter().enumerate() {
            assert_eq!(*k, idx as u64);
            assert_eq!(*v, idx as u64 * 10);
        }
    }

    #[test]
    fn test_iter_for_loop() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..10u64 {
            tree.insert(i, i * 100);
        }

        let mut count = 0;
        for (k, v) in &tree {
            assert_eq!(v, k * 100);
            count += 1;
        }
        assert_eq!(count, 10);
    }

    #[test]
    fn test_iter_concurrent_read() {
        use std::sync::Arc;
        use std::thread;

        let tree = Arc::new(BTree::<u64, u64>::new());
        for i in 0..1000u64 {
            tree.insert(i, i * 10);
        }

        let mut handles = vec![];
        for _ in 0..4 {
            let tree_ref = tree.clone();
            handles.push(thread::spawn(move || {
                let items: Vec<_> = tree_ref.iter().collect();
                assert_eq!(items.len(), 1000);
                for i in 1..items.len() {
                    assert!(items[i].0 > items[i - 1].0);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }
}