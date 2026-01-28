//! Node types and OLC primitives for the concurrent B-tree.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};

pub const DEFAULT_CAP: usize = 11;
pub const DEFAULT_CHILDREN_CAP: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(pub(crate) u32);

#[derive(Debug)]
pub struct Node<K, V, const CAP: usize = DEFAULT_CAP, const CHILDREN_CAP: usize = DEFAULT_CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    pub(crate) version: AtomicU64,

    pub(crate) keys: UnsafeCell<[K; CAP]>,
    pub(crate) values: UnsafeCell<[V; CAP]>,
    pub(crate) children: UnsafeCell<[Option<NodeId>; CHILDREN_CAP]>,
    pub(crate) len: UnsafeCell<usize>,
    pub(crate) is_leaf: UnsafeCell<bool>,
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

    pub(crate) fn new_uninit() -> Self {
        Self {
            version: AtomicU64::new(u64::MAX),
            len: UnsafeCell::new(0),
            is_leaf: UnsafeCell::new(true),
            keys: UnsafeCell::new([K::default(); CAP]),
            values: UnsafeCell::new([V::default(); CAP]),
            children: UnsafeCell::new([None; CHILDREN_CAP]),
        }
    }

    pub fn read_version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    pub fn validate(&self, start_version: u64) -> bool {
        std::sync::atomic::fence(Ordering::Acquire);
        let current = self.version.load(Ordering::Relaxed);
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

    /// Releases the write lock by incrementing version from odd to even.
    pub fn write_unlock(&self) {
        self.version.fetch_add(1, Ordering::Release);
    }

    #[inline]
    pub(crate) unsafe fn read_len(&self) -> usize {
        unsafe { std::ptr::read_volatile(self.len.get()) }
    }

    #[inline]
    pub(crate) unsafe fn read_is_leaf(&self) -> bool {
        unsafe { std::ptr::read_volatile(self.is_leaf.get()) }
    }

    #[inline]
    pub(crate) unsafe fn read_key(&self, idx: usize) -> K {
        unsafe {
            let ptr = self.keys.get() as *const K;
            std::ptr::read_volatile(ptr.add(idx))
        }
    }

    #[inline]
    pub(crate) unsafe fn read_value(&self, idx: usize) -> V {
        unsafe {
            let ptr = self.values.get() as *const V;
            std::ptr::read_volatile(ptr.add(idx))
        }
    }

    #[inline]
    pub(crate) unsafe fn read_child(&self, idx: usize) -> Option<NodeId> {
        unsafe {
            let ptr = self.children.get() as *const Option<NodeId>;
            std::ptr::read_volatile(ptr.add(idx))
        }
    }

    /// Writes a key at the given index using raw pointers.
    ///
    /// # Safety
    /// Caller must hold the write lock and ensure idx < CAP.
    #[inline]
    pub(crate) unsafe fn write_key(&self, idx: usize, key: K) {
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
    pub(crate) unsafe fn write_value(&self, idx: usize, value: V) {
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
    pub(crate) unsafe fn write_child(&self, idx: usize, child: Option<NodeId>) {
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
    pub(crate) unsafe fn write_len(&self, len: usize) {
        unsafe {
            std::ptr::write(self.len.get(), len);
        }
    }

    /// Shifts keys right by one position starting at `start` for `count` elements.
    ///
    /// # Safety
    /// Caller must hold the write lock and ensure start + count < CAP.
    #[inline]
    pub(crate) unsafe fn shift_keys_right(&self, start: usize, count: usize) {
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
    pub(crate) unsafe fn shift_values_right(&self, start: usize, count: usize) {
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
    pub(crate) unsafe fn shift_children_right(&self, start: usize, count: usize) {
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
    pub(crate) unsafe fn shift_keys_left(&self, start: usize, count: usize) {
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
    pub(crate) unsafe fn shift_values_left(&self, start: usize, count: usize) {
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
    pub(crate) unsafe fn shift_children_left(&self, start: usize, count: usize) {
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
    pub(crate) unsafe fn copy_keys_from(&self, dst_start: usize, src: &Self, src_start: usize, count: usize) {
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
    pub(crate) unsafe fn copy_values_from(&self, dst_start: usize, src: &Self, src_start: usize, count: usize) {
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
    pub(crate) unsafe fn copy_children_from(&self, dst_start: usize, src: &Self, src_start: usize, count: usize) {
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
    pub(crate) unsafe fn binary_search_raw(&self, key: &K, len: usize) -> Result<usize, usize> {
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
    pub(crate) fn new(node: &'a Node<K, V, CAP, CHILDREN_CAP>) -> Self {
        node.write_lock();
        Self { node, released: false }
    }

    #[inline]
    pub(crate) fn release(mut self) {
        if !self.released {
            self.node.write_unlock();
            self.released = true;
        }
    }

    #[inline]
    pub(crate) fn node(&self) -> &'a Node<K, V, CAP, CHILDREN_CAP> {
        self.node
    }

    #[inline]
    pub(crate) fn into_inner(mut self) -> &'a Node<K, V, CAP, CHILDREN_CAP> {
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
