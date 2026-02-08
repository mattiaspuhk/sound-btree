//! BTree struct and core operations (search, insert, delete).

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::iter::Iter;
use crate::node::{Node, NodeId, NodeWriteGuard, DEFAULT_CAP, DEFAULT_CHILDREN_CAP};

pub struct BTree<K, V, const CAP: usize = DEFAULT_CAP, const CHILDREN_CAP: usize = DEFAULT_CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{
    pages: UnsafeCell<Vec<Node<K, V, CAP, CHILDREN_CAP>>>,
    next_free_idx: AtomicUsize,
    pub(crate) root_id: AtomicU32,
    node_capacity: usize,
    entry_count: AtomicUsize,
}

unsafe impl<K, V, const CAP: usize, const CHILDREN_CAP: usize> Sync for BTree<K, V, CAP, CHILDREN_CAP>
where
    K: Copy + Ord + Default,
    V: Copy + Default,
{}

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
        Self::with_capacity(Self::DEFAULT_NODE_CAPACITY)
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

    pub(crate) fn node(&self, id: NodeId) -> &Node<K, V, CAP, CHILDREN_CAP> {
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
