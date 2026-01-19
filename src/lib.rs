use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::marker::Sync;

const B: usize = 6;
const CAPACITY: usize = 2 * B - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(u32);

#[derive(Debug)]
pub struct Node {
    pub version: AtomicU64,

    pub keys: UnsafeCell<[u64; CAPACITY]>,
    pub values: UnsafeCell<[u64; CAPACITY]>,
    pub children: UnsafeCell<[Option<NodeId>; CAPACITY + 1]>,
    pub len: UnsafeCell<usize>,
    pub is_leaf: bool,
}

unsafe impl Sync for Node {}

impl Node {
    pub fn new(is_leaf: bool) -> Self {
        Self {
            version: AtomicU64::new(0),
            len: UnsafeCell::new(0),
            is_leaf,
            keys: UnsafeCell::new([0; CAPACITY]),
            values: UnsafeCell::new([0; CAPACITY]),
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

pub struct BTree {
    pages: UnsafeCell<Vec<Node>>,
    next_free_idx: AtomicUsize,
    root_id: NodeId,
}

unsafe impl Sync for BTree {}
unsafe impl Send for BTree {}

impl BTree {
    pub fn new() -> Self {
        let max_nodes = 100_000;
        let mut pages = Vec::with_capacity(1024);
        pages.push(Node::new(true));
        for _ in 0..(max_nodes - 1) {
            pages.push(Node::new(true));
        }
        BTree {
            pages: UnsafeCell::new(pages),
            next_free_idx: AtomicUsize::new(1),
            root_id: NodeId(0),
        }
    }

    fn node(&self, id: NodeId) -> &Node {
        unsafe {
            let ptr = self.pages.get();
            &(*ptr)[id.0 as usize]
        }
    }

    fn new_node(&mut self, is_leaf: bool) -> NodeId {
        let idx = self.next_free_idx.fetch_add(1, Ordering::Relaxed);

        if idx >= 100_000 {
            panic!("Arena OOM: Increase max_nodes in BTree::new");
        }

        let n = self.node(NodeId(idx as u32));

        unsafe {
            *n.keys.get() = [0; CAPACITY];
            *n.values.get() = [0; CAPACITY];
            *n.children.get() = [None; CAPACITY + 1];
            *n.len.get() = 0;
            let ptr = n as *const Node as *mut Node;
            (*ptr).is_leaf = is_leaf;
            (*ptr).version = AtomicU64::new(0);
        }
        NodeId(idx as u32)
    }

    pub fn search(&self, key: u64) -> Option<u64> {
        'restart: loop {
            let mut current_id = self.root_id;

            loop {
                let node = self.node(current_id);

                let start_version = node.read_version();

                let (found, child_id) = unsafe {
                    let len = *node.len.get();
                    let keys = &*node.keys.get();
                    match keys[0..len].binary_search(&key) {
                        Ok(idx) => (Some((*node.values.get())[idx]), None),
                        Err(idx) => {
                            if node.is_leaf {
                                (None, None)
                            } else {
                                (None, (*node.children.get())[idx])
                            }
                        }
                    }
                };

                if !node.validate(start_version) {
                    continue 'restart;
                }

                if let Some(val) = found {
                    return Some(val);
                }
                match child_id {
                    Some(id) => current_id = id,
                    None => return None,
                }
            }
        }
    }

    pub fn insert(&mut self, key: u64, value: u64) {
        if self.pages[self.root_id.0 as usize].len == CAPACITY {
            let new_root_id = self.new_node(false);
            let old_root_id = self.root_id;

            self.pages[new_root_id.0 as usize].children[0] = Some(old_root_id);
            self.root_id = new_root_id;

            self.split_child(new_root_id, 0);
        }

        self.insert_non_full(self.root_id, key, value);
    }

    fn split_child(&mut self, parent_id: NodeId, child_idx: usize) {
        let child_id =
            self.pages[parent_id.0 as usize].children[child_idx].expect("Child must exist");

        let (median_key, median_val, right_id) = self.allocate_and_distribute(child_id);
        let parent = &mut self.pages[parent_id.0 as usize];

        parent
            .keys
            .copy_within(child_idx..parent.len, child_idx + 1);
        parent
            .values
            .copy_within(child_idx..parent.len, child_idx + 1);
        parent
            .children
            .copy_within(child_idx + 1..parent.len + 1, child_idx + 2);

        parent.keys[child_idx] = median_key;
        parent.values[child_idx] = median_val;
        parent.children[child_idx + 1] = Some(right_id);
        parent.len += 1;
    }

    fn allocate_and_distribute(&mut self, left_id: NodeId) -> (u64, u64, NodeId) {
        let is_leaf = self.pages[left_id.0 as usize].is_leaf;
        let right_id = self.new_node(is_leaf);

        let (left, right) = self.get_mut_pair(left_id, right_id);

        let mid = left.len / 2;
        let count = left.len - 1 - mid;

        right.keys[0..count].copy_from_slice(&left.keys[mid + 1..mid + 1 + count]);
        right.values[0..count].copy_from_slice(&left.values[mid + 1..mid + 1 + count]);

        if !left.is_leaf {
            right.children[0..=count].copy_from_slice(&left.children[mid + 1..mid + 2 + count]);
            left.children[mid + 1..mid + 2 + count].fill(None);
        }

        let median_key = left.keys[mid];
        let median_val = left.values[mid];

        left.len = mid;
        right.len = count;

        (median_key, median_val, right_id)
    }

    fn insert_non_full(&mut self, node_id: NodeId, key: u64, value: u64) {
        let is_leaf = self.pages[node_id.0 as usize].is_leaf;

        if is_leaf {
            let node = &mut self.pages[node_id.0 as usize];
            match node.search_key(key) {
                Ok(idx) => {
                    node.values[idx] = value;
                }
                Err(idx) => {
                    node.keys.copy_within(idx..node.len, idx + 1);
                    node.values.copy_within(idx..node.len, idx + 1);

                    node.keys[idx] = key;
                    node.values[idx] = value;
                    node.len += 1;
                }
            }
        } else {
            let (child_idx, must_split) = {
                let node = &self.pages[node_id.0 as usize];
                let idx = match node.search_key(key) {
                    Ok(i) => i + 1,
                    Err(i) => i,
                };
                let child_id = node.children[idx].expect("Internal node structure broken");
                (idx, self.pages[child_id.0 as usize].len == CAPACITY)
            };

            if must_split {
                self.split_child(node_id, child_idx);

                let node = &self.pages[node_id.0 as usize];
                let current_key_at_split = node.keys[child_idx];

                if key > current_key_at_split {
                    let next_child = node.children[child_idx + 1].unwrap();
                    self.insert_non_full(next_child, key, value);
                } else {
                    let next_child = node.children[child_idx].unwrap();
                    self.insert_non_full(next_child, key, value);
                }
            } else {
                let next_child = self.pages[node_id.0 as usize].children[child_idx].unwrap();
                self.insert_non_full(next_child, key, value);
            }
        }
    }

    pub fn print(&self) {
        println!("=== B-Tree Structure (Arena) ===");
        self.print_subtree(self.root_id, 0);
        println!("================================");
    }

    fn print_subtree(&self, node_id: NodeId, depth: usize) {
        let node = &self.pages[node_id.0 as usize];
        let indent = "  ".repeat(depth);
        println!(
            "{}Node[{}] (Leaf: {}) Keys: {:?}",
            indent,
            node_id.0,
            node.is_leaf,
            &node.keys[0..node.len]
        );
        if !node.is_leaf {
            for i in 0..=node.len {
                if let Some(child_id) = node.children[i] {
                    self.print_subtree(child_id, depth + 1);
                }
            }
        }
    }
}

// Tests (Same as before)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let mut tree = BTree::new();
        tree.insert(10, 100);
        tree.insert(20, 200);
        assert_eq!(tree.search(10), Some(100));
        assert_eq!(tree.search(20), Some(200));
    }

    #[test]
    fn test_splits() {
        let mut tree = BTree::new();
        for i in 0..20 {
            tree.insert(i, i * 10);
        }
        for i in 0..20 {
            assert_eq!(tree.search(i), Some(i * 10));
        }
        assert!(tree.pages.len() > 1);
    }
}
