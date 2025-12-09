use std::sync::atomic::AtomicU64;

const B: usize = 6;
const CAPACITY: usize = 2 * B - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(u32);

#[derive(Debug)]
pub struct Node {
    pub version: AtomicU64, // TODO: For lock coupling
    pub keys: [u64; CAPACITY],
    pub values: [u64; CAPACITY],
    pub children: [Option<NodeId>; CAPACITY + 1],
    pub len: usize,
    pub is_leaf: bool,
}

impl Node {
    pub fn new(is_leaf: bool) -> Self {
        Self {
            version: AtomicU64::new(0),
            len: 0,
            is_leaf,
            keys: [0; CAPACITY],
            values: [0; CAPACITY],
            children: [None; CAPACITY + 1],
        }
    }

    pub fn search_node(&self, key: u64) -> Result<usize, usize> {
        let valid_keys = &self.keys[0..self.len];
        valid_keys.binary_search(&key)
    }
}

pub struct BTree {
    pages: Vec<Node>,
    root_id: NodeId,
}

impl BTree {
    pub fn new() -> Self {
        let mut pages = Vec::with_capacity(1024);

        let root_node = Node::new(true);
        pages.push(root_node);

        BTree {
            pages,
            root_id: NodeId(0),
        }
    }

    fn new_node(&mut self, is_leaf: bool) -> NodeId {
        let node = Node::new(is_leaf);
        let id = self.pages.len() as u32;
        self.pages.push(node);
        NodeId(id)
    }

    fn get_mut_pair(&mut self, idx1: NodeId, idx2: NodeId) -> (&mut Node, &mut Node) {
        let i1 = idx1.0 as usize;
        let i2 = idx2.0 as usize;
        assert_ne!(i1, i2, "Cannot borrow the same node twice.");

        if i1 < i2 {
            let (left_slice, right_slice) = self.pages.as_mut_slice().split_at_mut(i2);
            (&mut left_slice[i1], &mut right_slice[0])
        } else {
            let (left_slice, right_slice) = self.pages.as_mut_slice().split_at_mut(i1);
            (&mut right_slice[0], &mut left_slice[i2])
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

        let (mid_key, mid_val, right_id) = self.split_node(child_id);
        let parent = &mut self.pages[parent_id.0 as usize];

        for i in (child_idx..parent.len).rev() {
            parent.keys[i + 1] = parent.keys[i];
            parent.values[i + 1] = parent.values[i];
            parent.children[i + 2] = parent.children[i + 1];
        }

        parent.keys[child_idx] = mid_key;
        parent.values[child_idx] = mid_val;
        parent.children[child_idx + 1] = Some(right_id);
        parent.len += 1;
    }

    fn split_node(&mut self, node_id: NodeId) -> (u64, u64, NodeId) {
        let is_leaf = self.pages[node_id.0 as usize].is_leaf;
        let right_id = self.new_node(is_leaf);
        let (left, right) = self.get_mut_pair(node_id, right_id);

        let mid = left.len / 2;
        let count = left.len - 1 - mid;

        for i in 0..count {
            right.keys[i] = left.keys[mid + i + 1];
            right.values[i] = left.values[mid + i + 1];
        }

        if !left.is_leaf {
            for i in 0..=count {
                right.children[i] = left.children[mid + i + 1];
                left.children[mid + i + 1] = None;
            }
        }

        let median_key = left.keys[mid];
        let median_val = left.values[mid];

        left.len = mid;
        right.len = count;
        (median_key, median_val, right_id)
    }

    pub fn insert_non_full(&mut self, node_id: NodeId, key: u64, value: u64) {
        let is_leaf = self.pages[node_id.0 as usize].is_leaf;

        if is_leaf {
            let node_mut = &mut self.pages[node_id.0 as usize];

            let idx = match node_mut.search_node(key) {
                Ok(idx) => {
                    node_mut.values[idx] = value;
                    return;
                }
                Err(idx) => idx,
            };

            for i in (idx..node_mut.len).rev() {
                node_mut.keys[i + 1] = node_mut.keys[i];
                node_mut.values[i + 1] = node_mut.values[i];
            }

            node_mut.keys[idx] = key;
            node_mut.values[idx] = value;
            node_mut.len += 1;
        } else {
            let (child_idx, child_is_full) = {
                let node = &self.pages[node_id.0 as usize];
                let idx = match node.search_node(key) {
                    Ok(idx) => idx + 1,
                    Err(idx) => idx,
                };
                let child_id = node.children[idx].expect("Internal node missing child");
                (idx, self.pages[child_id.0 as usize].len == CAPACITY)
            };

            if child_is_full {
                self.split_child(node_id, child_idx);

                let node_ref = &self.pages[node_id.0 as usize];
                if key > node_ref.keys[child_idx] {
                    let right_child_id = node_ref.children[child_idx + 1].unwrap();
                    self.insert_non_full(right_child_id, key, value);
                } else {
                    let child_id = node_ref.children[child_idx].unwrap();
                    self.insert_non_full(child_id, key, value);
                }
            } else {
                let child_id = self.pages[node_id.0 as usize].children[child_idx].unwrap();
                self.insert_non_full(child_id, key, value);
            }
        }
    }

    pub fn search(&self, key: u64) -> Option<u64> {
        let mut current_id = self.root_id;
        loop {
            let node = &self.pages[current_id.0 as usize];
            match node.search_node(key) {
                Ok(idx) => return Some(node.values[idx]),
                Err(idx) => {
                    if node.is_leaf {
                        return None;
                    }
                    current_id = node.children[idx].unwrap();
                }
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
                    println!("{}Child {}:", indent, i);
                    self.print_subtree(child_id, depth + 1);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert_search() {
        let mut tree = BTree::new();

        tree.insert(10, 100);
        tree.insert(20, 200);
        tree.insert(5, 50);

        assert_eq!(tree.search(10), Some(100));
        assert_eq!(tree.search(5), Some(50));
        assert_eq!(tree.search(20), Some(200));
        assert_eq!(tree.search(999), None);
    }

    #[test]
    fn test_root_split() {
        let mut tree = BTree::new();

        for i in 1..=11 {
            tree.insert(i * 10, i * 100);
        }

        assert_eq!(tree.pages.len(), 1);
        assert_eq!(tree.pages[0].len, 11);

        tree.insert(120, 1200);

        assert!(tree.pages.len() >= 3);

        let root = &tree.pages[tree.root_id.0 as usize];
        assert_eq!(root.is_leaf, false);
        assert_eq!(root.len, 1);

        let left_child_id = root.children[0].expect("Left child missing");
        let right_child_id = root.children[1].expect("Right child missing");

        assert!(left_child_id.0 < tree.pages.len() as u32);
        assert!(right_child_id.0 < tree.pages.len() as u32);
    }

    #[test]
    fn test_large_volume() {
        let mut tree = BTree::new();
        for i in 0..1000 {
            tree.insert(i, i * 10);
        }

        for i in 0..1000 {
            assert_eq!(tree.search(i), Some(i * 10));
        }

        println!("Total nodes allocated: {}", tree.pages.len());
        assert!(tree.pages.len() < 200);
    }
}
