const B: usize = 6;
const CAPACITY: usize = 2 * B - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(u32);

#[derive(Debug)]
pub struct Node {
    pub keys: [u64; CAPACITY],
    pub values: [u64; CAPACITY],
    pub children: [Option<NodeId>; CAPACITY + 1],
    pub len: usize,
    pub is_leaf: bool,
}

impl Node {
    pub fn new(is_leaf: bool) -> Self {
        Self {
            len: 0,
            is_leaf,
            keys: [0; CAPACITY],
            values: [0; CAPACITY],
            children: [None; CAPACITY + 1],
        }
    }

    pub fn search_key(&self, key: u64) -> Result<usize, usize> {
        self.keys[0..self.len].binary_search(&key)
    }
}

pub struct BTree {
    pages: Vec<Node>,
    root_id: NodeId,
}

impl BTree {
    pub fn new() -> Self {
        let mut pages = Vec::with_capacity(1024);
        pages.push(Node::new(true));
        BTree {
            pages,
            root_id: NodeId(0),
        }
    }

    fn new_node(&mut self, is_leaf: bool) -> NodeId {
        let id = self.pages.len() as u32;
        self.pages.push(Node::new(is_leaf));
        NodeId(id)
    }

    fn get_mut_pair(&mut self, idx1: NodeId, idx2: NodeId) -> (&mut Node, &mut Node) {
        let i1 = idx1.0 as usize;
        let i2 = idx2.0 as usize;
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

    pub fn search(&self, key: u64) -> Option<u64> {
        let mut current_id = self.root_id;
        loop {
            let node = &self.pages[current_id.0 as usize];
            match node.search_key(key) {
                Ok(idx) => return Some(node.values[idx]),
                Err(idx) => {
                    if node.is_leaf {
                        return None;
                    }
                    current_id = node.children[idx]?;
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
