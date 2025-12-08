use std::mem;

const B: usize = 6;
const CAPACITY: usize = 2 * B - 1;

#[derive(Debug)]
pub struct Node {
    pub keys: [u64; CAPACITY],
    pub values: [u64; CAPACITY],
    pub children: [Option<Box<Node>>; CAPACITY + 1],
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
            children: std::array::from_fn(|_| None),
        }
    }

    pub fn search_node(&self, key: u64) -> Result<usize, usize> {
        let valid_keys = &self.keys[0..self.len];
        valid_keys.binary_search(&key)
    }

    pub fn insert_non_full(&mut self, key: u64, value: u64) {
        let mut idx = match self.search_node(key) {
            Ok(idx) => {
                self.values[idx] = value;
                return;
            },
            Err(idx) => idx,
        };

        if self.is_leaf {
            if self.len == CAPACITY {
                panic!("Node is full");
            }

            for i in (idx..self.len).rev() {
                self.keys[i + 1] = self.keys[i];
                self.values[i + 1] = self.values[i];
            }

            self.keys[idx] = key;
            self.values[idx] = value;
            self.len += 1;
        } else {
            if self.children[idx].as_ref().unwrap().len == CAPACITY {
                let (mid_key, mid_val, right_child) = self.children[idx].as_mut().unwrap().split();

                for i in (idx..self.len).rev() {
                    self.keys[i+1] = self.keys[i];
                    self.values[i+1] = self.values[i];
                    self.children[i+2] = self.children[i+1].take();
                }

                self.keys[idx] = mid_key;
                self.values[idx] = mid_val;
                self.children[idx+1] = Some(Box::new(right_child));
                self.len += 1;

                if key > mid_key {
                    idx += 1;
                }
            }

            self.children[idx].as_mut().unwrap().insert_non_full(key, value);
        }
    }

    pub fn split(&mut self) -> (u64, u64, Node) {
        let mid = self.len / 2;

        let mut right = Node::new(self.is_leaf);

        let count = self.len - 1 - mid;

        for i in 0..count {
            right.keys[i] = self.keys[mid + i + 1];
            right.values[i] = self.values[mid + i + 1];
        }

        if !self.is_leaf {
            for i in 0..=count {
                right.children[i] = self.children[mid + i + 1].take();
            }
        }

        self.len = mid;
        right.len = count;

        let median_key = self.keys[mid];
        let median_value = self.values[mid];

        (median_key, median_value, right)
    }
}

pub struct BTree {
    root: Node,
}

impl BTree {
    pub fn new() -> Self {
        Self {
            root: Node::new(true),
        }
    }

    pub fn insert(&mut self, key: u64, value: u64) {
        if self.root.len == CAPACITY {
            let new_root = Node::new(false);

            let old_root = mem::replace(&mut self.root, new_root);
            self.root.children[0] = Some(Box::new(old_root));
        }

        self.root.insert_non_full(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_split() {
        let mut tree = BTree::new();

        for i in 1..=11 {
            tree.insert(i * 10, i * 100);
        }

        assert_eq!(tree.root.len, 11);
        assert!(tree.root.is_leaf);

        tree.insert(120, 1200);

        assert_eq!(tree.root.is_leaf, false);
        assert_eq!(tree.root.len, 1);
        assert!(tree.root.children[0].is_some());
        assert!(tree.root.children[1].is_some());

        println!("Root split successful!");
    }
}