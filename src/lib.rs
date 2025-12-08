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
        match self.search_node(key) {
            Ok(index) => {
                self.keys[index] = key;
            }
            Err(index) => {
                if self.len == CAPACITY {
                    panic!("Node is full");
                }

                for i in (index..self.len).rev() {
                    self.keys[i + 1] = self.keys[i];
                    self.values[i + 1] = self.values[i];
                }

                self.keys[index] = key;
                self.values[index] = value;

                self.len += 1;
            }
        }
    }
}