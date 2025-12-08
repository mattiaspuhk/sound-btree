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
}