#[derive(Debug, Clone)]
pub struct Node<K, V, const ORDER: usize> {
    pub keys: [Option<K>; ORDER],
    pub values: [Option<V>; ORDER],
    pub children: [Option<Box<Node<K, V, ORDER>>>; ORDER + 1],
    pub len: usize,
    pub is_leaf: bool,
}

impl<K, V, const ORDER: usize> Node<K, V, ORDER>
where
    K: Ord + Clone,
    V: Clone,
{
    pub fn new_leaf() -> Self {
        Self { keys: std::array::from_fn(|_| None), values: std::array::from_fn(|_| None), children: std::array::from_fn(|_| None), len: 0, is_leaf: true }
    }

    pub fn new_internal() -> Self {
        Self { keys: std::array::from_fn(|_| None), values: std::array::from_fn(|_| None), children: std::array::from_fn(|_| None), len: 0, is_leaf: false }
    }

    pub fn is_full(&self) -> bool {
        self.len >= ORDER
    }

    pub fn is_minimal(&self) -> bool {
        self.len == (ORDER - 1) / 2
    }
}