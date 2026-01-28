//! Iterator implementation for the concurrent B-tree.

use std::sync::atomic::Ordering;

use crate::node::{NodeId, DEFAULT_CAP, DEFAULT_CHILDREN_CAP};
use crate::tree::BTree;

/// An iterator over the key-value pairs of a BTree.
///
/// The iterator uses optimistic locking and will automatically restart
/// from the last successfully returned key if it detects a concurrent
/// modification.
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
    pub(crate) fn new(tree: &'a BTree<K, V, CAP, CHILDREN_CAP>) -> Self {
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
