#![allow(clippy::manual_is_multiple_of)]

mod iter;
mod node;
mod tree;

pub mod olc_result;
pub mod olc_unwind;

pub use iter::Iter;
pub use node::{Node, NodeId, NodeWriteGuard, DEFAULT_CAP, DEFAULT_CHILDREN_CAP};
pub use tree::BTree;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_safe_insert_search() {
        let tree = BTree::<u64, u64>::new();

        tree.insert(10, 100);
        tree.insert(20, 200);
        tree.insert(5, 50);

        assert_eq!(tree.search(10), Some(100));
        assert_eq!(tree.search(5), Some(50));
        assert_eq!(tree.search(20), Some(200));
        assert_eq!(tree.search(999), None);
    }

    #[test]
    fn test_splitting_logic() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..20u64 {
            tree.insert(i, i * 10);
        }
        for i in 0..20u64 {
            assert_eq!(tree.search(i), Some(i * 10));
        }
    }

    #[test]
    fn test_generic_with_i32() {
        let tree = BTree::<i32, i32>::new();
        tree.insert(-10, 100);
        tree.insert(0, 0);
        tree.insert(10, -100);

        assert_eq!(tree.search(-10), Some(100));
        assert_eq!(tree.search(0), Some(0));
        assert_eq!(tree.search(10), Some(-100));
        assert_eq!(tree.search(5), None);
    }

    #[test]
    fn test_len() {
        let tree = BTree::<u64, u64>::new();
        assert_eq!(tree.len(), 0);

        tree.insert(1, 10);
        assert_eq!(tree.len(), 1);

        tree.insert(2, 20);
        tree.insert(3, 30);
        assert_eq!(tree.len(), 3);

        tree.insert(2, 200);
        assert_eq!(tree.len(), 3);

        for i in 10..50u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 43);
    }

    #[test]
    fn test_is_empty() {
        let tree = BTree::<u64, u64>::new();
        assert!(tree.is_empty());

        tree.insert(1, 10);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_contains_key() {
        let tree = BTree::<u64, u64>::new();
        assert!(!tree.contains_key(1));

        tree.insert(1, 10);
        tree.insert(5, 50);
        tree.insert(10, 100);

        assert!(tree.contains_key(1));
        assert!(tree.contains_key(5));
        assert!(tree.contains_key(10));
        assert!(!tree.contains_key(2));
        assert!(!tree.contains_key(100));
    }

    #[test]
    fn test_clear() {
        let tree = BTree::<u64, u64>::new();

        for i in 0..50u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 50);
        assert!(!tree.is_empty());

        tree.clear();

        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
        assert!(!tree.contains_key(0));
        assert!(!tree.contains_key(25));

        tree.insert(100, 1000);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.search(100), Some(1000));
    }

    #[test]
    fn test_delete_single_key() {
        let tree = BTree::<u64, u64>::new();
        tree.insert(10, 100);
        assert_eq!(tree.delete(10), Some(100));
        assert_eq!(tree.search(10), None);
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_delete_nonexistent_key() {
        let tree = BTree::<u64, u64>::new();
        tree.insert(10, 100);
        assert_eq!(tree.delete(20), None);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.search(10), Some(100));
    }

    #[test]
    fn test_delete_from_empty_tree() {
        let tree = BTree::<u64, u64>::new();
        assert_eq!(tree.delete(10), None);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_delete_multiple_keys() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..10u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 10);

        assert_eq!(tree.delete(5), Some(50));
        assert_eq!(tree.len(), 9);
        assert_eq!(tree.search(5), None);

        assert_eq!(tree.delete(0), Some(0));
        assert_eq!(tree.len(), 8);

        assert_eq!(tree.delete(9), Some(90));
        assert_eq!(tree.len(), 7);

        for i in [1, 2, 3, 4, 6, 7, 8] {
            assert_eq!(tree.search(i), Some(i * 10));
        }
    }

    #[test]
    fn test_delete_all_keys() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..50u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 50);

        for i in 0..50u64 {
            assert_eq!(tree.delete(i), Some(i * 10), "Failed to delete key {}", i);
        }
        assert!(tree.is_empty());

        tree.insert(100, 1000);
        assert_eq!(tree.search(100), Some(1000));
    }

    #[test]
    fn test_delete_causes_underflow() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..30u64 {
            tree.insert(i, i * 10);
        }

        for i in 0..30u64 {
            let result = tree.delete(i);
            assert_eq!(result, Some(i * 10), "Failed to delete key {}", i);
            assert_eq!(tree.search(i), None);
        }
        assert!(tree.is_empty());
    }

    #[test]
    fn test_delete_reverse_order() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..50u64 {
            tree.insert(i, i * 10);
        }

        for i in (0..50u64).rev() {
            assert_eq!(tree.delete(i), Some(i * 10));
        }
        assert!(tree.is_empty());
    }

    #[test]
    fn test_delete_alternating() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..100u64 {
            tree.insert(i, i * 10);
        }

        for i in (0..100u64).step_by(2) {
            assert_eq!(tree.delete(i), Some(i * 10));
        }
        assert_eq!(tree.len(), 50);

        for i in (1..100u64).step_by(2) {
            assert_eq!(tree.search(i), Some(i * 10));
        }

        for i in (1..100u64).step_by(2) {
            assert_eq!(tree.delete(i), Some(i * 10));
        }
        assert!(tree.is_empty());
    }

    #[test]
    fn test_delete_then_insert() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..20u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 20);

        assert_eq!(tree.delete(5), Some(50));
        assert_eq!(tree.search(5), None, "Key 5 should be deleted");
        assert_eq!(tree.len(), 19);

        assert_eq!(tree.delete(10), Some(100));
        assert_eq!(tree.search(10), None, "Key 10 should be deleted");
        assert_eq!(tree.len(), 18);

        assert_eq!(tree.delete(15), Some(150));
        assert_eq!(tree.search(15), None, "Key 15 should be deleted");
        assert_eq!(tree.len(), 17);

        tree.insert(5, 500);
        assert_eq!(tree.len(), 18);

        tree.insert(10, 1000);
        assert_eq!(tree.len(), 19);

        tree.insert(15, 1500);
        assert_eq!(tree.len(), 20);

        assert_eq!(tree.search(5), Some(500));
        assert_eq!(tree.search(10), Some(1000));
        assert_eq!(tree.search(15), Some(1500));
    }

    #[test]
    fn test_delete_large_tree() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..1000u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 1000);

        for i in (0..1000u64).step_by(2) {
            assert_eq!(tree.delete(i), Some(i * 10));
        }
        assert_eq!(tree.len(), 500);

        for i in (1..1000u64).step_by(2) {
            assert_eq!(tree.search(i), Some(i * 10));
        }
    }

    #[test]
    fn test_concurrent_delete() {
        use std::sync::Arc;
        use std::thread;

        let tree = Arc::new(BTree::<u64, u64>::new());

        for i in 0..1000u64 {
            tree.insert(i, i * 10);
        }
        assert_eq!(tree.len(), 1000);

        let mut handles = vec![];
        for t in 0..4 {
            let tree_ref = tree.clone();
            handles.push(thread::spawn(move || {
                for i in (t * 250)..((t + 1) * 250) {
                    tree_ref.delete(i as u64);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert!(tree.is_empty());
    }

    #[test]
    fn test_concurrent_insert_delete() {
        use std::sync::Arc;
        use std::thread;

        let tree = Arc::new(BTree::<u64, u64>::new());

        for i in 0..500u64 {
            tree.insert(i, i * 10);
        }

        let mut handles = vec![];

        for t in 0..2 {
            let tree_ref = tree.clone();
            handles.push(thread::spawn(move || {
                for i in (500 + t * 250)..(500 + (t + 1) * 250) {
                    tree_ref.insert(i as u64, (i * 10) as u64);
                }
            }));
        }

        for t in 0..2 {
            let tree_ref = tree.clone();
            handles.push(thread::spawn(move || {
                for i in (t * 250)..((t + 1) * 250) {
                    tree_ref.delete(i as u64);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(tree.len(), 500);

        for i in 500..1000u64 {
            assert_eq!(tree.search(i), Some(i * 10));
        }

        for i in 0..500u64 {
            assert_eq!(tree.search(i), None);
        }
    }

    #[test]
    fn test_concurrent_search_delete() {
        use std::sync::Arc;
        use std::thread;
        use std::sync::atomic::{AtomicBool, Ordering};

        let tree = Arc::new(BTree::<u64, u64>::new());
        let done = Arc::new(AtomicBool::new(false));

        for i in 0..1000u64 {
            tree.insert(i, i * 10);
        }

        let mut handles = vec![];

        for _ in 0..2 {
            let tree_ref = tree.clone();
            let done_ref = done.clone();
            handles.push(thread::spawn(move || {
                while !done_ref.load(Ordering::Relaxed) {
                    for i in 0..1000u64 {
                        let _ = tree_ref.search(i);
                    }
                }
            }));
        }

        let tree_ref = tree.clone();
        let done_ref = done.clone();
        handles.push(thread::spawn(move || {
            for i in 0..1000u64 {
                tree_ref.delete(i);
            }
            done_ref.store(true, Ordering::Relaxed);
        }));

        for h in handles {
            h.join().unwrap();
        }

        assert!(tree.is_empty());
    }

    #[test]
    fn test_iter_empty() {
        let tree = BTree::<u64, u64>::new();
        assert_eq!(tree.iter().count(), 0);
    }

    #[test]
    fn test_iter_single() {
        let tree = BTree::<u64, u64>::new();
        tree.insert(42, 420);

        let items: Vec<_> = tree.iter().collect();
        assert_eq!(items, vec![(42, 420)]);
    }

    #[test]
    fn test_iter_ordered() {
        let tree = BTree::<u64, u64>::new();

        tree.insert(50, 500);
        tree.insert(20, 200);
        tree.insert(80, 800);
        tree.insert(10, 100);
        tree.insert(30, 300);

        let items: Vec<_> = tree.iter().collect();
        assert_eq!(items, vec![
            (10, 100),
            (20, 200),
            (30, 300),
            (50, 500),
            (80, 800),
        ]);
    }

    #[test]
    fn test_iter_large() {
        let tree = BTree::<u64, u64>::new();

        for i in (0..1000u64).rev() {
            tree.insert(i, i * 10);
        }

        let items: Vec<_> = tree.iter().collect();
        assert_eq!(items.len(), 1000);

        for (idx, (k, v)) in items.iter().enumerate() {
            assert_eq!(*k, idx as u64);
            assert_eq!(*v, idx as u64 * 10);
        }
    }

    #[test]
    fn test_iter_for_loop() {
        let tree = BTree::<u64, u64>::new();
        for i in 0..10u64 {
            tree.insert(i, i * 100);
        }

        let mut count = 0;
        for (k, v) in &tree {
            assert_eq!(v, k * 100);
            count += 1;
        }
        assert_eq!(count, 10);
    }

    #[test]
    fn test_iter_concurrent_read() {
        use std::sync::Arc;
        use std::thread;

        let tree = Arc::new(BTree::<u64, u64>::new());
        for i in 0..1000u64 {
            tree.insert(i, i * 10);
        }

        let mut handles = vec![];
        for _ in 0..4 {
            let tree_ref = tree.clone();
            handles.push(thread::spawn(move || {
                let items: Vec<_> = tree_ref.iter().collect();
                assert_eq!(items.len(), 1000);
                for i in 1..items.len() {
                    assert!(items[i].0 > items[i - 1].0);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }
}
