# sound_btree

A concurrent B-tree implementation in Rust using Optimistic Lock Coupling (OLC).

## Overview

This crate provides a thread-safe B-tree data structure that supports concurrent reads and writes without global locking. Readers proceed optimistically without acquiring locks, validating their reads afterward and retrying if a concurrent modification is detected.

## Features

- **Lock-free reads**: Readers don't block writers and vice versa under low contention
- **Fine-grained locking**: Writers use per-node locks instead of a global lock
- **Generic key/value types**: Works with any `Copy + Ord + Default` keys and `Copy + Default` values
- **Configurable node capacity**: Tune the branching factor via const generics
- **Miri-tested**: Verified for soundness under Stacked Borrows

## Usage

```rust
use sound_btree::BTree;
use std::sync::Arc;
use std::thread;

// Create a new tree
let tree = BTree::<u64, u64>::new();

// Insert values
tree.insert(1, 100);
tree.insert(2, 200);
tree.insert(3, 300);

// Search for values
assert_eq!(tree.search(1), Some(100));
assert_eq!(tree.search(4), None);

// Delete values
assert_eq!(tree.delete(2), Some(200));

// Iterate in sorted order
for (key, value) in &tree {
    println!("{}: {}", key, value);
}

// Thread-safe concurrent access
let tree = Arc::new(BTree::<u64, u64>::new());
let handles: Vec<_> = (0..4).map(|i| {
    let tree = tree.clone();
    thread::spawn(move || {
        for j in 0..1000 {
            tree.insert(i * 1000 + j, j);
        }
    })
}).collect();

for h in handles {
    h.join().unwrap();
}
```

## API

### BTree

| Method | Description |
|--------|-------------|
| `new()` | Create a tree with default capacity (10,000 nodes) |
| `with_capacity(n)` | Create a tree with space for `n` nodes |
| `insert(k, v)` | Insert or update a key-value pair |
| `search(k)` | Look up a value by key |
| `delete(k)` | Remove a key, returning its value if present |
| `contains_key(k)` | Check if a key exists |
| `len()` | Number of entries |
| `is_empty()` | Whether the tree is empty |
| `clear()` | Remove all entries |
| `iter()` | Iterate over entries in sorted order |

## Project Structure

```
src/
  lib.rs          - Public API, re-exports, and tests
  node.rs         - Node, NodeId, NodeWriteGuard, OLC primitives
  tree.rs         - BTree struct and core operations
  iter.rs         - Iterator implementation
  olc_result.rs   - Alternative: Result-based OLC retry strategy
  olc_unwind.rs   - Alternative: panic/catch_unwind OLC retry strategy

benches/
  benchmark.rs    - Performance benchmarks vs std::collections::BTreeMap
  contention.rs   - High-contention concurrent benchmarks
```

## OLC Retry Strategies

The crate includes two experimental modules demonstrating different approaches to handling OLC validation failures:

### Main implementation (inline restarts)
The default `BTree` uses `loop`/`continue` for restarts. This has the best performance because the compiler can optimize the happy path without any branch overhead.

### `olc_result` module
Uses `Result<T, VersionMismatch>` and the `?` operator to propagate validation failures. Cleaner code structure but adds branch instructions on every operation (~27% overhead in benchmarks).

### `olc_unwind` module
Uses `panic_any(VersionMismatch)` with `catch_unwind` for control flow. Zero overhead on the happy path because unwinding uses DWARF exception tables instead of conditional branches. The retry cost is higher but only matters under contention.

## Memory Management

The tree uses a pre-allocated arena that grows monotonically. Deleted nodes are not recycled - this is a deliberate safety constraint to avoid data races with concurrent readers. Call `clear()` to reset the tree and reuse the initial node.

For applications requiring memory reclamation, consider wrapping with `crossbeam-epoch` or implementing epoch-based reclamation.

## Running Tests

```bash
# Run all tests
cargo test

# Run with Miri for soundness verification
cargo +nightly miri test

# Run benchmarks
cargo bench
```

## License

MIT
