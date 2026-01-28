use criterion::{Criterion, black_box, criterion_group, criterion_main, BatchSize};
use rand::prelude::*;
use sound_btree::BTree;
use std::collections::BTreeMap;

type SoundBTree = BTree<u64, u64>;

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("Search Operations");

    let n = 10_000;
    let mut rng = rand::thread_rng();
    let keys: Vec<u64> = (0..n).map(|_| rng.gen_range(0..100_000)).collect();

    let sound_tree: SoundBTree = BTree::new();
    for &k in &keys {
        sound_tree.insert(k, k);
    }

    let mut std_tree = BTreeMap::new();
    for &k in &keys {
        std_tree.insert(k, k);
    }

    group.bench_function("SoundBTree", |b| {
        b.iter(|| {
            for &k in &keys {
                black_box(sound_tree.search(k));
            }
        })
    });

    group.bench_function("StdBTreeMap", |b| {
        b.iter(|| {
            for &k in &keys {
                black_box(std_tree.get(&k));
            }
        })
    });

    group.finish();
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("Random Insertion");

    let mut rng = rand::thread_rng();
    let random_keys: Vec<u64> = (0..1000).map(|_| rng.gen_range(0..10000)).collect();

    group.bench_function("SoundBTree", |b| {
        b.iter_batched(
            || -> SoundBTree { BTree::new() },
            |tree| {
                for k in &random_keys {
                    tree.insert(black_box(*k), black_box(*k));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("StdBTreeMap", |b| {
        b.iter_batched(
            || BTreeMap::new(),
            |mut tree| {
                for k in &random_keys {
                    tree.insert(black_box(*k), black_box(*k));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_incremental_insert(c: &mut Criterion) {
    use std::sync::atomic::{AtomicU64, Ordering};

    let mut group = c.benchmark_group("Incremental Insert (into existing tree)");

    group.bench_function("SoundBTree", |b| {
        let counter = AtomicU64::new(10000);
        b.iter_batched(
            || {
                let tree: SoundBTree = BTree::with_capacity(20_000);
                for i in 0..5000u64 {
                    tree.insert(i, i);
                }
                tree
            },
            |tree| {
                for _ in 0..100 {
                    let k = counter.fetch_add(1, Ordering::Relaxed);
                    tree.insert(black_box(k), black_box(k));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("StdBTreeMap", |b| {
        let counter = AtomicU64::new(10000);
        b.iter_batched(
            || {
                let mut tree = BTreeMap::new();
                for i in 0..5000u64 {
                    tree.insert(i, i);
                }
                tree
            },
            |mut tree| {
                for _ in 0..100 {
                    let k = counter.fetch_add(1, Ordering::Relaxed);
                    tree.insert(black_box(k), black_box(k));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("Delete Operations");

    let n = 1000;
    let keys: Vec<u64> = (0..n).collect();

    group.bench_function("SoundBTree", |b| {
        b.iter_batched(
            || {
                let tree: SoundBTree = BTree::new();
                for &k in &keys {
                    tree.insert(k, k);
                }
                tree
            },
            |tree| {
                for &k in &keys {
                    black_box(tree.delete(k));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("StdBTreeMap", |b| {
        b.iter_batched(
            || {
                let mut tree = BTreeMap::new();
                for &k in &keys {
                    tree.insert(k, k);
                }
                tree
            },
            |mut tree| {
                for &k in &keys {
                    black_box(tree.remove(&k));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_mixed_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mixed Operations (insert/search/delete)");

    let mut rng = rand::thread_rng();
    let keys: Vec<u64> = (0..1000).map(|_| rng.gen_range(0..5000)).collect();

    group.bench_function("SoundBTree", |b| {
        b.iter_batched(
            || {
                let tree: SoundBTree = BTree::new();
                for i in 0..2500u64 {
                    tree.insert(i, i);
                }
                tree
            },
            |tree| {
                for &k in &keys {
                    tree.insert(black_box(k), black_box(k));
                    black_box(tree.search(k));
                    tree.delete(black_box(k));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("StdBTreeMap", |b| {
        b.iter_batched(
            || {
                let mut tree = BTreeMap::new();
                for i in 0..2500u64 {
                    tree.insert(i, i);
                }
                tree
            },
            |mut tree| {
                for &k in &keys {
                    tree.insert(black_box(k), black_box(k));
                    black_box(tree.get(&k));
                    tree.remove(&black_box(k));
                }
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_insert, bench_search, bench_incremental_insert, bench_delete, bench_mixed_operations);
criterion_main!(benches);
