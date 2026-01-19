use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::prelude::*;
use sound_btree::BTree;
use std::collections::BTreeMap;

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("Search Operations");

    // 1. Setup: Create a big tree (10,000 items)
    // We do this OUTSIDE the measurement loop.
    let n = 10_000;
    let mut rng = rand::thread_rng();
    let keys: Vec<u64> = (0..n).map(|_| rng.gen_range(0..100_000)).collect();

    // Setup SoundBTree
    let mut sound_tree = BTree::new();
    for &k in &keys {
        sound_tree.insert(k, k);
    }

    // Setup Std BTreeMap
    let mut std_tree = BTreeMap::new();
    for &k in &keys {
        std_tree.insert(k, k);
    }

    // 2. Benchmark just the Searching
    group.bench_function("SoundBTree", |b| {
        b.iter(|| {
            // Pick a random key from our existing set to search for
            // (Using `black_box` to prevent compiler optimizing it away)
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
        b.iter(|| {
            let mut tree = BTree::new();
            for k in &random_keys {
                tree.insert(black_box(*k), black_box(*k));
            }
        })
    });

    group.bench_function("StdBTreeMap", |b| {
        b.iter(|| {
            let mut tree = BTreeMap::new();
            for k in &random_keys {
                tree.insert(black_box(*k), black_box(*k));
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_insert, bench_search);
criterion_main!(benches);
