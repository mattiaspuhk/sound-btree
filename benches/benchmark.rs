use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::prelude::*;
use sound_btree::BTree;
use std::collections::BTreeMap;

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("Search Operations");

    let n = 10_000;
    let mut rng = rand::thread_rng();
    let keys: Vec<u64> = (0..n).map(|_| rng.gen_range(0..100_000)).collect();

    let sound_tree = BTree::<u64, u64>::new();
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
        b.iter(|| {
            let tree = BTree::<u64, u64>::new();
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
