use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::prelude::*;
use sound_btree::BTree;
use std::collections::BTreeMap;

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

criterion_group!(benches, bench_insert);
criterion_main!(benches);
