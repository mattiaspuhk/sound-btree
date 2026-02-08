use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use rand::prelude::*;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::sync::Once;

use sound_btree::olc_result::BTreeResult;
use sound_btree::olc_unwind::BTreeUnwind;
use sound_btree::BTree;

// Suppress panic messages from olc_unwind (intentional panics for control flow)
static INIT_PANIC_HOOK: Once = Once::new();

fn suppress_version_mismatch_panics() {
    INIT_PANIC_HOOK.call_once(|| {
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if let Some(location) = info.location() {
                if location.file().contains("olc_unwind") {
                    return;
                }
            }
            default_hook(info);
        }));
    });
}

struct ContentionConfig {
    threads: usize,
    key_range: u64,
    write_ratio: f64,
    ops_per_thread: usize,
}

fn bench_high_contention(c: &mut Criterion) {
    suppress_version_mismatch_panics();
    let mut group = c.benchmark_group("OLC High Contention");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(5));

    let configs = [
        ("extreme", ContentionConfig {
            threads: 8,
            key_range: 10,
            write_ratio: 0.5,
            ops_per_thread: 100,
        }),
        ("high", ContentionConfig {
            threads: 8,
            key_range: 100,
            write_ratio: 0.2,
            ops_per_thread: 100,
        }),
        ("medium", ContentionConfig {
            threads: 4,
            key_range: 1000,
            write_ratio: 0.1,
            ops_per_thread: 100,
        }),
        ("low", ContentionConfig {
            threads: 4,
            key_range: 10000,
            write_ratio: 0.05,
            ops_per_thread: 100,
        }),
    ];

    for (name, config) in &configs {
        let setup_original = || {
            let tree = Arc::new(BTree::<u64, u64>::with_capacity(50_000));
            for i in 0..config.key_range {
                tree.insert(i, i);
            }
            tree
        };

        let setup_result = || {
            let tree = Arc::new(BTreeResult::<u64, u64>::with_capacity(50_000));
            for i in 0..config.key_range {
                tree.insert(i, i);
            }
            tree
        };

        let setup_unwind = || {
            let tree = Arc::new(BTreeUnwind::<u64, u64>::with_capacity(50_000));
            for i in 0..config.key_range {
                tree.insert(i, i);
            }
            tree
        };

        group.throughput(Throughput::Elements(
            (config.threads * config.ops_per_thread) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::new("Original", name),
            config,
            |b, config| {
                let tree = setup_original();
                b.iter(|| {
                    let mut handles = vec![];

                    for _ in 0..config.threads {
                        let tree_ref = tree.clone();
                        let key_range = config.key_range;
                        let write_ratio = config.write_ratio;
                        let ops = config.ops_per_thread;
                        handles.push(thread::spawn(move || {
                            let mut rng = thread_rng();
                            for _ in 0..ops {
                                let key = rng.gen_range(0..key_range);
                                if rng.gen_bool(write_ratio) {
                                    tree_ref.insert(key, key);
                                } else {
                                    let _ = tree_ref.search(key);
                                }
                            }
                        }));
                    }

                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Result", name),
            config,
            |b, config| {
                let tree = setup_result();
                b.iter(|| {
                    let mut handles = vec![];

                    for _ in 0..config.threads {
                        let tree_ref = tree.clone();
                        let key_range = config.key_range;
                        let write_ratio = config.write_ratio;
                        let ops = config.ops_per_thread;
                        handles.push(thread::spawn(move || {
                            let mut rng = thread_rng();
                            for _ in 0..ops {
                                let key = rng.gen_range(0..key_range);
                                if rng.gen_bool(write_ratio) {
                                    tree_ref.insert(key, key);
                                } else {
                                    let _ = tree_ref.search(key);
                                }
                            }
                        }));
                    }

                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Unwind", name),
            config,
            |b, config| {
                let tree = setup_unwind();
                b.iter(|| {
                    let mut handles = vec![];

                    for _ in 0..config.threads {
                        let tree_ref = tree.clone();
                        let key_range = config.key_range;
                        let write_ratio = config.write_ratio;
                        let ops = config.ops_per_thread;
                        handles.push(thread::spawn(move || {
                            let mut rng = thread_rng();
                            for _ in 0..ops {
                                let key = rng.gen_range(0..key_range);
                                if rng.gen_bool(write_ratio) {
                                    tree_ref.insert(key, key);
                                } else {
                                    let _ = tree_ref.search(key);
                                }
                            }
                        }));
                    }

                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_single_thread_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("OLC Single Thread Search (Happy Path)");

    let n = 10_000u64;

    let original = BTree::<u64, u64>::with_capacity(20_000);
    let result_tree = BTreeResult::<u64, u64>::with_capacity(20_000);
    let unwind_tree = BTreeUnwind::<u64, u64>::with_capacity(20_000);

    for i in 0..n {
        original.insert(i, i);
        result_tree.insert(i, i);
        unwind_tree.insert(i, i);
    }

    let keys: Vec<u64> = (0..1000).map(|i| i % n).collect();

    group.throughput(Throughput::Elements(keys.len() as u64));

    group.bench_function("Original", |b| {
        b.iter(|| {
            for &k in &keys {
                criterion::black_box(original.search(k));
            }
        });
    });

    group.bench_function("Result", |b| {
        b.iter(|| {
            for &k in &keys {
                criterion::black_box(result_tree.search(k));
            }
        });
    });

    group.bench_function("Unwind", |b| {
        b.iter(|| {
            for &k in &keys {
                criterion::black_box(unwind_tree.search(k));
            }
        });
    });

    group.finish();
}

fn bench_read_heavy_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("OLC Read-Heavy (95% reads)");
    group.sample_size(20);

    let config = ContentionConfig {
        threads: 8,
        key_range: 100,
        write_ratio: 0.05,
        ops_per_thread: 200,
    };

    let setup_original = || {
        let tree = Arc::new(BTree::<u64, u64>::with_capacity(50_000));
        for i in 0..config.key_range {
            tree.insert(i, i);
        }
        tree
    };

    let setup_result = || {
        let tree = Arc::new(BTreeResult::<u64, u64>::with_capacity(50_000));
        for i in 0..config.key_range {
            tree.insert(i, i);
        }
        tree
    };

    let setup_unwind = || {
        let tree = Arc::new(BTreeUnwind::<u64, u64>::with_capacity(50_000));
        for i in 0..config.key_range {
            tree.insert(i, i);
        }
        tree
    };

    group.throughput(Throughput::Elements(
        (config.threads * config.ops_per_thread) as u64,
    ));

    group.bench_function("Original", |b| {
        let tree = setup_original();
        b.iter(|| {
            let mut handles = vec![];

            for _ in 0..config.threads {
                let tree_ref = tree.clone();
                let key_range = config.key_range;
                let write_ratio = config.write_ratio;
                let ops = config.ops_per_thread;
                handles.push(thread::spawn(move || {
                    let mut rng = thread_rng();
                    for _ in 0..ops {
                        let key = rng.gen_range(0..key_range);
                        if rng.gen_bool(write_ratio) {
                            tree_ref.insert(key, key);
                        } else {
                            let _ = tree_ref.search(key);
                        }
                    }
                }));
            }

            for h in handles {
                h.join().unwrap();
            }
        });
    });

    group.bench_function("Result", |b| {
        let tree = setup_result();
        b.iter(|| {
            let mut handles = vec![];

            for _ in 0..config.threads {
                let tree_ref = tree.clone();
                let key_range = config.key_range;
                let write_ratio = config.write_ratio;
                let ops = config.ops_per_thread;
                handles.push(thread::spawn(move || {
                    let mut rng = thread_rng();
                    for _ in 0..ops {
                        let key = rng.gen_range(0..key_range);
                        if rng.gen_bool(write_ratio) {
                            tree_ref.insert(key, key);
                        } else {
                            let _ = tree_ref.search(key);
                        }
                    }
                }));
            }

            for h in handles {
                h.join().unwrap();
            }
        });
    });

    group.bench_function("Unwind", |b| {
        let tree = setup_unwind();
        b.iter(|| {
            let mut handles = vec![];

            for _ in 0..config.threads {
                let tree_ref = tree.clone();
                let key_range = config.key_range;
                let write_ratio = config.write_ratio;
                let ops = config.ops_per_thread;
                handles.push(thread::spawn(move || {
                    let mut rng = thread_rng();
                    for _ in 0..ops {
                        let key = rng.gen_range(0..key_range);
                        if rng.gen_bool(write_ratio) {
                            tree_ref.insert(key, key);
                        } else {
                            let _ = tree_ref.search(key);
                        }
                    }
                }));
            }

            for h in handles {
                h.join().unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_thread_search,
    bench_read_heavy_contention,
    bench_high_contention,
);
criterion_main!(benches);
