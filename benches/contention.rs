//! High-Contention Benchmark: Result vs Unwind OLC Retry Strategies
//!
//! This benchmark is specifically designed to force frequent OLC validation failures
//! by having multiple threads hammer the same small key range. Under normal usage,
//! conflicts are rare and both approaches perform similarly. This benchmark reveals
//! the differences.
//!
//! ## What We're Measuring
//!
//! 1. **Happy Path Performance**: When no conflicts occur, how much overhead does
//!    each approach add? (The `?` branches vs clean code path)
//!
//! 2. **Retry Path Performance**: When conflicts DO occur, how expensive is each
//!    retry mechanism? (Result return vs stack unwinding)
//!
//! 3. **Mixed Workload**: Realistic mix of reads and writes under contention.
//!
//! ## Expected Results
//!
//! - **Low contention**: Both approaches similar, slight edge to Unwind (fewer branches)
//! - **High contention**: Result wins because unwinding is ~100-1000x slower than return
//! - **Very high contention**: The retry overhead dominates, Result significantly faster
//!
//! ## CPU Mechanics at Play
//!
//! ### Result Approach (Branch Prediction)
//!
//! Each `?` operator compiles to a conditional branch:
//! ```asm
//! test rax, rax      ; check if Result is Err
//! jne .error_path    ; branch if error
//! ```
//!
//! Modern CPUs predict branches based on history. For OLC:
//! - Happy path (no conflict) is predicted correctly ~99%+ of the time
//! - When mispredicted, costs ~15-20 cycles per misprediction
//! - With N stack frames, worst case is N mispredictions on retry
//!
//! ### Unwind Approach (DWARF Exception Tables)
//!
//! Panic unwinding uses a different mechanism entirely:
//! 1. `panic_any()` triggers the unwinding runtime
//! 2. Runtime consults DWARF `.eh_frame` tables (or SEH on Windows)
//! 3. For each frame, it:
//!    - Looks up the frame's unwind info in exception tables
//!    - Calls destructors (personality routine)
//!    - Restores callee-saved registers
//!    - Jumps to the next frame
//!
//! This is MUCH slower than returning (~microseconds vs nanoseconds) but:
//! - Zero overhead on happy path (no branches at all)
//! - Exception tables are only consulted when unwinding
//!
//! ### Net Effect
//!
//! | Scenario              | Result                    | Unwind                  |
//! |-----------------------|---------------------------|-------------------------|
//! | Happy path (1 frame)  | 1 branch                  | 0 branches              |
//! | Happy path (N frames) | N branches                | 0 branches              |
//! | Retry (1 frame)       | ~20 cycles                | ~1000+ cycles           |
//! | Retry (N frames)      | ~20*N cycles              | ~1000*N cycles          |
//!
//! The crossover point depends on:
//! - Tree depth (N)
//! - Conflict rate
//! - Branch predictor accuracy

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use rand::prelude::*;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use sound_btree::olc_result::BTreeResult;
use sound_btree::olc_unwind::BTreeUnwind;
use sound_btree::BTree;

/// Configuration for contention benchmarks
struct ContentionConfig {
    /// Number of threads
    threads: usize,
    /// Key range size (smaller = more contention)
    key_range: u64,
    /// Ratio of writes to total operations (0.0 - 1.0)
    write_ratio: f64,
    /// Number of operations per thread per iteration
    ops_per_thread: usize,
}

/// High contention benchmark group
fn bench_high_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("OLC High Contention");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(5));

    // Test configurations: (threads, key_range, write_ratio)
    let configs = [
        // Very high contention: 8 threads, 10 keys, 50% writes
        ("extreme", ContentionConfig {
            threads: 8,
            key_range: 10,
            write_ratio: 0.5,
            ops_per_thread: 100,
        }),
        // High contention: 8 threads, 100 keys, 20% writes
        ("high", ContentionConfig {
            threads: 8,
            key_range: 100,
            write_ratio: 0.2,
            ops_per_thread: 100,
        }),
        // Medium contention: 4 threads, 1000 keys, 10% writes
        ("medium", ContentionConfig {
            threads: 4,
            key_range: 1000,
            write_ratio: 0.1,
            ops_per_thread: 100,
        }),
        // Low contention: 4 threads, 10000 keys, 5% writes
        ("low", ContentionConfig {
            threads: 4,
            key_range: 10000,
            write_ratio: 0.05,
            ops_per_thread: 100,
        }),
    ];

    for (name, config) in &configs {
        // Pre-populate trees
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

        // Benchmark Original (continue 'restart)
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

        // Benchmark Result approach
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

        // Benchmark Unwind approach
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

/// Single-threaded search benchmark (pure happy path comparison)
fn bench_single_thread_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("OLC Single Thread Search (Happy Path)");

    let n = 10_000u64;

    // Setup trees
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

/// Read-heavy contention (tests happy path under concurrent load)
fn bench_read_heavy_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("OLC Read-Heavy (95% reads)");
    group.sample_size(20);

    let config = ContentionConfig {
        threads: 8,
        key_range: 100,
        write_ratio: 0.05, // 5% writes
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
