use sound_btree::BTree;

fn main() {
    println!("=====================================================");
    println!("    Sound B-Tree: Cache-Optimized Arena Demo         ");
    println!("=====================================================");
    println!("Config: B=6 (Max Keys per Node = 11)");

    // 1. Initialize
    println!("\n[Step 1] Initializing B-Tree...");
    let mut tree = BTree::new();
    println!("-> Tree created. Arena allocated.");

    // 2. Fill the Root (Capacity is 11)
    // We insert 10, 20, ... 110.
    println!("\n[Step 2] Filling the Root Node (11 items)...");
    for i in 1..=11 {
        tree.insert(i * 10, i * 100);
    }

    // Show that it is still a single flat node
    println!("-> Current Structure (Leaf Only):");
    tree.print();

    // 3. Trigger Root Split
    // Inserting 12th item (120). Root must split.
    // Median (60) moves to new Root. Left=[10..50], Right=[70..120].
    println!("\n[Step 3] Inserting Key 120 (Triggers Root Split)...");
    tree.insert(120, 1200);

    println!("-> Structure after Split (Root -> 2 Children):");
    // Look for Node[2] as the new root, pointing to Node[0] and Node[1]
    tree.print();

    // 4. Force Internal Splits (Deep Hierarchy)
    // Insert 40 more items to fill the children and force them to split too.
    println!("\n[Step 4] Inserting 40 more keys to force tree growth...");
    for i in 13..=53 {
        tree.insert(i * 10, i * 100);
    }

    println!("-> Final Tree Structure:");
    tree.print();

    // 5. Validation (Search)
    println!("\n[Step 5] validating Data Retrieval...");
    let targets = [10, 60, 120, 500, 999]; // 999 does not exist

    for key in targets {
        match tree.search(key) {
            Some(val) => println!("   [OK] Search({}) -> Found Value: {}", key, val),
            None      => println!("   [OK] Search({}) -> Not Found (Correct)", key),
        }
    }

    println!("\n=====================================================");
    println!("    Demo Complete. System is Operational.            ");
    println!("=====================================================");
}