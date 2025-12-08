use sound_btree::Node;

fn main() {
    let mut node = Node::new(true);

    // node.keys[0] = 10;
    // node.keys[1] = 20;
    // node.keys[2] = 30;
    // node.len = 3;
    //
    // let result_found = node.search_node(20);
    // println!("Searching for 20: {:?}", result_found);
    //
    // let result_missing = node.search_node(15);
    // println!("Searching for 15: {:?}", result_missing);

    // println!("Inserting 20...");
    // node.insert_non_full(20, 200);
    //
    // println!("Inserting 10...");
    // node.insert_non_full(10, 100); // Should go to the front
    //
    // println!("Inserting 30...");
    // node.insert_non_full(30, 300); // Should go to the back
    //
    // println!("Inserting 15...");
    // node.insert_non_full(15, 150); // Should go in the middle
    //
    // // Print the valid keys
    // // We use the slice [0.len] to hide the garbage zeros
    // println!("Resulting Keys: {:?}", &node.keys[0..node.len]);

    for i in 1..=11 {
        node.insert_non_full(i * 10, i * 100);
    }
    println!("Full Node: {:?}", &node.keys[0..node.len]);

    // 2. Split it!
    let (mid_key, _, right_node) = node.split();

    println!("Split Complete!");
    println!("Left Node Keys: {:?}", &node.keys[0..node.len]);
    println!("Middle Key (Promoted): {}", mid_key);
    println!("Right Node Keys: {:?}", &right_node.keys[0..right_node.len]);
}