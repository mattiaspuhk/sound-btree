use sound_btree::Node;

fn main() {
    let mut node = Node::new(true);

    node.keys[0] = 10;
    node.keys[1] = 20;
    node.keys[2] = 30;
    node.len = 3;

    let result_found = node.search_node(20);
    println!("Searching for 20: {:?}", result_found);

    let result_missing = node.search_node(15);
    println!("Searching for 15: {:?}", result_missing);
}