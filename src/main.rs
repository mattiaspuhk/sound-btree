use sound_btree::Node;

fn main() {
    let node = Node::new(true);
    println!("Node created! Length: {}", node.len);
}