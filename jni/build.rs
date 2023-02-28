use json;

fn main() {
    json::parse(include_str!("../tiktoken/registry.json")).expect("Failed to parse internal JSON");
    json::parse(include_str!("../tiktoken/model_to_encoding.json")).expect("Failed to parse internal JSON");
    println!("JSON Parsing validated");
}
