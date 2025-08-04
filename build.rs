fn main() {
    uniffi_build::generate_scaffolding("src/tiktoken.udl").unwrap();
}