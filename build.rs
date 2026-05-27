fn main() {
    #[cfg(feature = "uniffi")]
    uniffi_build::generate_scaffolding("src/tiktoken.udl").unwrap();
}