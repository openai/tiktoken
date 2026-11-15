fn main() {
    // Apple's new linker (ld_prime, the default on current macOS CI runners)
    // reserves far less Mach-O header padding than the old ld did. Homebrew's
    // post-install relocation rewrites each dylib's LC_ID_DYLIB from
    // "@rpath/<leaf>" to an absolute path under the formula's opt prefix; with
    // only the few dozen bytes ld_prime leaves spare, that rewrite overruns the
    // header and fails with "Updated load commands do not fit in the header".
    // Reserve PATH_MAX per dylib command so the wheels stay relocatable, the
    // same way rpds-py and pyca/cryptography now do.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-headerpad_max_install_names");
    }
}
