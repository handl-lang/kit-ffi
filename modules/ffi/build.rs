//! Build script for plugin-ffi.
//!
//! On Linux, export dynamic symbols so that FFI tests can find `#[no_mangle]
//! extern "C"` test functions via `Library::this()`. Without this linker flag,
//! the test binary's symbols are not in the dynamic symbol table and `dlsym`
//! cannot find them.

fn main() {
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-arg=-Wl,--export-dynamic");
    }
    #[cfg(target_os = "macos")]
    {
        // macOS exports symbols by default; no extra flag needed.
    }
}
