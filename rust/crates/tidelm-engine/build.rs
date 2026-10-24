//! The binaries link NCCL through tidelm-comm; a library's build script cannot put an rpath
//! on its dependents' executables, so it goes here (NCCL_HOME, default ~/.local/nccl).
fn main() {
    let home = std::env::var("NCCL_HOME")
        .unwrap_or_else(|_| format!("{}/.local/nccl", std::env::var("HOME").unwrap_or_default()));
    println!("cargo:rerun-if-env-changed=NCCL_HOME");
    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{home}/lib");
}
