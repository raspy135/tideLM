//! Link NCCL from NCCL_HOME (default ~/.local/nccl: the engine image's libnccl 2.28.3 and
//! nccl.h copied out, so the host needs no system package). The rpath keeps the binary
//! runnable on both boxes without LD_LIBRARY_PATH.
fn main() {
    let home = std::env::var("NCCL_HOME")
        .unwrap_or_else(|_| format!("{}/.local/nccl", std::env::var("HOME").unwrap_or_default()));
    println!("cargo:rerun-if-env-changed=NCCL_HOME");
    println!("cargo:rustc-link-search=native={home}/lib");
    println!("cargo:rustc-link-lib=dylib=nccl");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{home}/lib");
}
