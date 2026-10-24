use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    for dir in [format!("{cuda_home}/lib64"), format!("{cuda_home}/targets/sbsa-linux/lib")] {
        if Path::new(&dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
}
