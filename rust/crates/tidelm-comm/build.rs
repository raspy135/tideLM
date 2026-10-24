//! Link NCCL from NCCL_HOME (default ~/.local/nccl: the engine image's libnccl 2.28.3 and
//! nccl.h copied out, so the host needs no system package), and build the RoCE one-shot
//! gather: the kernel from its in-tree source (tools/roce/roce.cu, one source of truth with
//! the Python engine) and the torch-free proxy (native/roce_proxy.cpp), against ibverbs.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(cmd: &mut Command) {
    let desc = format!("{cmd:?}");
    let status = cmd.status().unwrap_or_else(|e| panic!("failed to spawn {desc}: {e}"));
    if !status.success() {
        panic!("{desc} failed with {status}");
    }
}

fn main() {
    let home = env::var("NCCL_HOME").unwrap_or_else(|_| format!("{}/.local/nccl", env::var("HOME").unwrap_or_default()));
    println!("cargo:rerun-if-env-changed=NCCL_HOME");
    println!("cargo:rustc-link-search=native={home}/lib");
    println!("cargo:rustc-link-lib=dylib=nccl");
    // this crate's own examples and tests (dependents' binaries get theirs from tidelm-engine)
    println!("cargo:rustc-link-arg=-Wl,-rpath,{home}/lib");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo = manifest.ancestors().nth(3).expect("crate sits at rust/crates/<name>").to_path_buf();
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let roce_dir = repo.join("tools/roce");
    let kernel = roce_dir.join("roce.cu");
    let proxy = manifest.join("native/roce_proxy.cpp");
    for f in [&kernel, &proxy, &roce_dir.join("roce_common.h")] {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    println!("cargo:rerun-if-env-changed=TIDELM_CUDA_ARCH");
    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let nvcc = format!("{cuda_home}/bin/nvcc");
    let nvcc = if Path::new(&nvcc).exists() { nvcc } else { "nvcc".into() };
    let arch = env::var("TIDELM_CUDA_ARCH").unwrap_or_else(|_| "121a".into());
    let inc = format!("-I{}", roce_dir.display());

    let kobj = out.join("roce_kernel.o");
    run(Command::new(&nvcc).args([
        "-std=c++17", "-O3", "-DNDEBUG", "-Xcompiler=-fPIC", "-c", &inc, "-gencode",
        &format!("arch=compute_{arch},code=sm_{arch}"), kernel.to_str().unwrap(), "-o", kobj.to_str().unwrap(),
    ]));
    let pobj = out.join("roce_proxy.o");
    run(Command::new(&nvcc).args([
        "-std=c++17", "-O3", "-DNDEBUG", "-Xcompiler=-fPIC", "-Xcompiler=-pthread", "-c", &inc,
        proxy.to_str().unwrap(), "-o", pobj.to_str().unwrap(),
    ]));
    let archive = out.join("libtidelm_roce.a");
    let _ = std::fs::remove_file(&archive);
    run(Command::new("ar").args(["crs", archive.to_str().unwrap(), kobj.to_str().unwrap(), pobj.to_str().unwrap()]));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=tidelm_roce");
    println!("cargo:rustc-link-lib=dylib=ibverbs");
    for dir in [format!("{cuda_home}/lib64"), format!("{cuda_home}/targets/sbsa-linux/lib")] {
        if Path::new(&dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
