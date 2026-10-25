//! Compile the repo's native kernel sources exactly as the Python JIT builders do
//! (tools/fp4_moe_cuda.py::_library, engine/engram_native.py::_build), so a Rust and a
//! Python process on the same box run bit-identical kernels. The sources stay where
//! they are — tools/fp4_moe_cuda.cu and engine/engram_gather.cpp — one source of truth.
//!
//! The FP4 defines are the shipped-profile values and are pinned here, not read from
//! DSV41_* environment variables: per-rank compile divergence is exactly what the
//! boot guard exists to prevent, and a build-time env var would reintroduce it.
//! FP4_RELAXED_REDUCE=1 / FP4_V2_FMA_SCALE=0 / FP4_ROWS_PER_WARP=1 match the defaults
//! the Python loader hashes into its cache key.

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
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo = manifest.ancestors().nth(3).expect("crate sits at rust/crates/<name>").to_path_buf();
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    let fp4_src = repo.join("tools/fp4_moe_cuda.cu");
    let engram_src = repo.join("engine/engram_gather.cpp");
    let ds41_src = manifest.join("cuda/ds41.cu");
    let vision_src = manifest.join("cuda/vision.cu");
    let glm_src = manifest.join("cuda/glm.cu");
    let exl3_src = repo.join("tools/exl3_moe_cuda.cu");
    println!("cargo:rerun-if-changed={}", ds41_src.display());
    println!("cargo:rerun-if-changed={}", vision_src.display());
    println!("cargo:rerun-if-changed={}", glm_src.display());
    println!("cargo:rerun-if-changed={}", exl3_src.display());
    println!("cargo:rerun-if-changed={}", fp4_src.display());
    println!("cargo:rerun-if-changed={}", engram_src.display());
    println!("cargo:rerun-if-env-changed=TIDELM_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let nvcc = env::var("NVCC").unwrap_or_else(|_| format!("{cuda_home}/bin/nvcc"));
    let nvcc = if Path::new(&nvcc).exists() { nvcc } else { "nvcc".into() };
    // The architecture-specific `a` target is required for cvt.e2m1x2; nvcc's convenient
    // `-arch=sm_121a` spelling emits generic sm_121 PTX with CUDA 13.0, hence -gencode.
    let arch = env::var("TIDELM_CUDA_ARCH").unwrap_or_else(|_| "121a".into());

    let fp4_obj = out.join("fp4_moe_cuda.o");
    run(Command::new(&nvcc).args([
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-Xcompiler=-fPIC",
        "-c",
        "-DFP4_ROWS_PER_WARP=1",
        "-DFP4_RELAXED_REDUCE=1",
        "-DFP4_V2_FMA_SCALE=0",
        "-gencode",
        &format!("arch=compute_{arch},code=sm_{arch}"),
        fp4_src.to_str().unwrap(),
        "-o",
        fp4_obj.to_str().unwrap(),
    ]));

    // The Rust engine's own decode kernels: no tuning defines, same arch flags.
    let ds41_obj = out.join("ds41.o");
    run(Command::new(&nvcc).args([
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-Xcompiler=-fPIC",
        "-c",
        "-gencode",
        &format!("arch=compute_{arch},code=sm_{arch}"),
        ds41_src.to_str().unwrap(),
        "-o",
        ds41_obj.to_str().unwrap(),
    ]));

    // the vision tower's kernels, the same way
    let vision_obj = out.join("vision.o");
    run(Command::new(&nvcc).args([
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-Xcompiler=-fPIC",
        "-c",
        "-gencode",
        &format!("arch=compute_{arch},code=sm_{arch}"),
        vision_src.to_str().unwrap(),
        "-o",
        vision_obj.to_str().unwrap(),
    ]));

    // GLM-5.3: the engine's own kernels, and the EXL3 trellis kernels exactly as
    // tools/exl3_moe_cuda.py::_library builds them (no tuning defines)
    let mut objs = vec![];
    for (src, name) in [(&glm_src, "glm.o"), (&exl3_src, "exl3_moe_cuda.o")] {
        let obj = out.join(name);
        run(Command::new(&nvcc).args([
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-Xcompiler=-fPIC",
            "-c",
            "-gencode",
            &format!("arch=compute_{arch},code=sm_{arch}"),
            src.to_str().unwrap(),
            "-o",
            obj.to_str().unwrap(),
        ]));
        objs.push(obj);
    }

    let engram_obj = out.join("engram_gather.o");
    run(Command::new("g++").args([
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-fPIC",
        "-pthread",
        "-c",
        engram_src.to_str().unwrap(),
        "-o",
        engram_obj.to_str().unwrap(),
    ]));

    let archive = out.join("libtidelm_kernels_native.a");
    let _ = std::fs::remove_file(&archive);
    run(Command::new("ar").args([
        "crs",
        archive.to_str().unwrap(),
        fp4_obj.to_str().unwrap(),
        ds41_obj.to_str().unwrap(),
        vision_obj.to_str().unwrap(),
        engram_obj.to_str().unwrap(),
        objs[0].to_str().unwrap(),
        objs[1].to_str().unwrap(),
    ]));

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=tidelm_kernels_native");
    for dir in [format!("{cuda_home}/lib64"), format!("{cuda_home}/targets/sbsa-linux/lib")] {
        if Path::new(&dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
