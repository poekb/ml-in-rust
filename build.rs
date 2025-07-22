use std::path::PathBuf;
use std::{env, fs};

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/");

    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc_path = PathBuf::from(cuda_home).join("bin").join("nvcc");

    if !nvcc_path.exists() {
        panic!(
            "nvcc not found at {:?}. Please set the CUDA_HOME environment variable.",
            nvcc_path
        );
    }

    let kernel_dir = "src/kernels";
    let kernel_files = find_cuda_kernels(&kernel_dir).unwrap();
    for kernel in kernel_files {
        println!("cargo:rerun-if-changed={}", kernel.to_str().unwrap());
        println!("Compiling CUDA kernel: {:?}", kernel);
        let output = std::process::Command::new(&nvcc_path)
            .arg(&kernel)
            .arg("-ptx")
            .arg("-o")
            .arg(kernel.with_extension("ptx"))
            .output()
            .expect("Failed to compile CUDA kernel");

        if !output.status.success() {
            panic!(
                "nvcc failed to compile {:?}:\n{}",
                kernel,
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

fn find_cuda_kernels(dir: &str) -> std::io::Result<Vec<PathBuf>> {
    let mut kernels = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            kernels.extend(find_cuda_kernels(path.to_str().unwrap())?);
        } else if path.extension().map_or(false, |e| e == "cu") {
            kernels.push(path);
        }
    }
    Ok(kernels)
}
