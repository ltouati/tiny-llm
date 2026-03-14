use std::env;
use std::path::PathBuf;
use std::process::Command;

fn compile_cuda_kernel(filename: &str) {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let name = filename.split('.').next().unwrap();
    let ptx_path = format!("{}/{}.ptx", out_dir, name);
    let src_path = format!("src/{}", filename);

    if PathBuf::from(&src_path).exists() {
        println!("cargo:warning=Compiling custom {} CUDA kernel...", name);
        let build_status = Command::new("nvcc")
            .args([
                "--ptx",
                "-O3",
                "-allow-unsupported-compiler",
                "-arch=compute_80",
                "-o",
                &ptx_path,
                &src_path,
            ])
            .status()
            .unwrap_or_else(|e| panic!("Failed to run nvcc: {}", e));

        if !build_status.success() {
            panic!("Failed to compile {} to PTX.", src_path);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=requirements.txt");
    println!("cargo:rerun-if-changed=src/adamw.cu");
    println!("cargo:rerun-if-changed=src/cross_entropy.cu");
    println!("cargo:rerun-if-changed=src/rope.cu");

    compile_cuda_kernel("adamw.cu");
    compile_cuda_kernel("cross_entropy.cu");
    compile_cuda_kernel("rope.cu");
}
