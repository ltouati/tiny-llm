use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=prep_data.py");
    println!("cargo:rerun-if-changed=requirements.txt");
    println!("cargo:rerun-if-changed=src/adamw.cu");

    prepare_dataset();
    compile_cuda_kernel();
}

fn prepare_dataset() {
    let bin_file = "fineweb_edu.bin";
    let tokenizer_file = "tokenizer.json";

    if fs::metadata(bin_file).is_ok() && fs::metadata(tokenizer_file).is_ok() {
        println!("cargo:warning=Dataset and tokenizer already exist. Skipping Python prep script.");
        return;
    }

    println!("cargo:warning=Dataset not found. Setting up Python venv and downloading data (this may take a few minutes)...");
    let venv_dir = PathBuf::from(".venv");
    if !venv_dir.exists() {
        println!("cargo:warning=Creating Python virtual environment in .venv...");
        let status = Command::new("python3")
            .args(["-m", "venv", ".venv"])
            .status()
            .unwrap_or_else(|_| {
                Command::new("python")
                    .args(["-m", "venv", ".venv"])
                    .status()
                    .expect("Failed to create venv. Is Python installed and in your PATH?")
            });

        if !status.success() {
            panic!("Failed to create Python virtual environment.");
        }
    }

    let python_exe = if cfg!(target_os = "windows") {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    };

    println!("cargo:warning=Installing Python dependencies from requirements.txt...");
    let pip_status = Command::new(&python_exe)
        .args(["-m", "pip", "install", "-r", "requirements.txt"])
        .status()
        .expect("Failed to execute pip install");

    if !pip_status.success() {
        panic!("Failed to install requirements. Check your internet connection.");
    }

    println!("cargo:warning=Running prep_data.py. Streaming dataset from HuggingFace...");
    let prep_status = Command::new(&python_exe)
        .arg("prep_data.py")
        .status()
        .expect("Failed to execute prep_data.py");

    if !prep_status.success() {
        panic!("prep_data.py failed to complete. Check the Python script for errors.");
    }

    println!("cargo:warning=Data preparation complete!");
}

fn compile_cuda_kernel() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let ptx_path = format!("{}/adamw.ptx", out_dir);

    if PathBuf::from("src/adamw.cu").exists() {
        println!("cargo:warning=Compiling custom Fused AdamW CUDA kernel...");
        let build_status = Command::new("nvcc")
            .args([
                "--ptx",
                "-O3",
                "-allow-unsupported-compiler",
                "-o",
                &ptx_path,
                "src/adamw.cu",
            ])
            .status()
            .unwrap_or_else(|e| panic!("Failed to run nvcc: {}", e));

        if !build_status.success() {
            panic!("Failed to compile adamw.cu to PTX.");
        }
    }
}
