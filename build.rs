fn main() {
    // Link to the OpenCL library
    println!("cargo:rustc-link-lib=OpenCL");

    // Specify the directory where the OpenCL library is located, if necessary
    println!("cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\lib\\x64");
}
