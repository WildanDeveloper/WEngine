fn main() {
    println!("cargo:rustc-link-search=native=/root/WEngine/native");
    println!("cargo:rustc-link-lib=dylib=wildandev_accel");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/root/WEngine/native");
}
