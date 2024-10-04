fn main() {
  println!("cargo:rerun-if-changed=fathom");
  cc::Build::new()
    .include("fathom/src")
    .file(   "fathom/src/tbprobe.c")
    .file(   "fathom/src/inline-wrapper.c")
    .compile("fathom")
}
