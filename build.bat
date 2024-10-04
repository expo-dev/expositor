set VERSION=5????+win
set BUILD=built at %time% on %date%
set RUSTFLAGS=--codegen target-cpu=native --codegen link-args=/STACK:33554432
cargo build --release
