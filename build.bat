set VERSION=2WN29+win
set BUILD=%time% on %date%
set RUSTFLAGS=-C target-cpu=native -C link-args=/STACK:16777216
cargo build --release
