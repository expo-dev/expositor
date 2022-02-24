set VERSION="2WQ23+win"
set BUILD=%time% on %date%
set RUSTFLAGS=-C target-cpu=native
cargo build --release
