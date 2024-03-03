set VERSION=4WR02+win
set BUILD=built at %time% on %date%
set RUSTFLAGS=-C target-cpu=native -C link-args=/STACK:33554432
cargo build --release
