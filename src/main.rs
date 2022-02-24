#![allow(dead_code)]
#![allow(non_upper_case_globals)]
#![feature(iter_intersperse)]
#![feature(maybe_uninit_uninit_array)]
#![feature(portable_simd)]

mod algebraic ;
mod apply     ;
mod basis     ;
mod cache     ;
mod color     ;
mod context   ;
mod default   ;
mod dest      ;
mod endgame   ;
mod exchange  ;
mod fen       ;
mod import    ;
mod limits    ;
mod misc      ;
mod movegen   ;
mod movesel   ;
mod movetype  ;
mod nnue      ;
mod perft     ;
mod piece     ;
mod rand      ;
mod regress   ;
mod resolve   ;
mod score     ;
mod search    ;
mod show      ;
mod simd      ;
mod span      ;
mod state     ;
mod test      ;
mod training  ;
mod uci       ;
mod util      ;
mod zobrist   ;

// TODO remove unnecessary uses of .iter()

// TODO remove unnecessary uses of &

// TODO integer division in Rust rounds toward zero, and so division of a signed integer by
//   a power of two cannot be optimized into only an arithmetic right shift. Go through the
//   code and check all instances of integer division to make sure they have the proper
//   semantics and are fully optimizable.

// TODO for toggling color, consider ^ 8. Also consider adding an as_offset function.

// TODO for clearing the last bit, consider a function.
//   (No need for an intrinsic; LLVM recognizes the pattern.)

fn main() -> std::io::Result<()>
{
  util::set_stacksize(16777216);
  if util::isatty(util::STDERR) {
    eprintln!("Expositor {} \x1B[2mbuilt at {}\x1B[22m", util::VERSION, util::BUILD);
  }
  dest::generate_tables();
  cache::initialize_cache(67108864);  // 64 MiB ~ 4 million entries
  return uci::uci();
}
