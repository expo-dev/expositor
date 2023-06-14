#![allow(dead_code)]
// #![feature(const_maybe_uninit_zeroed)]
#![feature(io_error_other)]
#![feature(iter_intersperse)]
#![feature(maybe_uninit_uninit_array)]
#![feature(portable_simd)]

mod algebraic ;
mod apply     ;
mod basis     ;
mod cache     ;
mod color     ;
mod constants ;
mod context   ;
mod datagen   ;
mod default   ;
mod dest      ;
mod endgame   ;
mod exchange  ;
mod fen       ;
mod formats   ;
mod global    ;
// mod hce       ;
mod import    ;
mod limits    ;
mod misc      ;
mod movegen   ;
mod movesel   ;
mod movetype  ;
mod nnue      ;
mod nonsense  ;
mod perft     ;
mod piece     ;
mod quick     ;
mod rand      ;
mod resolve   ;
mod score     ;
mod search    ;
mod show      ;
mod simd      ;
mod span      ;
mod state     ;
mod syzygy    ;
mod tablebase ;
mod training  ;
mod uci       ;
mod util      ;
mod zobrist   ;

// TODO less undefined behavior
// TODO remove unnecessary uses of .iter()
//   (use implicit iteration or .into_iter() instead as appropriate)
// TODO remove unnecessary uses of &
// TODO integer division in Rust rounds toward zero, and so division of a signed integer by
//   a power of two cannot be optimized into only an arithmetic right shift. Go through the
//   code and check all instances of integer division to make sure they have the proper
//   semantics and are fully optimizable.
// TODO for clearing the last bit, consider a function.
//   (No need for an intrinsic; LLVM recognizes the pattern.)
// TODO go through uses of unwrap and see if let Some or ? can be used
// TODO decide when panic should or should not be used
// TODO consistent comment styling
// TODO remove unnecessary uses of pub
// TODO review use of inline
// TODO mv implements copy, but the use of "mv" versus "&mv" should still indicate ownership

fn main() -> std::io::Result<()>
{
  let mut args = std::env::args();
  args.next();
  if args.next().is_some() {
    eprintln!("{}", uci::HELP);
    return Ok(());
  }
  util::set_stacksize(134_217_728);
  if util::isatty(util::STDERR) {
    eprintln!(
      "Expositor {} \x1B[2mbuilt at {}\x1B[22m",
      util::VERSION, util::BUILD
    );
  }
  dest::generate_tables();
  tablebase::build_3man();
  return uci::uci();
}
