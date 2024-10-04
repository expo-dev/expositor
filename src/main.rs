#![allow(dead_code)]
#![allow(incomplete_features)]
#![allow(static_mut_refs)]
#![feature(bigint_helper_methods)]
#![feature(generic_const_exprs)]
#![feature(iter_intersperse)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(portable_simd)]

mod apply      ;
mod basis      ;
mod cache      ;
mod color      ;
mod constants  ;
mod context    ;
mod conv       ;
mod datagen    ;
mod default    ;
mod dest       ;
mod endgame    ;
mod exchange   ;
mod fen        ;
mod formats    ;
mod global     ;
// mod hce     ;
mod import     ;
mod limits     ;
mod misc       ;
mod movegen    ;
mod movesel    ;
mod movetype   ;
mod nnext      ;
// mod nnue    ;
mod perft      ;
mod piece      ;
// mod policy  ;
// mod proof   ;
mod quick      ;
// mod quant   ;
mod rand       ;
mod resolve    ;
mod score      ;
mod search     ;
mod show       ;
mod simd       ;
mod simplex    ;
mod span       ;
mod state      ;
mod syzygy     ;
mod tablebase  ;
mod term       ;
// mod training;
mod uci        ;
mod util       ;
mod zobrist    ;

// TODO fix undefined behavior
// TODO remove unnecessary uses of .iter()
//   (use implicit iteration or .into_iter() instead as appropriate)
// TODO remove unnecessary uses of &
// TODO integer division in Rust rounds toward zero, and so division of a signed
//   integer by a power of two cannot be optimized into only an arithmetic right
//   shift. Go through the code and check all instances of integer division to
//   make sure they have the proper semantics and are fully optimizable.
// TODO consider a function taking a mutable reference for clearing the last bit
//   (no need for an intrinsic – LLVM recognizes the pattern)
// TODO go through uses of unwrap and see if let Some or ? can be used
// TODO decide when panic should or should not be used
// TODO consistent comment styling
// TODO remove unnecessary uses of pub
// TODO review use of inline
// TODO mv implements copy, but the use of "mv" versus "&mv" should still
//   indicate ownership
// TODO use "if let Ctor(ref name) = obj" more often – it's cool

fn main() -> std::io::Result<()>
{
  let mut args = std::env::args();
  args.next();
  if args.next().is_some() {
    eprintln!("{}", uci::HELP);
    return Ok(());
  }
  if util::isatty(util::STDERR) {
    #[cfg(debug_assertions)]
    eprintln!(
      "Expositor {} \x1B[91mdebug\x1B[39m \x1B[2m{}\x1B[22m",
      util::VERSION, util::BUILD
    );
    #[cfg(not(debug_assertions))]
    eprintln!("Expositor {} \x1B[2m{}\x1B[22m",
      util::VERSION, util::BUILD
    );
  }
  util::set_stacksize(134_217_728);
  dest::generate_tables();
  tablebase::build_3man();
  unsafe {
    nnext::FNETWORK = std::mem::transmute(*crate::default::DEFAULT_NETWORK);
    nnext::QUANTIZED.emulate(&*(&raw const nnext::FNETWORK));
  }
  return uci::uci();
}
