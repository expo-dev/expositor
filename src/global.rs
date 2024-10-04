use crate::cache::HugePage;
use crate::context::Context;
use crate::limits::Limits;
use crate::state::State;

use std::sync::atomic::{AtomicUsize, AtomicBool};
use std::sync::atomic::Ordering::{Acquire, Release};

// TODO only allow use_nnue to be set/unset when search not in progress
//   and be sure to reinitialize the network

#[repr(align(64))]
pub struct Configuration {
  pub num_blocks   : AtomicUsize, //  8   8
  pub num_threads  : AtomicUsize, //  8  16
  pub use_prev_gen : AtomicBool,  //  1  17
  pub use_nnue     : AtomicBool,  //  1  18
  pub use_tb       : AtomicBool,  //  1  19
}

#[repr(align(64))]
pub struct Globals {
  pub cache      : Vec<HugePage>,   //  24  24
  pub context    : Vec<Context>,    //  24  48
  pub generation : AtomicUsize,     //   8  56
  pub num_search : AtomicUsize,     //   8  64
  pub searching  : AtomicBool,      //   1  65
  pub abort      : AtomicBool,      //   1  66
  pub kill       : AtomicBool,      //   1  67
  pub supervisor : Option<std::thread::Thread>,
  pub state      : State,
  pub history    : Vec<(u64, bool)>,
  pub limits     : Limits,
  pub clock      : Option<std::time::Instant>,
}

pub static mut CONF : Configuration = Configuration {
  num_blocks:   AtomicUsize::new(0),
  num_threads:  AtomicUsize::new(0),
  use_prev_gen: AtomicBool::new(false),
  use_nnue:     AtomicBool::new(true),
  use_tb:       AtomicBool::new(false),
};

pub static mut GLOB : Globals = Globals {
  cache:      Vec::new(),
  context:    Vec::new(),
  generation: AtomicUsize::new(0),
  num_search: AtomicUsize::new(0),
  searching:  AtomicBool::new(false),
  abort:      AtomicBool::new(false),
  kill:       AtomicBool::new(false),
  supervisor: None,
  state:      State::new(),
  history:    Vec::new(),
  limits:     Limits::new(),
  clock:      None,
};

#[inline] pub fn num_blocks      () -> usize { unsafe { return CONF.num_blocks  .load(Acquire); } }
#[inline] pub fn num_threads     () -> usize { unsafe { return CONF.num_threads .load(Acquire); } }
#[inline] pub fn prev_gen_enabled() -> bool  { unsafe { return CONF.use_prev_gen.load(Acquire); } }
#[inline] pub fn nnue_enabled    () -> bool  { unsafe { return CONF.use_nnue    .load(Acquire); } }
#[inline] pub fn syzygy_enabled  () -> bool  { unsafe { return CONF.use_tb      .load(Acquire); } }

#[inline] pub fn generation() -> usize { unsafe { return GLOB.generation.load(Acquire); } }
#[inline] pub fn num_search() -> usize { unsafe { return GLOB.num_search.load(Acquire); } }
#[inline] pub fn searching () -> bool  { unsafe { return GLOB.searching .load(Acquire); } }
#[inline] pub fn abort     () -> bool  { unsafe { return GLOB.abort     .load(Acquire); } }
#[inline] pub fn kill      () -> bool  { unsafe { return GLOB.kill      .load(Acquire); } }

#[inline] pub fn set_num_blocks (n : usize) { unsafe { CONF.num_blocks  .store(n, Release); } }
#[inline] pub fn set_num_threads(n : usize) { unsafe { CONF.num_threads .store(n, Release); } }
#[inline] pub fn enable_prev_gen(x : bool ) { unsafe { CONF.use_prev_gen.store(x, Release); } }
#[inline] pub fn enable_nnue    (x : bool ) { unsafe { CONF.use_nnue    .store(x, Release); } }
#[inline] pub fn enable_syzygy  (x : bool ) { unsafe { CONF.use_tb      .store(x, Release); } }

#[inline] pub fn set_generation(n : usize) { unsafe { GLOB.generation.store(n, Release); } }
#[inline] pub fn set_num_search(n : usize) { unsafe { GLOB.num_search.store(n, Release); } }
#[inline] pub fn set_searching (x : bool ) { unsafe { GLOB.searching .store(x, Release); } }
#[inline] pub fn set_abort     (x : bool ) { unsafe { GLOB.abort     .store(x, Release); } }
#[inline] pub fn set_kill      (x : bool ) { unsafe { GLOB.kill      .store(x, Release); } }

#[inline]
pub fn increment_generation()
{
  // The documentation notes that "using Acquire makes the store part of this operation Relaxed,
  //   and using Release makes the load part Relaxed." The generation is only ever set by the
  //   supervisor, so there is no write contention from other threads, so we can let the load
  //   be relaxed (in fact, this doesn't need to become a `lock inc` instruction or anything).
  unsafe { GLOB.generation.fetch_add(1, Release); }
}
