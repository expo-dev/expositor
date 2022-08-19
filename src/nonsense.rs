use crate::context::*;
use crate::movegen::*;
use crate::rand::*;
use crate::resolve::*;
use crate::state::*;

use std::collections::HashSet;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

const OPENING_LEN :    u8 = 8;
const RANDOMIZE   :  bool = true;
const SEED        :   u64 = 0;
const MAX_BRANCH  : usize = 8;

const RESTRICT  : Rest = Rest::ViableLeaves;
const MAX_SCORE :  i16 = 200;


// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
enum Rest {         // Restriction
  Unrestricted = 0, //   allow all lines
  ViableLeaves = 1, //   restrict to lines that end with a viable score
  ViableLines  = 2, //   restrict to lines that are viable throughout
}

pub fn run_nonsense_openings()
{
  set_rand(SEED);

  let mut state = State::new();
  let mut context = Context::new();
  let mut stats = Statistics::new();
  state.initialize_nnue();

  let mut visited = HashSet::new();
  nonsense_openings(&mut state, &mut context, &mut stats, OPENING_LEN, &mut visited);
}

fn nonsense_openings(
  state   : &mut State,
  context : &mut Context,
  stats   : &mut Statistics,
  depth   : u8,
  visited : &mut HashSet<u64>,
)
{
  if !visited.insert(state.key ^ (depth as u64)) { return; }
  if RESTRICT != Rest::Unrestricted {
    let score = resolving_search(state, 0, 0, -(MAX_SCORE+50), MAX_SCORE+50, context, stats);
    if RESTRICT == Rest::ViableLines || depth == 0 {
      if score.abs() > MAX_SCORE { return; }
    }
  }
  if depth == 0 {
    println!("{}", state.to_fen());
    return;
  }
  let metadata = state.save();
  let mut early_moves = Vec::with_capacity(16);
  let mut  late_moves = Vec::with_capacity(32);
  state.generate_legal_moves(Selectivity::Everything, &mut early_moves, &mut late_moves);
  let mut num_explored = 0;
  if RANDOMIZE {
    early_moves.append(&mut late_moves);
    let num_moves = early_moves.len();
    let num_swap = num_moves.min(MAX_BRANCH);
    for x in 0..num_swap {
      let k = rand() as usize % (num_moves - x);
      early_moves.swap(x, x+k);
    }
    for mv in early_moves {
      if num_explored >= MAX_BRANCH { break; }
      state.apply(&mv);
      nonsense_openings(state, context, stats, depth-1, visited);
      state.undo(&mv);
      state.restore(&metadata);
      num_explored += 1;
    }
  }
  else {
    for mv in early_moves {
      if num_explored >= MAX_BRANCH { break; }
      state.apply(&mv);
      nonsense_openings(state, context, stats, depth-1, visited);
      state.undo(&mv);
      state.restore(&metadata);
      num_explored += 1;
    }
    for mv in late_moves {
      if num_explored >= MAX_BRANCH { break; }
      state.apply(&mv);
      nonsense_openings(state, context, stats, depth-1, visited);
      state.undo(&mv);
      state.restore(&metadata);
      num_explored += 1;
    }
  }
}
