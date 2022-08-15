use crate::context::*;
use crate::movegen::*;
use crate::rand::*;
use crate::resolve::*;
use crate::state::*;

use std::collections::HashSet;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

const opening_len :    u8 = 8;
const randomize   :  bool = true;
const seed        :   u64 = 0;
const max_branch  : usize = 8;

const restrict  : Rest = Rest::ViableLeaves;
const max_score :  i16 = 200;


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
  set_rand(seed);

  let mut state = State::new();
  let mut context = Context::new();
  let mut stats = Statistics::new();
  state.initialize_nnue();

  let mut visited = HashSet::new();
  nonsense_openings(&mut state, &mut context, &mut stats, opening_len, &mut visited);
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
  if restrict != Rest::Unrestricted {
    let score = resolving_search(state, 0, 0, -(max_score+50), max_score+50, context, stats);
    if restrict == Rest::ViableLines || depth == 0 {
      if score.abs() > max_score { return; }
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
  if randomize {
    early_moves.append(&mut late_moves);
    let num_moves = early_moves.len();
    let num_swap = num_moves.min(max_branch);
    for x in 0..num_swap {
      let k = rand() as usize % (num_moves - x);
      early_moves.swap(x, x+k);
    }
    for mv in early_moves {
      if num_explored >= max_branch { break; }
      state.apply(&mv);
      nonsense_openings(state, context, stats, depth-1, visited);
      state.undo(&mv);
      state.restore(&metadata);
      num_explored += 1;
    }
  }
  else {
    for mv in early_moves {
      if num_explored >= max_branch { break; }
      state.apply(&mv);
      nonsense_openings(state, context, stats, depth-1, visited);
      state.undo(&mv);
      state.restore(&metadata);
      num_explored += 1;
    }
    for mv in late_moves {
      if num_explored >= max_branch { break; }
      state.apply(&mv);
      nonsense_openings(state, context, stats, depth-1, visited);
      state.undo(&mv);
      state.restore(&metadata);
      num_explored += 1;
    }
  }
}
