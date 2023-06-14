use crate::context::Context;
use crate::movegen::Selectivity::Everything;
use crate::rand::{set_rand, rand};
use crate::resolve::resolving_search;
use crate::state::State;

use std::collections::HashSet;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

const CHESS324 : bool = true;

const OPENING_LEN :    u8 = 8;
const MAX_BRANCH  : usize = 8;

const RANDOMIZE   :  bool = true;
const SEED        :   u64 = 0;

const RESTRICT  : Rest = Rest::ViableLeaves;
const MAX_SCORE :  i16 = 200; // must be less than or equal to 255

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
  state.initialize_nnue();

  let mut visited = HashSet::new();
  if CHESS324 {
    for n in 0..324 {
      state = State::new324(n);
      nonsense_openings(&mut state, &mut context, OPENING_LEN-2, &mut visited);
    }
  }
  else {
    nonsense_openings(&mut state, &mut context, OPENING_LEN, &mut visited);
  }
}

fn nonsense_openings(
  state   : &mut State,
  context : &mut Context,
  depth   : u8,
  visited : &mut HashSet<u64>,
)
{
  if !visited.insert(state.key ^ (depth as u64)) { return; }
  if (RESTRICT == Rest::ViableLeaves && depth == 0) || RESTRICT == Rest::ViableLines {
    let score = resolving_search(state, 0, 0, -(MAX_SCORE+50), MAX_SCORE+50, context);
    if score.abs() > MAX_SCORE { return; }
  }
  if depth == 0 {
    println!("{}", state.to_fen());
    return;
  }
  let metadata = state.save();
  let (mut early_moves, mut late_moves) = state.legal_moves(Everything);
  early_moves.append(&mut late_moves);
  let mut num_explored = 0;
  if RANDOMIZE {
    let num_moves = early_moves.len();
    let num_swap = num_moves.min(MAX_BRANCH);
    for x in 0..num_swap {
      let k = rand() as usize % (num_moves - x);
      early_moves.swap(x, x+k);
    }
  }
  for mv in early_moves {
    if num_explored >= MAX_BRANCH { break; }
    state.apply(&mv);
    nonsense_openings(state, context, depth-1, visited);
    state.undo(&mv);
    state.restore(&metadata);
    num_explored += 1;
  }
}

pub fn run_exploration(root : &State)
{
  let mut state = root.clone();
  let mut context = Context::new();
  let mut visited = HashSet::new();
  explore(&mut state, &mut context, OPENING_LEN, &mut visited);
}

fn explore(
  state   : &mut State,
  context : &mut Context,
  depth   : u8,
  visited : &mut HashSet<u64>,
)
{
  if !visited.insert(state.key ^ (depth as u64)) { return; }
  println!("{}", state.to_fen());
  if depth == 0 { return; }
  let metadata = state.save();
  let (mut early_moves, mut late_moves) = state.legal_moves(Everything);
  early_moves.append(&mut late_moves);
  let mut viable_moves = Vec::with_capacity(32);
  for mut mv in early_moves {
    state.apply(&mv);
    let score = -resolving_search(state, 0, 0, -(MAX_SCORE+50), MAX_SCORE+50, context);
    state.undo(&mv);
    state.restore(&metadata);
    if score.abs() <= MAX_SCORE {
      mv.score = (score >> 1) as i8;
      viable_moves.push(mv);
    }
  }
  let n = std::cmp::min(viable_moves.len(), MAX_BRANCH);
  for num_explored in 0..n {
    let mut sel_index = num_explored;
    let mut sel_score = viable_moves[num_explored].score;
    for x in (num_explored+1)..(viable_moves.len()) {
      if viable_moves[x].score > sel_score {
        sel_index = x;
        sel_score = viable_moves[x].score;
      }
    }
    if sel_index != num_explored {
      viable_moves.swap(num_explored, sel_index);
    }
    let mv = &viable_moves[num_explored];
    state.apply(&mv);
    explore(state, context, depth-1, visited);
    state.undo(&mv);
    state.restore(&metadata);
  }
}
