use crate::algebraic::Algebraic;
use crate::color::Color::*;
use crate::context::Context;
use crate::formats::*;
use crate::limits::Limits;
use crate::misc::vmirror;
use crate::movegen::Selectivity::Everything;
use crate::movetype::Move;
use crate::policy::{PolicyNetwork, PolicyBuffer};
use crate::resolve::resolving_search;
use crate::score::*;
use crate::search::{threefold, twofold};
use crate::state::State;
use crate::util::{isatty, STDOUT, STDERR};

use std::time::{Instant, Duration};

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum SimplexConfig {
  ValueOnly,  // value network only w/ resolving search
  Hybrid,     // value network w/ r.s. for captures and policy network for quiet moves
  PolicyOnly  // policy network only
}

pub fn simplexitor(
  state   : &State,
  history : &Vec<(u64, bool)>,
  limits  : Limits,
  config  : SimplexConfig,
  network : &PolicyNetwork
)
{
  use SimplexConfig::*;
  let clock = Instant::now();

  let mut buf = PolicyBuffer::zero();
  if config != ValueOnly { network.initialize(&state, &mut buf); }

  let mut state = state.clone();
  let mut context = Context::new();
  context.state_history = history.clone();

  let mut best_score = PROVEN_LOSS;
  let mut best_move = Move::NULL;

  let mut best_loss = f32::INFINITY;
  let mut poly_move = Move::NULL;

  let metadata = state.save();
  let (early_moves, late_moves) = state.legal_moves(Everything);
  let num_successors = early_moves.len() + late_moves.len();
  context.m_nodes_at_height[0] = num_successors;

  if state.dfz > 100 || threefold(&context.state_history) {
    best_score = 0;
  }
  else if num_successors == 0 {
    best_score = if state.incheck { PROVEN_LOSS } else { 0 };
  }
  else if state.dfz == 100 {
    best_score = 0;
  }
  else {
    for mv in early_moves.into_iter().chain(late_moves.into_iter()) {
      if config == PolicyOnly || (config == Hybrid && !mv.is_gainful()) {
        let src = mv.src as usize;
        let src = match state.turn { White => src, Black => vmirror(src) };
        let dst = mv.dst as usize;
        let dst = match state.turn { White => dst, Black => vmirror(dst) };
        let loss = network.evaluate(&buf, mv.piece.kind(), src, dst);

        if loss < best_loss {
          best_loss = loss;
          poly_move = mv.clone();
        }
      }

      state.apply(&mv);
      context.state_history.push((state.key, mv.is_zeroing()));
      let score =
        if twofold(&context.state_history) {
          0
        }
        else {
          let (em, lm) = state.legal_moves(Everything);
          if em.is_empty() && lm.is_empty() {
            if state.incheck { PROVEN_MATE } else { 0 }
          }
          else {
            -resolving_search(&mut state, 0, 1, LOWEST_SCORE, HIGHEST_SCORE, &mut context)
          }
        };
      context.state_history.pop();
      state.undo(&mv);
      state.restore(&metadata);

      if score > best_score {
        best_score = score;
        best_move = mv.clone();
      }
    }
  }

  if !best_move.is_null() {
    if config == PolicyOnly || (config == Hybrid && !best_move.is_gainful()) {
      best_move = poly_move;
    }
  }

  let time_active = clock.elapsed().as_secs_f64();

  let mut ext_depth = 0;
  for x in 0..128 { if context.r_nodes_at_height[x] != 0 { ext_depth = x; } }
  let m_nodes = context.m_nodes_at_height[0];
  let r_nodes = context.r_nodes_at_height.iter().sum::<usize>();
  let nodes = m_nodes + r_nodes;

  if isatty(STDERR) {
    let rectified = if state.turn == Black { -best_score } else { best_score };
    eprintln!("\x1B[1m{}\x1B[22m {}", format_score(rectified), best_move);
  }
  if !isatty(STDOUT) || !isatty(STDERR) {
    println!(
      "info depth 1 seldepth {} nodes {} time {:.0} score {} multipv 1 pv {}",
      ext_depth+1, nodes, time_active * 1000.0, format_uci_score(best_score),
      best_move.algebraic()
    );
  }

  let wait = limits.target.unwrap_or(limits.cutoff.unwrap_or(0.0));
  let remaining = wait - clock.elapsed().as_secs_f64();
  if remaining > 0.0 { std::thread::park_timeout(Duration::from_secs_f64(remaining)); }
  let elapsed = clock.elapsed().as_secs_f64();

  if isatty(STDERR) {
    eprintln!("depth 1/{}", ext_depth+1);
    eprintln!("{}node", variable_format_node(nodes));
    eprintln!(
      "{} {}",
      variable_format_time(elapsed),
      if elapsed < 60.0 { "seconds" } else { "elapsed" }
    );
  }
  if !isatty(STDOUT) || !isatty(STDERR) {
    println!("bestmove {}", best_move.algebraic());
  }
}
