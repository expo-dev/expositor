// use crate::color::Color::*;
use crate::context::Context;
use crate::formats::*;
use crate::limits::Limits;
// use crate::misc::vmirror;
use crate::movegen::Selectivity::Everything;
use crate::movetype::Move;
// use crate::policy::{PolicyBuffer, POLICY};
use crate::rand::Rand;
use crate::score::PovScore;
use crate::search::{threefold, twofold};
use crate::state::State;
use crate::util::{isatty, STDOUT, STDERR, get_random};

use std::time::{Instant, Duration};

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum SimplexConfig {
  ValueOnly,  // value network only w/ resolving search
  Hybrid,     // value network w/ r.s. for captures and policy network for quiet moves
  PolicyOnly  // policy network only
}

const HORIZON : u16 = 20;

const RANDOMIZE : bool = true;

pub fn simplexitor(
  state   : &State,
  history : &Vec<(u64, bool)>,
  limits  : Limits,
//config  : SimplexConfig,
) -> bool
{
  // use SimplexConfig::*;
  let clock = Instant::now();

  let mut rng = if RANDOMIZE { get_random() } else { 0 };

  // let mut buf = PolicyBuffer::zero();
  // if config != ValueOnly { unsafe { POLICY.initialize(&state, &mut buf); } }

  let mut state = state.clone();
  let mut context = Context::new();
  context.state_history = history.clone();

  let mut snd_score  = PovScore::LOST;
  let mut best_score = PovScore::LOST;
  let mut best_move = Move::NULL;

  // let mut best_loss = f32::INFINITY;
  // let mut poly_move = Move::NULL;

  let metadata = state.save();
  let legal_moves = state.collect_legal_moves(Everything);
  let num_successors = legal_moves.len();
  context.m_nodes_at_height[0] = num_successors;

  if state.dfz > 100 || threefold(&context.state_history) {
    best_score = PovScore::ZERO;
  }
  else if num_successors == 0 {
    best_score = if state.incheck { PovScore::LOST } else { PovScore::ZERO };
  }
  else if state.dfz == 100 {
    best_score = PovScore::ZERO;
  }
  else {
    for mv in legal_moves.into_iter() {
      /*
      if config == PolicyOnly || (config == Hybrid && !mv.is_gainful()) {
        let src = mv.src as usize;
        let src = match state.turn { White => src, Black => vmirror(src) };
        let dst = mv.dst as usize;
        let dst = match state.turn { White => dst, Black => vmirror(dst) };
        let loss = unsafe { POLICY.evaluate(&buf, mv.piece.kind(), src, dst) };

        if loss < best_loss {
          best_loss = loss;
          poly_move = mv.clone();
        }
      }
      */

      state.apply(&mv);
      context.state_history.push((state.key, mv.is_zeroing()));
      let score =
        if twofold(&context.state_history) {
          PovScore::ZERO
        }
        else {
          let lm = state.collect_legal_moves(Everything);
          if lm.is_empty() {
            if state.incheck { PovScore::realized_win(1) } else { PovScore::ZERO }
          }
          else {
            let raw = -resolving_search(
              &mut state, 0, 1, PovScore::MIN, PovScore::MAX, &mut context
            );
            if RANDOMIZE && !raw.is_winloss() {
              raw + PovScore::new((u16::rand_with(&mut rng) % 32) as i16 - 15)
            } else { raw }
          }
        };
      context.state_history.pop();
      state.undo(&mv);
      state.restore(&metadata);

      if score > best_score {
        snd_score = best_score;
        best_score = score;
        best_move = mv.clone();
      }
      else if score > snd_score {
        snd_score = score;
      }
    }
  }

/*
  if !best_move.is_null() {
    if config == PolicyOnly || (config == Hybrid && !best_move.is_gainful()) {
      best_move = poly_move;
    }
  }
*/

  // If we're decidedly winning, let Expositor finish the game so that we make
  //   forward progress and don't frustrate players.
  if best_score >= PovScore::new(10_00) { return false; }

  let time_active = clock.elapsed().as_secs_f64();

  let mut ext_depth = 0;
  for x in 0..128 { if context.r_nodes_at_height[x] != 0 { ext_depth = x; } }
  let m_nodes = context.m_nodes_at_height[0];
  let r_nodes = context.r_nodes_at_height.iter().sum::<usize>();
  let nodes = m_nodes + r_nodes;

  if isatty(STDERR) {
    eprintln!("\x1B[1m{}\x1B[22m {}", best_score.from(state.turn), best_move);
  }
  if !isatty(STDOUT) || !isatty(STDERR) {
    println!(
      "info depth 1 seldepth {} nodes {} time {:.0} score {:#o} multipv 1 pv {:#o}",
      ext_depth+1, nodes, time_active * 1000.0, best_score, best_move
    );
  }

  let jitter = 2.0_f64.powf(u32::rand() as f64 / 2147483647.5 - 1.4150375);
  let shuffle = 1.0 - std::cmp::min(state.dfz, 20) as f64 * (0.75 / 20.0);
  let gap = std::cmp::min(best_score.as_i16() - snd_score.as_i16(), 9_00) as f64 / 100.0;
  let multiplier = jitter * shuffle / (1.0 + gap);

  let wait = limits.target.unwrap_or(limits.cutoff.unwrap_or(0.0));
  let wait = (wait * multiplier).min(limits.cutoff.unwrap_or(0.0));
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
    println!("bestmove {:#o}", best_move);
  }

  return true;
}

fn game_eval(state : &State) -> PovScore
{
  if state.dfz >= HORIZON { return PovScore::ZERO; }
  if let Some(score) = state.endgame() { return score; }
  return PovScore::new((state.evaluate() * 100.0).round() as i16);
}

fn resolving_search(
  state      : &mut State,
  length     : u8,
  height     : u8,
  alpha      : PovScore,
  beta       : PovScore,
  context    : &mut Context,
) -> PovScore
{
  use crate::misc::Op;
  use crate::movegen::Selectivity;
  use crate::movesel::MoveSelector;
  use crate::movetype::fast_eq;
  use crate::piece::Piece::Null;

  if state.dfz > 100 { return PovScore::ZERO; }

  context.r_nodes_at_length[length as usize] += 1;
  context.r_nodes_at_height[height as usize] += 1;

  let worst_possible = PovScore::realized_loss(height);
  let  best_possible = PovScore::realized_win(height + 1);

  // You can't stand pat if you're in check
  let static_eval = if state.incheck { worst_possible } else { game_eval(state) };
  let mut estimate = static_eval;

  let mut alpha = std::cmp::max(estimate, alpha);
  let beta = std::cmp::min(best_possible, beta);

  if alpha >= beta { return alpha; }

  // Don't consider moves which only give check (which are neither
  //   captures nor promotions) if last turn you didn't capture or
  //   promote either
  // Once we're far enough from the leaf, we disallow these kinds
  //   of moves altogether
  // (If you're in check, the selector will automatically set the
  //   selectivity to Everything)
  let selectivity =
    if length >= 4 {
      Selectivity::GainfulOnly
    }
    else if length >= 2 && !context.gainful[length as usize - 2] {
      Selectivity::GainfulOnly
    }
    else {
      Selectivity::ActiveOnly
    };

  let mut selector = MoveSelector::new(selectivity, height, Move::NULL);
  let metadata = state.save();
  let mut successors = 0;
  while let Some(mv) = selector.next(state, context) {
    successors += 1;

    // We always allow a move to be considered if it gives discovered
    //   check, even if static exchange analysis predicts it is losing
    if !state.incheck && !mv.is_unusual() && !mv.gives_discovered_check() {
      if mv.is_capture() {
        // Ignore losing captures and, as we move away from the leaf,
        //   start ignoring merely neutral captures as well
        let threshold = if length < 2 || mv.gives_check() { 0 } else { 1 };
        if mv.score < threshold { continue; }
      }
      else {
        let prediction = state.analyze_exchange(
          Op {square: mv.src, piece: mv.piece},
          Op {square: mv.dst, piece: Null}
        );
        if prediction < 0 { continue; }
      }
    }

    context.gainful[length as usize] = mv.is_gainful() || state.incheck;
    state.apply(&mv);
    let score =
      -resolving_search(state, length+1, height+1, -beta, -alpha, context);
    state.undo(&mv);
    state.restore(&metadata);

    estimate = std::cmp::max(estimate, score);
    alpha    = std::cmp::max(alpha,    score);
    if alpha >= beta {
      if !mv.is_gainful() {
        let killers = &mut context.killer_table[height as usize];
        if !fast_eq(&mv, &killers.0) {
          killers.1 = (killers.0).clone();
          killers.0 = mv;
        }
      }
      break;
    }
  }

  // When there are no active/gainful successors and we're in check, this is actually mate!
  //   since every move counts as active/gainful when you are in check.
  if successors == 0 && state.incheck { return PovScore::realized_loss(height); }
  else if state.dfz == 100 { return PovScore::ZERO; }
  else { return estimate; }
}
