use crate::color::Color::*;
use crate::context::Context;
use crate::misc::NodeKind;
use crate::movegen::Selectivity::Everything;
use crate::movetype::Move;
use crate::global::*;
use crate::resolve::{resolving_search, resolving_search_leaves};
use crate::score::{PROVEN_LOSS, Outcome};
use crate::state::{State, MiniState};
use crate::search::{MINIMUM_DEPTH, main_search, threefold};

use std::collections::HashSet;
use std::io::{BufWriter, Write};
use std::fs::File;
use std::time::Instant;

// TODO switch everything to game_eval (including resolving_search_leaves)
//   so that the HCE can be used

pub fn selfplaytest(path : &str, initial : &State) -> std::io::Result<()>
{
  let mut w = BufWriter::new(File::create(path)?);
  return selfplay(&mut w, initial, 0.25, DEFAULT_CONDITIONS, 15_00);
}

pub struct EndConditions {
  pub draw_onset : u16, // The game is drawn if this many halfmoves have been played and,
  pub draw_len   : u16, //   for this many consecutive ply,
  pub draw_score : u16, //   the absolute score of the position has been this or less.
  pub rsgn_len   : u16, // The game is won or lost if, for this many consecutive ply,
  pub rsgn_score : u16, //   the absolute score is greater than this.
}

const DEFAULT_CONDITIONS : EndConditions =
  EndConditions {
    draw_onset: 80,
    draw_len:   20,
    draw_score: 10,
    rsgn_len:    6,
    rsgn_score: 15_00
  };

const QUIET_THRESHOLD : i16 = 10;

pub fn selfplay(
  out         : &mut BufWriter<File>,
  initial     : &State,
  search_time : f64,
  conditions  : EndConditions,
  max_score   : u16,
) -> std::io::Result<()>
{
  // NOTE make sure that Hash, Persist, and SyzygyPath are set to the
  //   values you want, since those values will be used during search.

  set_searching(true);
  set_abort(false);

  let mut emitted   = HashSet::<u64>::new();
  let mut positions = Vec::<MiniState>::new();
  let mut leaves    = Vec::<(State, i16)>::new();
  let mut adjacent  = Vec::<State>::new();

  let mut state = initial.clone();
  let mut context = Context::new();
  state.initialize_nnue();

  let mut halfmoves_played : u16 = 0;
  let mut draw_run : u16 = 0;
  let mut rsgn_run : u16 = 0;
  let outcome;
  loop {
    // We first obtain an estimate to determine the initial aspiration window
    //   and, while we're at it, gather some leaves.
    let static_eval = (state.evaluate() * 100.0).round() as i16;
    let q_eval = resolving_search_leaves(
      &mut state, 0, 0, i16::MIN+1, i16::MAX, &mut context, &mut leaves
    );
    if q_eval == PROVEN_LOSS {
      outcome = match state.turn {
        White => Outcome::Black,
        Black => Outcome::White
      };
      break;
    }
    let (score, mv) = search_score(&mut state, &mut context, q_eval, search_time);
    let score = match state.turn { White => score, Black => -score };
    halfmoves_played += 1;

    // If the score is within range and this is is a quiet position, add it to the list of
    //   game positions that we'll emit (that are both scored and marked with an outcome).
    if score.abs() as u16 <= max_score
    && (static_eval - q_eval).abs() <= QUIET_THRESHOLD {
      emitted.insert(state.key);
      let mut mini = MiniState::from(&state);
      mini.score = score;
      positions.push(mini);
    }

    // We're also interested in adjacent positions that are obviously bad, since most moves
    //   in a search tree are bad, and we want to provide a somewhat representative sample.
    let metadata = state.save();
    let (early_moves, late_moves) = state.legal_moves(Everything);
    for nonsense in early_moves.into_iter().chain(late_moves.into_iter()) {
      state.apply(&nonsense);
      adjacent.push(state.clone_truncated());
      state.undo(&nonsense);
      state.restore(&metadata);
    }

    // eprintln!(
    //   "{:3} {:7} {:+5} {:+5} {:+5}",
    //   halfmoves_played, &mv, score, q_eval, static_eval
    // );

    draw_run = if score.abs() as u16 > conditions.draw_score { 0 } else { draw_run + 1 };
    rsgn_run = if score.abs() as u16 > conditions.rsgn_score { rsgn_run + 1 } else { 0 };

    if draw_run >= conditions.draw_len && halfmoves_played >= conditions.draw_onset {
      outcome = Outcome::Draw;
      break;
    }
    if rsgn_run >= conditions.rsgn_len {
      outcome = if score < 0 { Outcome::Black } else { Outcome::White };
      break;
    }

    state.apply(&mv);
    context.state_history.push((state.key, mv.is_zeroing()));
    if state.dfz > 100 || threefold(&context.state_history) {
      outcome = Outcome::Draw;
      break;
    }
    increment_generation();
  }

  for mut mini in positions {
    mini.set_outcome(outcome);
    out.write(mini.to_quick().as_bytes())?;
    out.write(b"\n")?;
    // eprintln!("{} = {:+} {}", State::from(&mini).to_fen(), mini.score, outcome);
  }

  for (mut leaf, q_eval) in leaves {
    if !emitted.insert(leaf.key) { continue; }
    context.state_history.clear();
    let (score, _) = search_score(&mut leaf, &mut context, q_eval, search_time);
    if score.abs() as u16 > max_score { continue; }
    let mut mini = MiniState::from(&leaf);
    mini.score = match leaf.turn { White => score, Black => -score };
    out.write(mini.to_quick().as_bytes())?;
    out.write(b"\n")?;
    // eprintln!("{} = {:+}", State::from(&mini).to_fen(), mini.score);
  }

  let mut index = 0;
  for mut nonsense in adjacent {
    if !emitted.insert(nonsense.key) { continue; }
    // We have to check for quietness, unlike for the leaves above.
    let static_eval = nonsense.game_eval();
    let q_eval = resolving_search(&mut nonsense, 0, 0, i16::MIN+1, i16::MAX, &mut context);
    if (static_eval - q_eval).abs() > QUIET_THRESHOLD { continue; }
    // There are more of these than we want, so we don't emit them all.
    index += 1; if index & 15 != 0 { continue; }
    // The rest from here is identical.
    context.state_history.clear();
    let (score, _) = search_score(&mut nonsense, &mut context, q_eval, search_time);
    if score.abs() as u16 > max_score { continue; }
    let mut mini = MiniState::from(&nonsense);
    mini.score = match nonsense.turn { White => score, Black => -score };
    out.write(mini.to_quick().as_bytes())?;
    out.write(b"\n")?;
    // eprintln!("{} = {:+}", State::from(&mini).to_fen(), mini.score);
  }

  out.flush()?;

  set_abort(true);
  set_searching(false);
  return Ok(());
}

pub fn search_score(
  state   : &mut State,
  context : &mut Context,
  guess   : i16,
  target  : f64,
) -> (i16, Move)
{
  context.partial_reset();

  let mut prev_time_to_depth = 0.0;
  let mut ratios = [1.5, 1.5, 1.5, 1.5];

  let mut score = guess;
  let mut best = Move::NULL;

  let clock = Instant::now();
  for step in 1..65 {

    let relax = 250.0;
    let tight = (25.0 + (score.abs() as f64)*0.25).min(100.0);
    let init_width = (tight + (relax-tight)*((step-1) as f64 * -0.25).exp2()) as i16;

    let mut alpha_width = init_width;
    let mut beta_width  = init_width;
    let mut alpha_failures = 0;
    let mut  beta_failures = 0;
    let mut searching = true;
    let mut tentative = score;
    while searching {
      let alpha = if alpha_failures >= 2 { i16::MIN+1 } else { score - alpha_width };
      let  beta = if  beta_failures >= 2 { i16::MAX   } else { score +  beta_width };
      context.nominal = step;
      tentative = main_search(state, step, 0, alpha, beta, false, NodeKind::PV, context);
      searching = tentative >= beta || alpha >= tentative;
      if alpha >= tentative { alpha_failures += 1; alpha_width *= 2; }
      if tentative >= beta  {  beta_failures += 1;  beta_width *= 2; }
    }
    score = tentative;
    if !context.pv[0].is_empty() { best = context.pv[0][0].clone(); }
    let time_to_depth = clock.elapsed().as_secs_f64();

    if step >= MINIMUM_DEPTH {
      if prev_time_to_depth > 0.0 {
        ratios = [time_to_depth / prev_time_to_depth, ratios[0], ratios[1], ratios[2]];
      }
      prev_time_to_depth = time_to_depth;
      let arith_mean = ratios.iter().sum::<f64>() * 0.25;
      let mut idx = 0;
      let mut dev = (ratios[0] - arith_mean).abs();
      for x in 1..4 {
        let d = (ratios[x] - arith_mean).abs();
        if d >= dev { idx = x; dev = d; }
      }
      let outlier = ratios[idx];
      ratios[idx] = 1.0;
      let geom_mean = ratios.iter().product::<f64>().cbrt();
      ratios[idx] = outlier;
      let estimate = time_to_depth * geom_mean;

      if estimate - target > target - time_to_depth { break; }
    }
  }
  return (score, best);
}
