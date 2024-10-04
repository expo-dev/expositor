use crate::cache::table_lookup;
use crate::context::Context;
use crate::global::{prev_gen_enabled, generation};
use crate::misc::Op;
use crate::movegen::Selectivity;
use crate::movesel::MoveSelector;
use crate::movetype::{Move, fast_eq};
use crate::piece::Piece::Null;
use crate::score::PovScore;
use crate::state::State;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// TODO generize to remove code duplication (here and in Simplexitor) and then
//   monomorphize to the function used in main search
pub fn resolving_search(
  state      : &mut State,
  length     : u8,
  height     : u8,
  alpha      : PovScore,
  beta       : PovScore,
  context    : &mut Context,
) -> PovScore
{
  if state.dfz > 100 { return PovScore::ZERO; }

  context.r_nodes_at_length[length as usize] += 1;
  context.r_nodes_at_height[height as usize] += 1;

  let worst_possible = PovScore::realized_loss(height);
  let  best_possible = PovScore::realized_win(height + 1);

  // You can't stand pat if you're in check
  let static_eval = if state.incheck { worst_possible } else { state.game_eval() };
  let mut estimate = static_eval;

  let mut alpha = std::cmp::max(estimate, alpha);
  let beta = std::cmp::min(best_possible, beta);

  if alpha >= beta { return alpha; }

  // Transposition table lookup
  let entry = table_lookup(state.key);
  let okay_generation = prev_gen_enabled() || (entry.generation == generation() as u16);
  if entry.hint_move != 0 && okay_generation {
    let hint_move = Move::decompress(state, entry.hint_move);
    #[cfg(debug_assertions)] {
      if hint_move.is_null() {
        eprintln!("\x1B[2mkey collision or decode error: unable to decompress\x1B[22m");
      }
    }
    if !hint_move.is_null() {
      let score =
        if      entry.hint_score.is_proven_win()  { entry.hint_score - PovScore::new(height as i16) }
        else if entry.hint_score.is_proven_loss() { entry.hint_score + PovScore::new(height as i16) }
        else                                      { entry.hint_score };
      if entry.kind.pv_or_cut() && score >= beta  { return score; }
      if entry.kind.pv_or_all() && alpha >= score { return score; }
    }
  }

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

  // TODO when the side to move has only a king,
  //   or maybe only a king and pawns,
  //   generate Everything and check for stalemate
  //   or being forced to push a pawn

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
        // Delta pruning
        //   We disable delta pruning when the score is near mate because the
        //   assumptions of delta pruning no longer hold. In a KR v K endgame,
        //   for instance, capturing the rook doesn't gain roughly 5 pawns –
        //   it gains 100 pawns or more (converting a loss into a draw).
        // TODO rather than using the static exchange score, consider using the
        //   most optimistic score (capturing without retaliation) which results
        //   in safer, more conservative behavior.
        if !estimate.is_winloss() {
          let gain = (mv.score as i32 * 10 * context.tuning.delta_scale) / 4096
                   + context.tuning.delta_margin;
          if static_eval + PovScore::new(gain as i16) < alpha { continue; }
        }
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

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
// This is identical to resolving_search but prints the search tree as it goes

fn indent(tab : u8, last : bool) {
  if tab > 0 {
    for _ in 0..tab {
      eprint!("\x1B[2m│\x1B[22m  ");
    }
  }
  if last {
    eprint!("\x1B[2m└─\x1B[22m ");
  }
  else {
    eprint!("\x1B[2m├─\x1B[22m ");
  }
}

pub fn debug_resolving_search(
  state   : &mut State,
  length  : u8,
  height  : u8,
  alpha   : PovScore,
  beta    : PovScore,
  context : &mut Context,
) -> PovScore
{
  let mut leaf = !state.incheck;

  let worst_possible = PovScore::realized_loss(height);
  let  best_possible = PovScore::realized_win(height + 1);

  let static_eval = if state.incheck { worst_possible } else { state.game_eval() };
  let mut estimate = static_eval;

  indent(height, false);
  eprintln!("static {:#}", estimate);

  let mut alpha = std::cmp::max(estimate, alpha);
  let beta = std::cmp::min(best_possible, beta);

  if alpha >= beta {
    indent(height, false);
    eprintln!("cutoff alpha {:#} ≥ beta {:#}", alpha, beta);
    indent(height, true);
    eprintln!("return {:#}", alpha);
    return alpha;
  }

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

    if !state.incheck && !mv.is_unusual() && !mv.gives_discovered_check() {
      if mv.is_capture() {
        let threshold = if length < 2 || mv.gives_check() { 0 } else { 1 };
        if mv.score < threshold {
          indent(height, false);
          eprintln!(
            "{:#} skip \x1B[2mexch {} < {}\x1B[22m",
            mv.disambiguate(state), mv.score, threshold
          );
          continue;
        }
        if !mv.gives_check() && !estimate.is_winloss() {
          let gain = (mv.score as i32 * 10 * context.tuning.delta_scale) / 4096
                   + context.tuning.delta_margin;
          if static_eval + PovScore::new(gain as i16) < alpha {
            indent(height, false);
            eprintln!(
              "{:#} skip \x1B[2meval+delta {} < alpha {}\x1B[22m",
              mv.disambiguate(state),
              static_eval + PovScore::new(gain as i16),
              alpha
            );
            continue;
          }
        }
      }
      else {
        let prediction = state.analyze_exchange(
          Op {square: mv.src, piece: mv.piece},
          Op {square: mv.dst, piece: Null}
        );
        if prediction < 0 {
          indent(height, false);
          eprintln!(
            "{:#} skip \x1B[2m{} < 0\x1B[22m",
            mv.disambiguate(state), prediction
          );
          continue;
        }
      }
    }

    indent(height, false);
    eprintln!("{:#}", mv.disambiguate(state));

    context.gainful[length as usize] = mv.is_gainful() || state.incheck;
    state.apply(&mv);
    let score =
      -debug_resolving_search(state, length+1, height+1, -beta, -alpha, context);
    state.undo(&mv);
    state.restore(&metadata);

    if score > estimate { leaf = false; }

    estimate = std::cmp::max(estimate, score);
    alpha    = std::cmp::max(alpha,    score);
    if alpha >= beta {
      indent(height, false);
      eprintln!("cutoff alpha {:#} ≥ beta {:#}", alpha, beta);
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

  if successors == 0 && state.incheck {
    indent(height, true);
    let mate = PovScore::realized_loss(height);
    eprintln!("mate {:#}", mate);
    return mate;
  }

  if estimate < alpha { leaf = false; }

  indent(height, true);
  eprint!("return {:#}", estimate);
  if leaf { eprint!(" \x1B[2mleaf\x1B[22m"); }
  eprintln!();
  return estimate;
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
// This is identical to resolving_search but collects the leaves

pub fn resolving_search_leaves(
  state   : &mut State,
  length  : u8,
  height  : u8,
  alpha   : PovScore,
  beta    : PovScore,
  context : &mut Context,
  leaves  : &mut Vec<State>
) -> PovScore
{
  let mut leaf = !state.incheck;

  let worst_possible = PovScore::realized_loss(height);
  let  best_possible = PovScore::realized_win(height + 1);

  let static_eval =
    if state.incheck { worst_possible } else { state.game_eval() };
  let mut estimate = static_eval;

  let mut alpha = std::cmp::max(estimate, alpha);
  let beta = std::cmp::min(best_possible, beta);

  if alpha >= beta { return alpha; }

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

    if !state.incheck && !mv.is_unusual() && !mv.gives_discovered_check() {
      if mv.is_capture() {
        let threshold = if length < 2 || mv.gives_check() { 0 } else { 1 };
        if mv.score < threshold { continue; }
        if !mv.gives_check() && !estimate.is_winloss() {
          let gain = (mv.score as i32 * 10 * context.tuning.delta_scale) / 4096
                   + context.tuning.delta_margin;
          if static_eval + PovScore::new(gain as i16) < alpha { continue; }
        }
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
      -resolving_search_leaves(state, length+1, height+1, -beta, -alpha, context, leaves);
    state.undo(&mv);
    state.restore(&metadata);

    if score > estimate { leaf = false; }

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

  if successors == 0 && state.incheck { return PovScore::realized_loss(height); }
  if estimate < alpha { leaf = false; }
  if leaf { leaves.push(state.clone_truncated()); }
  return estimate;
}
