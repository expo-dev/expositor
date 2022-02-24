use crate::context::*;
use crate::misc::*;
use crate::movegen::*;
use crate::movesel::*;
use crate::movetype::*;
use crate::piece::*;
use crate::score::*;
use crate::state::*;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub fn resolving_search(
  state      : &mut State,
  length     : u8,
  height     : u8,
  alpha      : i16,
  beta       : i16,
  context    : &mut Context,
  statistics : &mut Statistics,
) -> i16
{
  statistics.r_nodes_at_length[length as usize] += 1;
  statistics.r_nodes_at_height[height as usize] += 1;

  // You can't stand pat if you're in check
  let static_eval = if state.incheck { i16::MIN + 1 } else { state.evaluate_in_game() };
  let mut estimate = static_eval;

  if estimate >= beta { return estimate; }

  let mut alpha = std::cmp::max(estimate, alpha);

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

  let mut selector = MoveSelector::new(selectivity, height, NULL_MOVE);
  let metadata = state.save();
  let mut successors = 0;
  while let Some(mv) = selector.next(state, context) {
    successors += 1;

    // We always allow a move to be considered if it gives discovered
    //   check, even if static exchange analysis predicts it is losing
    if !state.incheck && !mv.is_unusual() && !mv.gives_discovered_check() {
      if mv.is_capture() {
        // Delta pruning
        //   We disable delta pruning when the score is near mate because the
        //   assumptions of delta pruning no longer hold. In a KR v K endgame,
        //   for instance, capturing the rook doesn't gain roughly 5 pawns –
        //   it gains 100 pawns or more (converting a loss into a draw).
        // TODO rather than using the static exchange score, use the most
        //   optimistic score (capturing without retaliation) which results
        //   in safer, more conservative behavior.
        if !mv.gives_check() && estimate.abs() < LIKELY_MATE {
          if static_eval + (mv.score as i16 * 20) + 100 < alpha { continue; }
        }
        // Ignore losing captures and, as we move away from the leaf,
        //   start ignoring merely neutral captures as well
        let threshold = if length < 2 || mv.gives_check() { 0 } else { 1 };
        if mv.score < threshold { continue; }
      }
      else {
        let prediction = state.analyze_exchange(
          Op {square: mv.src, piece: mv.piece},
          Op {square: mv.dst, piece: Piece::NullPiece}
        );
        if prediction < 0 { continue; }
      }
    }

    context.gainful[length as usize] = mv.is_gainful() || state.incheck;
    state.apply(&mv);
    let score =
      -resolving_search(state, length+1, height+1, -beta, -alpha, context, statistics);
    state.undo(&mv);
    state.restore(&metadata);

    estimate = std::cmp::max(estimate, score);
    alpha    = std::cmp::max(alpha,    score);
    if alpha >= beta {
      if !mv.is_gainful() {
        let killers = &mut context.killer_table[height as usize];
        if !quick_eq(&mv, &killers.0) {
          killers.1 = (killers.0).clone();
          killers.0 = mv;
        }
      }
      break;
    }
  }

  // When there are no active/gainful successors and we're in check, this is actually mate!
  //   since every move counts as active/gainful when you are in check
  if successors == 0 && state.incheck { return PROVEN_LOSS + height as i16; }

  return estimate;
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
// This is identical to resolving_search but prints the search tree as it goes

fn indent(tab : u8) { for _ in 0..tab { eprint!("\x1B[2m│\x1B[22m  "); } }
fn last_indent(tab : u8) { if tab > 0 { indent(tab-1); eprint!("\x1B[2m└\x1B[22m  "); } }

pub fn debug_resolving_search(
  state      : &mut State,
  length     : u8,
  height     : u8,
  alpha      : i16,
  beta       : i16,
  context    : &mut Context,
) -> i16
{
  let static_eval =
    if state.incheck { i16::MIN + 1 } else { (state.evaluate() * 100.0).round() as i16 };
  let mut estimate = static_eval;

  if estimate >= beta {
    last_indent(height); eprintln!("cutoff est {} >= beta {}", estimate, beta);
    return estimate;
  }

  let mut alpha = std::cmp::max(estimate, alpha);

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

  let mut selector = MoveSelector::new(selectivity, height, NULL_MOVE);
  let metadata = state.save();
  let mut successors = 0;
  while let Some(mv) = selector.next(state, context) {
    successors += 1;

    if !state.incheck && !mv.is_unusual() && !mv.gives_discovered_check() {
      if mv.is_capture() {
        if !mv.gives_check() && estimate.abs() < LIKELY_MATE {
          if static_eval + (mv.score as i16 * 20) + 100 < alpha {
            indent(height); eprintln!("{} delta \x1B[2m{} < {}\x1B[22m", mv, static_eval + (mv.score as i16 * 10) + 150, alpha);
            continue;
          }
        }
        let threshold = if length < 2 || mv.gives_check() { 0 } else { 1 };
        if mv.score < threshold {
          indent(height); eprintln!("{} skipped \x1B[2m{} < {}\x1B[22m", mv, mv.score, threshold);
          continue;
        }
      }
      else {
        let prediction = state.analyze_exchange(
          Op {square: mv.src, piece: mv.piece},
          Op {square: mv.dst, piece: Piece::NullPiece}
        );
        if prediction < 0 {
          indent(height); eprintln!("{} skipped \x1B[2m{} < 0\x1B[22m", mv, prediction);
          continue;
        }
      }
    }

    indent(height); eprintln!("{} \x1B[2m{}\x1B[22m", mv, mv.score);

    context.gainful[length as usize] = mv.is_gainful() || state.incheck;
    state.apply(&mv);
    let score =
      -debug_resolving_search(state, length+1, height+1, -beta, -alpha, context);
    state.undo(&mv);
    state.restore(&metadata);

    estimate = std::cmp::max(estimate, score);
    alpha    = std::cmp::max(alpha,    score);
    if alpha >= beta {
      indent(height); eprintln!("cutoff alpha {} >= beta {}", alpha, beta);
      if !mv.is_gainful() {
        let killers = &mut context.killer_table[height as usize];
        if !quick_eq(&mv, &killers.0) {
          killers.1 = (killers.0).clone();
          killers.0 = mv;
        }
      }
      break;
    }
  }

  if successors == 0 && state.incheck {
    last_indent(height); eprintln!("mate {}", height);
    return PROVEN_LOSS + height as i16;
  }

  last_indent(height); eprintln!("= {}", estimate);
  return estimate;
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
// This is identical to resolving_search but prints the FEN of the leaves

pub fn resolving_search_leaves(
  state      : &mut State,
  length     : u8,
  height     : u8,
  alpha      : i16,
  beta       : i16,
  context    : &mut Context,
) -> i16
{
  let mut leaf = !state.incheck;

  let static_eval =
    if state.incheck { i16::MIN + 1 } else { (state.evaluate() * 100.0).round() as i16 };
  let mut estimate = static_eval;

  if estimate >= beta { return estimate; }

  let mut alpha = std::cmp::max(estimate, alpha);

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

  let mut selector = MoveSelector::new(selectivity, height, NULL_MOVE);
  let metadata = state.save();
  let mut successors = 0;
  while let Some(mv) = selector.next(state, context) {
    successors += 1;

    if !state.incheck && !mv.is_unusual() && !mv.gives_discovered_check() {
      if mv.is_capture() {
        if !mv.gives_check() && estimate.abs() < LIKELY_MATE {
          if static_eval + (mv.score as i16 * 20) + 100 < alpha { continue; }
        }
        let threshold = if length < 2 || mv.gives_check() { 0 } else { 1 };
        if mv.score < threshold { continue; }
      }
      else {
        let prediction = state.analyze_exchange(
          Op {square: mv.src, piece: mv.piece},
          Op {square: mv.dst, piece: Piece::NullPiece}
        );
        if prediction < 0 { continue; }
      }
    }

    context.gainful[length as usize] = mv.is_gainful() || state.incheck;
    state.apply(&mv);
    let score =
      -resolving_search_leaves(state, length+1, height+1, -beta, -alpha, context);
    state.undo(&mv);
    state.restore(&metadata);

    if score > estimate { leaf = false; }

    estimate = std::cmp::max(estimate, score);
    alpha    = std::cmp::max(alpha,    score);
    if alpha >= beta {
      if !mv.is_gainful() {
        let killers = &mut context.killer_table[height as usize];
        if !quick_eq(&mv, &killers.0) {
          killers.1 = (killers.0).clone();
          killers.0 = mv;
        }
      }
      break;
    }
  }

  if successors == 0 && state.incheck { return PROVEN_LOSS + height as i16; }

  if estimate < alpha { leaf = false; }
  if leaf { println!("{}", state.to_fen()); }
  return estimate;
}
