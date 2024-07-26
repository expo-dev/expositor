use crate::algebraic::Algebraic;
use crate::cache::*;
use crate::color::Color::*;
use crate::context::Context;
use crate::constants::*;
use crate::formats::*;
use crate::global::*;
use crate::misc::NodeKind;
use crate::limits::Limits;
use crate::movegen::Selectivity::Everything;
use crate::movesel::{Stage, MoveSelector};
use crate::movetype::{Move, fast_eq};
use crate::piece::Kind::*;
use crate::resolve::resolving_search;
use crate::score::*;
use crate::state::State;
use crate::syzygy::{syzygy_support, probe_syzygy_wdl};
use crate::tablebase::probe_3man;
use crate::util::{STDOUT, STDERR, isatty, get_terminal_width};

use std::time::{Instant, Duration};

pub const MINIMUM_DEPTH : u8 = 4;

pub const MAX_DEPTH  : usize = 192;
pub const MAX_HEIGHT : usize = 256;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
// A word about minimax with α/β pruning
//
// With perfect move ordering, search nodes have this structure:
//
//          P-V           Cut           All
//      ┌────┼────┐        │        ┌────┼────┐
//     P-V  Cut  ...      All      Cut  ...  ...
//
//   where moves are searched left-to-right and an ellipsis indicates zero or more nodes of the
//   previous type.
//
// With imperfect move ordering, they have this structure:
//
//                       P-V                              Cut                All
//        ┌────────┬──────┼────┬────┬────┐          ┌──────┼────┐        ┌────┼────┐
//     P-V/Cut  P-V/Cut  ...  P-V  Cut  ...      P-V/Cut  ...  All      Cut  ...  ...
//                             ↑
//                         best move
//
// A tree with perfect move ordering then looks like:
//
//                                             P-V
//                         ┌───────────┬────────┼────────┐
//                        P-V         Cut      Cut      Cut
//             ┌──────┬────┼────┐      │        │        │
//            P-V    Cut  Cut  Cut    All      All      All
//      ┌───┬──┼──┐   │    │    │   ┌──┼──┐  ┌──┼──┐  ┌──┼──┐
//     P-V  C  C  C  All  All  All  C  C  C  C  C  C  C  C  C
//     ┌┼┐  │  │  │  ┌┼┐  ┌┼┐  ┌┼┐  │  │  │  │  │  │  │  │  │
//
// Actual search trees are quite good – at the time of writing, 96% of cut nodes have only a
//   single child and in 65% of P-V nodes the first child is the best move. This is partially
//   explained by iterative deepening with a transposition table: the tree is continually being
//   reëvaluated, and at each iteration the previously calculated best responses are now ordered
//   first. And since deeper searches are more likely to be accurate, the closer a node is to
//   the root, the more likely it is for its first move to be ordered correctly. So the interior
//   of the tree looks nearly ideal, but the leaves might be garbage (over the next few steps
//   they will be massively refined, but no longer be leaf nodes). What's curious, though, is
//   that the overall statistics are still good even given the fact that more than half of all
//   nodes are leaf nodes.
//
// If you believe you are an All node, you should predict your children are Cut nodes.
//
// If you believe you are a Cut node, you should predict your children are All nodes.
//
// If you believe you are a P-V node, you should predict your first child is a P-V node and
//   every subsequent child is a Cut node. There are conditions under which you might revise
//   your beliefs:
//
//   • If you strongly believe your move ordering is perfect and the first child fails to raise
//     alpha (meaning it was a Cut node), then you should consider yourself an All node.
//
//   • If you suspect your move ordering is fallible (or you strongly believe you are in fact
//     a P-V node) and the first child fails to raise alpha, then you should predict the next
//     child is a P-V node (and not a Cut node) and that every child thereafter is a cut node
//     (and so on until a child raises alpha).

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub fn threefold(history : &Vec<(u64, bool)>) -> bool
{
  let current = match history.last() { Some(a) => a.0, None => return false };
  let end = history.len() - 1;
  let mut seen = false;
  for &(position, zeroing) in history[0..end].iter().rev() {
    if position == current { if seen { return true; } else { seen = true; } }
    if zeroing { return false; }
  }
  return false;
}

pub fn twofold(history : &Vec<(u64, bool)>) -> bool
{
  let current = match history.last() { Some(a) => a.0, None => return false };
  let end = history.len() - 1;
  for &(position, zeroing) in history[0..end].iter().rev() {
    if position == current { return true; }
    if zeroing { return false; }
  }
  return false;
}

pub fn main_search(
  state    : &mut State,
  depth    : u8,
  height   : u8,
  alpha    : i16,
  beta     : i16,
  zerowind : bool,
  // = zero window (when true, this node is not expected to be part of the principal variation)
  expected : NodeKind,
  context  : &mut Context,
) -> i16
{
  table_prefetch(state.key);

  context.pv[height as usize].clear();
  if abort() && context.nominal > MINIMUM_DEPTH { return INVALID_SCORE; }

  // Step 0. 50-move rule and draw-by-repetition detection
  // The condition for two-fold repetition needs to be
  //   at least "height > 1" to prevent false draw scores.
  if state.dfz > 100 { return 0; }
  if height > 3 { if twofold(&context.state_history) { return 0; } }
  else        { if threefold(&context.state_history) { return 0; } }

  // Step 1. Resolving search
  if depth == 0 { return resolving_search(state, 0, height, alpha, beta, context); }

  // Step 2. Update statistics
  context.m_nodes_at_height[height as usize] += 1;
  let mut alpha = alpha;
  let mut beta  = beta;

  // Step 3. Mate-distance pruning and tablebase access
  if height > 0 {
    let worst_possible = PROVEN_LOSS + (height as i16);
    let  best_possible = PROVEN_MATE - (height as i16 + 1);
    alpha = std::cmp::max(alpha, worst_possible);
    beta  = std::cmp::min( beta,  best_possible);
    if alpha >= beta { return alpha; }
  }

  let excluded_move = context.exclude[height as usize].clone();

  if height > 0
  && state.rights == 0
  && state.enpass < 0
  && excluded_move.is_null()
  && state.dfz == 0
  {
    let men = (state.sides[White] | state.sides[Black]).count_ones();
    if men == 3 { return probe_3man(state, height as i16); }
    if alpha < MINIMAL_PROVEN_MATE
    && beta  > MINIMAL_PROVEN_LOSS
    && syzygy_enabled()
    && syzygy_support() >= men {
      if let Some(score) = probe_syzygy_wdl(state, height as i16) {
        context.tb_hits += 1;
        use std::cmp::Ordering::*;
        match score.cmp(&0) {
          Equal   => { return score; }
          Greater => { alpha = std::cmp::max(alpha, score); }
          Less    => { beta  = std::cmp::min( beta, score); }
          // TODO is it okay (is it correct) to lower beta even
          //   when it doesn't cause a cutoff? (it's certainly
          //   okay to return early in the Less case if the TB
          //   score is less than or equal to alpha)
        }
        if alpha >= beta { return score; }
      }
    }
  }

  // Step 4. Transposition table lookup and internal iterative reduction
  let mut depth = depth;

  let mut hint_move = Move::NULL;
  let mut hint_score = INVALID_SCORE;

  let entry =
    if excluded_move.is_null() {
      // // We mix dfz into the key when dfz is close to 100 to prevent false
      // //   positives and negatives in draw detection due to transpositions.
      // table_lookup(if state.dfz > 92 { state.key ^ state.dfz as u64 } else { state.key })
      table_lookup(state.key)
    }
    else {
      TableEntry::NULL
    };

  let okay_generation = prev_gen_enabled() || (entry.generation == generation() as u16);
  if entry.hint_move != 0 && okay_generation {
    hint_move = Move::decompress(state, entry.hint_move);
    #[cfg(debug_assertions)] {
      if hint_move.is_null() {
        let p   = ((entry.hint_move >> 15) & 0x0001) as i8;
        let k   = ((entry.hint_move >> 12) & 0x0007) as i8;
        let dst = ((entry.hint_move >>  6) & 0x003F) as i8;
        let src = ( entry.hint_move        & 0x003F) as i8;
        eprintln!("key collision or decode error: unable to decompress");
        eprintln!("  {}", state.to_fen());
        eprintln!(
          "  generation = {}, depth = {}, kind = {}",
          entry.generation, entry.depth, entry.kind.as_str()
        );
        eprintln!(
          "  hint move = {:01b} {:03b} {:06b} {:06b}, src = {}, dst = {}",
          p, k, dst, src, src.algebraic(), dst.algebraic()
        );
        eprintln!("  hint score = {:+} = {}", entry.hint_score, format_score(entry.hint_score));
      }
    }
    if !hint_move.is_null() {
      let score =
        if      entry.hint_score >= MINIMAL_TB_MATE { entry.hint_score - height as i16 }
        else if MINIMAL_TB_LOSS >= entry.hint_score { entry.hint_score + height as i16 }
        else                                        { entry.hint_score                 };
      if zerowind && entry.depth >= depth {
        if (entry.kind as u8) & (NodeKind::Cut as u8) != 0 && score >= beta  { return score; }
        if (entry.kind as u8) & (NodeKind::All as u8) != 0 && alpha >= score { return score; }
      }
      hint_score = score;
    }
  }
  else if height >= 2 && depth >= 6 && excluded_move.is_null() {
    depth -= 1;
  }

  // Step 5. Snapshot saving
  let metadata = state.save();

  // Step 6. Reductions and pruning
  let mut lower_estimate = INVALID_SCORE;
  let mut futile = false;

  if context.nominal > 4 && height > 0 && zerowind && !state.incheck {

    let pieces = state.boards[state.turn+Queen ]
               | state.boards[state.turn+Rook  ]
               | state.boards[state.turn+Bishop]
               | state.boards[state.turn+Knight];
    let num_pieces = pieces.count_ones();

    let static_eval =
      if !hint_move.is_null() && entry.kind == NodeKind::PV { 0 /* this will be ignored */ }
    //else if depth > 4 { resolving_search(state, 0, height, alpha, beta, context) }
      else { state.game_eval() };

    let ignore_hint = hint_move.is_null() || entry.kind == NodeKind::All;
    let upper_estimate = if ignore_hint { static_eval } else { hint_score };

    let ignore_hint = hint_move.is_null() || entry.kind == NodeKind::Cut;
    lower_estimate = if ignore_hint { static_eval } else { hint_score };

    // Step 6a. Reverse futility pruning
    if depth < 8
      && num_pieces >= 1
      && !context.null[height as usize - 1]
      && upper_estimate >= beta + RFP_OFFSET + (depth as i16)*RFP_SCALE
    { return beta; }

    // Step 6b. Null-move reduction and pruning
    if depth > 4
      && num_pieces >= 1
      && !context.null[height as usize - 1]
      && upper_estimate >= beta
    {
      let reduction = NULL_BASE + (depth as u32)*NULL_SCALE;
      let null_depth = ((depth as u32)*4096 - reduction) / 4096;
      state.apply_null();
      context.null[height as usize] = true;
      let subscore = main_search(
        state, null_depth as u8, height+1, -beta, -beta+1, true,
        NodeKind::All, context
      );
      context.null[height as usize] = false;
      state.undo_null();
      state.restore(&metadata);
      if subscore == INVALID_SCORE { return INVALID_SCORE; }
      let null_score = -subscore;
      if null_score >= beta {
        // At high depths, add verification search? (regular search at low depth)
        //   Verification search is required to also equal or exceed beta to
        //   actually perform the pruning.
        // TODO perhaps come up with a more sophisticated
        //   estimate of when zugzwang isn't a concern.
        if num_pieces >= 2 { return null_score; }
        depth = std::cmp::min(depth - 4, depth / 2);
      }
    }
  }

  // Step 7. Extensions and multicut

  //   Step 7a. Check extension
  //   TODO should this or the original depth be recorded in the cache update?
  if state.incheck { depth += 1; }

  //   Step 7b. Singular extension and multicut
  let mut extend_hint = false;
  if height > 0
    && depth >= 9
    && expected != NodeKind::All
    && !state.incheck
    && !hint_move.is_null()
    && entry.depth >= depth - 2
    && (entry.kind as u8) & (NodeKind::Cut as u8) != 0
    && hint_score.abs() < MINIMAL_TB_MATE
  {
    let reduction = SSE_BASE + (depth as i32)*SSE_SCALE;
    let sse_depth = ((depth as u32)*4096 - (reduction as u32)) / 4096;

    let margin = (SSE_MARGIN_OFFSET + (depth as i32)*SSE_MARGIN_SCALE) / 256;
    let margin = margin as i16;

    context.exclude[height as usize] = hint_move.clone();
    let subscore = main_search(
      state, sse_depth as u8, height, hint_score-margin-1, hint_score-margin,
      true, expected, context
    );
    context.exclude[height as usize] = Move::NULL;
    if subscore == INVALID_SCORE { return INVALID_SCORE; }
    let singular_score = subscore;
    if singular_score >= hint_score - margin {
      if hint_score - margin >= beta { return hint_score - margin; }
    }
    else {
      extend_hint = true;
    }
  }

  // Step 8. Move iteration
  let mut selector = MoveSelector::new(
    Everything,
    height,
    if excluded_move.is_null() { hint_move.clone() } else { excluded_move.clone() }
  );

  let mut best_score = LOWEST_SCORE;
  let mut best_move  = Move::NULL;

  let mut successors   = 0;
  let mut raised_alpha = false;
  let mut last_raise   = 0;

  let mut next_expected = match expected {
    NodeKind::Unk => NodeKind::Unk,
    NodeKind::All => NodeKind::Cut,
    NodeKind::Cut => NodeKind::All,
    NodeKind::PV  => NodeKind::PV,
  };

  let mut emit_hint = !hint_move.is_null();
  loop {
    // Step 8a. Move application
    let mv;
    if emit_hint {
      emit_hint = false;
      state.apply(&hint_move);
      if state.in_check(!state.turn) {
        #[cfg(debug_assertions)] {
          eprintln!("key collision or decode error: illegal state");
          eprintln!("  {}", state.to_fen());
          eprintln!(
            "  hint move = {}, src = {}, dst = {}",
            hint_move, hint_move.src.algebraic(), hint_move.dst.algebraic()
          );
          eprintln!("  hint score = {:+} = {}", hint_score, format_score(hint_score));
        }
        state.undo(&hint_move);
        state.restore(&metadata);
        hint_move = Move::NULL;
        continue;
      }
      state.incheck = state.in_check(state.turn);
      // NOTE that givescheck is not set properly here (although gives_check may be
      //   true, gives_direct_check and gives_discovered_check will always be false)
      hint_move.givescheck = if state.incheck { 1u8 << 2 } else { 0 };
      mv = hint_move.clone();
    }
    else {
      mv = match selector.next(state, context) { Some(x) => x, None => break };

      // Step 8b. Futility pruning (set up in step 6)
      if depth < 8
        && alpha > INEVITABLE_LOSS
        && lower_estimate != INVALID_SCORE
        && lower_estimate + FP_OFFSET + (depth as i16)*FP_SCALE < alpha
      { futile = true; }

      if successors > 1
        && futile
        && !mv.is_gainful()
        && !mv.gives_check()
        && mv.score < FP_THRESH
      { continue; }

      state.apply(&mv);
    }
    successors += 1;

    if expected == NodeKind::PV && raised_alpha { next_expected = NodeKind::Cut; }

    context.state_history.push((state.key, mv.is_zeroing()));

    let mut score = INVALID_SCORE;
    let okay = loop {
      if successors > 1 {
        // Step 8c. Reduced-depth zero-window search
        if depth > 2 {
          let mut reduction : i16 = 0;
          if height > 0 && !mv.is_gainful() && successors > 3 {
            let moves_since_raise = (successors - last_raise) as f64;
            let mut r = LMR_BASE
                      + LMR_SCALE * ((depth-1) as f64).log2() * moves_since_raise.log2()
                      - HST_SCALE * (mv.score as f64);
            if !zerowind { r *= FW_RATIO; }
            reduction = r.floor() as i16;
          }
          if reduction > 0 {
            let next_depth = std::cmp::max(0, depth as i16 - 1 - reduction) as u8;
            let subscore = main_search(
              state, next_depth, height+1, -alpha-1, -alpha,
              true, next_expected, context
            );
            if subscore == INVALID_SCORE { break false; }
            score = -subscore;
            if !(score > alpha) { break true; }
          }
        }
        // Step 8d. Full-depth zero-window search (within a full-window search)
        if !zerowind {
          let next_depth = depth - 1;
          let subscore = main_search(
            state, next_depth, height+1, -alpha-1, -alpha,
            true, next_expected, context
          );
          if subscore == INVALID_SCORE { break false; }
          score = -subscore;
          if !(score > alpha) { break true; }
        }
      }

      // Step 8e. Full-depth full-window search (within a full-window search) or
      //            zero-window search (within a zero-window search)
      let next_depth = if successors == 1 && extend_hint { depth } else { depth - 1 };
      let subscore = main_search(
        state, next_depth, height+1, -beta, -alpha,
        zerowind, next_expected, context
      );
      if subscore == INVALID_SCORE { break false; }
      score = -subscore;
      break true;
    };

    // Step 8f. State restoration
    context.state_history.pop();
    state.undo(&mv);
    state.restore(&metadata);

    if !okay { return INVALID_SCORE; }

    debug_assert!(score != INVALID_SCORE, "score was not set");

    // Step 8g. Folding
    if score > best_score {
      best_score = score;
      best_move = mv.clone();
      context.pv.swap(height as usize, height as usize + 1);
    }
    if score > alpha {
      raised_alpha = true;
      alpha = score;
      last_raise = successors;
    }

    // Step 8h. Beta cutoff
    if alpha >= beta {
      if !mv.is_gainful() {
        let killers = &mut context.killer_table[height as usize];
        if !fast_eq(&mv, &killers.0) {
          killers.1 = (killers.0).clone();
          killers.0 = mv;
        }
      }
      if depth >= 2 { selector.update_history_scores(context, depth); }
      break;
    }
  }

  // Step 9. Mate detection
  if successors == 0 {
    best_score =
      if !excluded_move.is_null() { alpha }
      else if state.incheck { PROVEN_LOSS + height as i16 }
      else { 0 };
  }
  else if state.dfz == 100 {
    best_score = 0;
  }

  // Step 10. Node classification
  let kind =
    if alpha > beta { NodeKind::Cut }
    // It's possible for all of your children to be cut-nodes (which consider them-
    //   selves lower bounds) to return scores (that are to you upper bounds) that
    //   are all equal to alpha – and so no child raised alpha above its initial
    //   value. This means you are not a PV node, but an All node! And this is
    //   why we can't simply write "best_score < alpha".
    else if !raised_alpha { NodeKind::All }
    // When alpha equals beta, we can only declare this an exact score when we
    //   searched every node!
    // There's an additional edge case, introduced by mate-distance pruning. When we
    //   artificially manipulate alpha and beta, we lose information and are unable to
    //   distinguish Cut nodes from PV nodes when alpha == beta.
    else if alpha == beta &&
      (selector.stage != Stage::Done || best_score.abs() >= MINIMAL_PROVEN_MATE) { NodeKind::Cut }
    // The last case – this is actually a PV node.
    else { NodeKind::PV };

  // Step 11. Transposition table update
  if excluded_move.is_null() {
    let table_score =
      if      best_score >= MINIMAL_TB_MATE { best_score + height as i16 }
      else if best_score <= MINIMAL_TB_LOSS { best_score - height as i16 }
      else                                  { best_score                 };

    table_update(&TableEntry {
    //key:        if state.dfz > 92 { state.key ^ state.dfz as u64 } else { state.key },
      key:        state.key,
      generation: generation() as u16,
      hint_move:  best_move.compress(),
      hint_score: table_score,
      depth:      depth,
      kind:       kind,
    });
  }

  // Step 12. Returning
  if kind == NodeKind::PV { context.pv[height as usize].insert(0, best_move); }
  return best_score;
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn support(
  thread_id : usize,
  mut state : State,
  history   : Vec<(u64, bool)>,
)
{
  let context = unsafe { &mut GLOB.context[thread_id] };
  context.reset();
  context.state_history = history;

  let mut score = resolving_search(&mut state, 0, 0, LOWEST_SCORE, HIGHEST_SCORE, context);

  'iterative_deepening: for step in 1..((MAX_DEPTH+1) as u8) {
    let jitter = ((thread_id % 4) + 1) as f64 * 62.5;
    let relax = 250.0 + jitter;
    let tight = (25.0 + (score.abs() as f64)*0.25).min(100.0);
    let init_width = (tight + (relax-tight)*((step-1) as f64 * -0.25).exp2()) as i16;

    let mut alpha_width = init_width;
    let mut  beta_width = init_width;
    let mut alpha_failures = 0;
    let mut  beta_failures = 0;
    let mut searching = true;
    let mut tentative = score;
    while searching {
      let alpha = if alpha_failures >= 2 {  LOWEST_SCORE } else { score - alpha_width };
      let  beta = if  beta_failures >= 2 { HIGHEST_SCORE } else { score +  beta_width };
      context.nominal = step;
      tentative = main_search(
        &mut state, step, 0, alpha, beta, false, NodeKind::PV, context
      );
      if abort() { break 'iterative_deepening; }
      debug_assert!(tentative != INVALID_SCORE);
      searching = tentative >= beta || alpha >= tentative;
      if alpha >= tentative { alpha_failures += 1; alpha_width *= 2; }
      if tentative >= beta  {  beta_failures += 1;  beta_width *= 2; }
    }
    score = tentative;
  }
}

fn best_move(
  supervisor : std::thread::Thread,
  mut state  : State,
  history    : Vec<(u64, bool)>,
  limits     : Limits,
)
{
  // ↓↓↓ DEBUG ↓↓↓
  // unsafe { crate::apply::KING_MOVES  = 0; }
  // unsafe { crate::apply::RESET_COUNT = 0; }
  // ↑↑↑ DEBUG ↑↑↑

  let term_width = if isatty(STDERR) { get_terminal_width() } else { 0 };

  // Setup

  let context = unsafe { &mut GLOB.context[0] };
  context.reset();
  context.state_history = history;

  let mut prev_time_to_depth = 0.0;
  let mut ratios = [1.5, 1.5, 1.5, 1.5];
  let mut last_best = Move::NULL;
  let mut stability = 0;

  let mut last_step = 0;
  let mut last_pv;
  let mut last_score;
  let mut score =
    resolving_search(&mut state, 0, 0, LOWEST_SCORE, HIGHEST_SCORE, context);
  let mut best = Move::NULL;
  let mut no_change = false;

  // Iterative deepening loop

  let clock = Instant::now();
  'iterative_deepening: for step in 1..(limits.depth+1) {

    // Part 1. Search at the next depth

    last_pv = context.pv[0].clone();

    let relax = 250.0;
    let tight = (25.0 + (score.abs() as f64)*0.25).min(100.0);
    let init_width = (tight + (relax-tight)*((step-1) as f64 * -0.25).exp2()) as i16;

    let mut alpha_width = init_width;
    let mut  beta_width = init_width;
    let mut alpha_failures = 0;
    let mut  beta_failures = 0;
    let mut searching = true;
    let mut tentative = score;
    while searching {
      let alpha = if alpha_failures >= 2 {  LOWEST_SCORE } else { score - alpha_width };
      let  beta = if  beta_failures >= 2 { HIGHEST_SCORE } else { score +  beta_width };
      context.nominal = step;
      tentative = main_search(
        &mut state, step, 0, alpha, beta, false, NodeKind::PV, context
      );
      if abort() && step > MINIMUM_DEPTH { break 'iterative_deepening; }
      debug_assert!(tentative != INVALID_SCORE);
      searching = tentative >= beta || alpha >= tentative;
      if alpha >= tentative { alpha_failures += 1; alpha_width *= 2; }
      if tentative >= beta  {  beta_failures += 1;  beta_width *= 2; }
    }
    last_score = score;
    score = tentative;
    if !context.pv[0].is_empty() { best = context.pv[0][0].clone(); }
    last_step = step;
    let time_to_depth = clock.elapsed().as_secs_f64();

    stability = if fast_eq(&best, &last_best) { stability + 1 } else { 1 };
    last_best = best.clone();

    // Part 2. Report to stderr and stdout

    let mut ext_depth = 0;
    for x in 0..MAX_HEIGHT { if context.r_nodes_at_height[x] != 0 { ext_depth = x; } }

    let mut nodes  = 0;
    let mut tbhits = 0;
    unsafe {
      let num_threads = num_threads();
      for id in 0..num_threads {
        // These values might be slightly out of date, but we don't mind
        nodes  += GLOB.context[id].m_nodes_at_height.iter().sum::<usize>();
        nodes  += GLOB.context[id].r_nodes_at_height.iter().sum::<usize>();
        tbhits += GLOB.context[id].tb_hits;
      }
    }

    let nps = nodes as f64 / time_to_depth;

    if isatty(STDERR) {
      eprint!("\r");  // TODO replace with ANSI escape code
      if no_change { eprint!("\x1B[A\x1B[K"); }
      no_change = score == last_score && context.pv[0] == last_pv;
      let rectified = if state.turn == Black { -score } else { score };
      // 3 + 1 + 3 + 1 + 6 + 1 + 5 = 20
      eprint!(
        "{:3} \x1B[2m{}/{}\x1B[22m {} \x1B[2m{}\x1B[22m",
        step, alpha_failures, beta_failures,
        format_time(time_to_depth), format_node_compact(nodes)
      );
      let mut line_width = 20;
      // 1 + 5 = 6
      if syzygy_enabled() {
        eprint!(" \x1B[2m{}\x1B[22m", format_node_compact(tbhits));
        line_width += 6;
      }
      // 1 + 6 = 7
      eprint!(" \x1B[1m{:>6}\x1B[22m", format_score(rectified));
      line_width += 7;

      let mut scratch = state.clone_empty();
      for mv in context.pv[0].iter() {
        let short = mv.in_context(&scratch);
        let mv_width = 1 + short.len() as u16;
        if term_width > 0 && line_width + mv_width > term_width { break; }
        eprint!(" {}", short);
        line_width += mv_width;
        scratch.apply(mv);
      }
      eprint!("\n");
    }
    if !isatty(STDOUT) || !isatty(STDERR) {
      print!(
        "info depth {} seldepth {} nodes {} tbhits {} time {:.0} nps {:.0} score {}",
        step, ext_depth+1, nodes, tbhits, time_to_depth * 1000.0, nps, format_uci_score(score)
      );
      if !context.pv[0].is_empty() {
        print!(" multipv 1 pv");
        for mv in context.pv[0].iter() { print!(" {}", mv.algebraic()); }
      }
      print!("\n");
    }

    // Part 3. Check time management

    if step >= MINIMUM_DEPTH {

      // We don't break above when step = MINIMUM_DEPTH so that we can write info,
      //   but with that out of the way, we need to break now

      if abort() { break; }

      // Decide whether to stop early if we're given a target (soft limit)

      if let Some(target) = limits.target {

        // Update the history of time to depth ratios
        if prev_time_to_depth > 0.0 {
           ratios = [time_to_depth / prev_time_to_depth, ratios[0], ratios[1], ratios[2]];
        }
        prev_time_to_depth = time_to_depth;
        // Remove the outlier
        //   Unfortunately, since f64 doesn't implement Ordering, we can't use max_by_key:
        //   let (idx, val) =
        //     ratios.iter().enumerate().max_by_key(|(_, x)| (*x - arith_mean).abs()).unwrap();
        let arith_mean = ratios.iter().sum::<f64>() * 0.25;
        let mut idx = 0;
        let mut dev = (ratios[0] - arith_mean).abs();
        for x in 1..4 {
          let d = (ratios[x] - arith_mean).abs();
          if d >= dev { idx = x; dev = d; }
        }
        let outlier = ratios[idx];
        ratios[idx] = 1.0;
        // Take the geometric mean
        let geom_mean = ratios.iter().product::<f64>().cbrt();
        ratios[idx] = outlier;
        // Estimate the next time to depth
        let estimate = time_to_depth * geom_mean;

        if let Some(hard_limit) = limits.cutoff {
          if estimate > hard_limit && stability > 1 { break; }
        }

        if estimate - target > target - time_to_depth { break; }
      }
    }
  }

  // Report final summary and bestmove

  let elapsed = clock.elapsed().as_secs_f64();

  let mut main_depth = 0;
  let mut  ext_depth = 0;
  for x in 0..MAX_HEIGHT { if context.m_nodes_at_height[x] != 0 { main_depth = x; } }
  for x in 0..MAX_HEIGHT { if context.r_nodes_at_height[x] != 0 {  ext_depth = x; } }

  let mut m_nodes = 0;
  let mut r_nodes = 0;
  let mut leaves  = 0;
  unsafe {
    let num_threads = num_threads();
    for id in 0..num_threads {
      // These values might be slightly out of date, but we don't mind
      m_nodes += GLOB.context[id].m_nodes_at_height.iter().sum::<usize>();
      r_nodes += GLOB.context[id].r_nodes_at_height.iter().sum::<usize>();
      leaves  += GLOB.context[id].r_nodes_at_length[0];
    }
  }

  let nodes = m_nodes + r_nodes;
  let nps = nodes as f64 / elapsed;

  if isatty(STDERR) {
    eprintln!("  depth {}/{}/{}", last_step, main_depth+1, ext_depth+1);
    eprintln!(
      "   \x1B[2m{}node main {:2.0}%\x1B[22m",
      format_node(m_nodes), m_nodes as f64 * 100.0 / nodes as f64
    );
    eprintln!(
      "   \x1B[2m{}node leaf {:2.0}%\x1B[22m",
      format_node(leaves), leaves as f64 * 100.0 / nodes as f64
    );
    eprintln!(
      "   \x1B[2m{}node extn {:2.0}%\x1B[22m",
      format_node(r_nodes - leaves), (r_nodes - leaves) as f64 * 100.0 / nodes as f64
    );
    eprintln!("   {}node", format_node(nodes));
    eprintln!("   {}node/s", format_node(nps as usize));
    let num_threads = num_threads();
    if num_threads > 1 {
      eprintln!("   {}nps/th", format_node((nps / num_threads as f64) as usize));
    }
    eprintln!(
      " {} {}",
      format_time(elapsed), if elapsed < 60.0 { "seconds" } else { "elapsed" }
    );
  }
  if !isatty(STDOUT) || !isatty(STDERR) {
    println!("bestmove {}", best.algebraic());
  }

  // ↓↓↓ DEBUG ↓↓↓
  // unsafe { eprintln!("{:5.2}%", crate::apply::KING_MOVES  as f64 * 100.0 / nodes as f64); }
  // unsafe { eprintln!("{:5.2}%", crate::apply::RESET_COUNT as f64 * 100.0 / nodes as f64); }
  // ↑↑↑ DEBUG ↑↑↑

  // Stop the search if it hasn't been stopped already

  if !abort() { set_abort(true); supervisor.unpark(); }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn supervise(
  state       : State,
  history     : Vec<(u64, bool)>,
  limits      : Limits,
  num_threads : usize,
)
{
  set_num_threads(num_threads);

  let time_limit = limits.cutoff;
  let clock = Instant::now();

  unsafe {
    if GLOB.context.len() < num_threads {
      let deficit = num_threads - GLOB.context.len();
      for _ in 0..deficit { GLOB.context.push(Context::new()); }
    }
  }
  set_abort(false);

  let mut handles = Vec::new();
  {
    let supervisor = std::thread::current();
    let state = state.clone();
    let history = history.clone();
    handles.push(
      std::thread::Builder::new()
        .name(String::from("search.0"))
        .spawn(move || { best_move(supervisor, state, history, limits); })
        .unwrap()
    );
  }
  if num_threads > 1 {
    for id in 1..num_threads {
      let state = state.clone();
      let history = history.clone();
      handles.push(
        std::thread::Builder::new()
          .name(format!("search.{}", id))
          .spawn(move || { support(id, state, history); })
          .unwrap()
      );
    }
  }

  if let Some(stop_time) = time_limit {
    loop {
      let time_to_go = stop_time - clock.elapsed().as_secs_f64();
      if !(time_to_go > 0.0) {
        set_abort(true);
        break;
      }
      std::thread::park_timeout(Duration::from_secs_f64(time_to_go));
      if abort() { break; }
    }
  }

  for h in handles { h.join().expect("unable to join search thread"); }

  set_searching(false);
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// These two remaining methods are called only by the main thread (managing the UCI interface)
//   and so can never be running concurrently.

pub fn start_search(
  state       : &State,
  history     : &Vec<(u64, bool)>,
  limits      : Limits,
  num_threads : usize,
) -> std::thread::JoinHandle<()>
{
  let num_threads = std::cmp::max(num_threads, 1);
  increment_generation();
  set_searching(true);
  let state = state.clone();
  let history = history.clone();
  return std::thread::Builder::new()
    .name(String::from("supervisor"))
    .spawn(move || { supervise(state, history, limits, num_threads); })
    .unwrap();
}

pub fn stop_search(handle : &std::thread::JoinHandle<()>)
{
  set_abort(true);
  handle.thread().unpark();
}
