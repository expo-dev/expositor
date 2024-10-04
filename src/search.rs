use crate::cache::*;
use crate::color::Color::*;
// use crate::constants::SearchConstants;
use crate::context::Context;
use crate::formats::*;
use crate::global::*;
use crate::misc::NodeKind;
use crate::limits::Limits;
use crate::movegen::Selectivity::Everything;
use crate::movesel::{Stage, MoveSelector};
use crate::movetype::{Move, fast_eq};
use crate::piece::Kind::*;
use crate::resolve::resolving_search;
use crate::score::PovScore;
use crate::state::State;
use crate::syzygy::{syzygy_support, probe_syzygy_wdl};
use crate::tablebase::probe_3man;
use crate::term::get_window_width;
use crate::util::{STDOUT, STDERR, isatty};

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
//              P-V                              Cut                All
//        ┌──────┼────┬────┬────┐          ┌──────┼────┐        ┌────┼────┐
//     P-V/Cut  ...  P-V  Cut  ...      P-V/Cut  ...  All      Cut  ...  ...
//                    ↑
//                best move
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

const ONE : PovScore = PovScore::new(1);

pub fn threefold(history : &Vec<(u64, bool)>) -> bool
{
  let current = match history.last() { Some(a) => a.0, None => return false };
  let end = history.len() - 1;
  let mut seen = false;
  // TODO we can actually step by two
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
  // TODO we can actually step by two
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
  alpha    : PovScore,
  beta     : PovScore,
  zerowind : bool,
  // = zero window (when true, this node is not expected to be part of the principal variation)
  expected : NodeKind,
  context  : &mut Context,
) -> PovScore
{
  table_prefetch(state.key);

  context.pv[height as usize].clear();
  if abort() && context.nominal > MINIMUM_DEPTH { return PovScore::SENTINEL; }

  // Step 0. 50-move rule and draw-by-repetition detection
  // The condition for two-fold repetition needs to be
  //   at least "height > 1" to prevent false draw scores.
  if state.dfz > 100 { return PovScore::ZERO; }
  if height > 3 { if twofold(&context.state_history) { return PovScore::ZERO; } }
  else        { if threefold(&context.state_history) { return PovScore::ZERO; } }

  // Step 1. Resolving search
  if depth == 0 { return resolving_search(state, 0, height, alpha, beta, context); }

  // Step 2. Update statistics
  context.m_nodes_at_height[height as usize] += 1;
  let mut alpha = alpha;
  let mut beta  = beta;

  // Step 3. Mate-distance pruning and tablebase access
  if height > 0 {
    let worst_possible = PovScore::realized_loss(height);
    let  best_possible = PovScore::realized_win(height + 1);
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
    if men == 3 { return probe_3man(state, height); }
    if !alpha.is_realized_win()
    && !beta.is_realized_loss()
    && syzygy_enabled()
    && syzygy_support() >= men {
      if let Some(score) = probe_syzygy_wdl(state, height) {
        context.tb_hits += 1;
        use std::cmp::Ordering::*;
        match score.cmp(&PovScore::ZERO) {
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
  let mut hint_score = PovScore::SENTINEL;

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

  // TODO only read generation once then store it in thread context
  let okay_generation = prev_gen_enabled() || (entry.generation == generation() as u16);
  if entry.hint_move != 0 && okay_generation {
    hint_move = Move::decompress(state, entry.hint_move);
    #[cfg(debug_assertions)] {
      use crate::misc::Algebraic;
      if hint_move.is_null() {
        let p   = ((entry.hint_move >> 15) & 0x0001) as i8;
        let k   = ((entry.hint_move >> 12) & 0x0007) as i8;
        let dst = ((entry.hint_move >>  6) & 0x003F) as i8;
        let src = ( entry.hint_move        & 0x003F) as i8;
        eprint!("\x1B[2m");
        eprintln!("key collision or decode error: unable to decompress");
        eprintln!("  {}", state.to_fen());
        eprintln!(
          "  generation = {}, depth = {}, kind = {}",
          entry.generation, entry.depth, entry.kind.as_str()
        );
        eprintln!(
          "  hint move = {:01b} {:03b} {:06b} {:06b}, src = {}, dst = {}",
          p, k, dst, src, src.id(), dst.id()
        );
        eprintln!("  hint score = {}", entry.hint_score);
        eprint!("\x1B[22m");
      }
    }
    if !hint_move.is_null() {
      let score =
        if      entry.hint_score.is_proven_win()  { entry.hint_score - PovScore::new(height as i16) }
        else if entry.hint_score.is_proven_loss() { entry.hint_score + PovScore::new(height as i16) }
        else                                      { entry.hint_score };
      if zerowind && entry.depth >= depth {
        if entry.kind.pv_or_cut() && score >= beta  { return score; }
        if entry.kind.pv_or_all() && alpha >= score { return score; }
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
  let mut lower_estimate = PovScore::SENTINEL;
  let mut futile = false;

  if context.nominal > 4 && height > 0 && zerowind && !state.incheck {

    let pieces = state.boards[state.turn+Queen ]
               | state.boards[state.turn+Rook  ]
               | state.boards[state.turn+Bishop]
               | state.boards[state.turn+Knight];
    let num_pieces = pieces.count_ones();

    let static_eval =
      if !hint_move.is_null() && entry.kind == NodeKind::PV {
        PovScore::ZERO  // this will be ignored
      }
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
      && upper_estimate >= beta + PovScore::new(context.tuning.rfp_offset + (depth as i16) * context.tuning.rfp_scale)
    { return beta; }

    // Step 6b. Null-move reduction and pruning
    if depth >= 4
      && num_pieces >= 1
      && !context.null[height as usize - 1]
      && upper_estimate >= beta
    {
      let reduction = context.tuning.null_base + (depth as u32) * context.tuning.null_scale;
      let null_depth = ((depth as u32)*4096 - reduction) / 4096;
      state.apply_null();
      context.null[height as usize] = true;
      let subscore = main_search(
        state, null_depth as u8, height+1, -beta, -beta+ONE, true,
        NodeKind::All, context
      );
      context.null[height as usize] = false;
      state.undo_null();
      state.restore(&metadata);
      if subscore.is_sentinel() { return PovScore::SENTINEL; }
      let null_score = -subscore;
      if null_score >= beta {
        // TODO perhaps come up with a more sophisticated
        //   estimate of when zugzwang isn't a concern.
        if depth < 10 {
          if num_pieces >= 2 { return null_score; } else { depth = depth / 2; }
        }
        else {
          let verif_score = main_search(
            state, null_depth as u8, height, beta-ONE, beta, true,
            NodeKind::Cut, context
          );
          // Inexplicably, using ">" rather than ">="
          //   here makes a significant difference.
          if verif_score > beta { return verif_score; }
        }
      }
    }
  }
  // Step 7. Extensions and multicut

  //   Step 7a. Check extension
  //   TODO should this or the original depth be recorded in the cache update?
  //     Probably the original – the argument the function was provided.
  if state.incheck { depth += 1; }

  //   Step 7b. Singular extension and multicut
  let mut extend_hint = false;
  if height > 0
    && depth >= 9
    && expected != NodeKind::All
    && !state.incheck
    && !hint_move.is_null()
    && entry.depth >= depth - 2
    && entry.kind.pv_or_cut()
    && !hint_score.is_proven_winloss()
  {
    let reduction = context.tuning.sse_base + (depth as i32) * context.tuning.sse_scale;
    let sse_depth = ((depth as u32)*4096 - (reduction as u32)) / 4096;

    let margin = (context.tuning.sse_margin_offset + (depth as i32) * context.tuning.sse_margin_scale) / 256;
    let margin = PovScore::new(margin as i16);

    context.exclude[height as usize] = hint_move.clone();
    let subscore = main_search(
      state, sse_depth as u8, height, hint_score-margin-ONE, hint_score-margin,
      true, expected, context
    );
    context.exclude[height as usize] = Move::NULL;
    if subscore.is_sentinel() { return PovScore::SENTINEL; }
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

  let mut best_score = PovScore::MIN;
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
          use crate::misc::Algebraic;
          eprint!("\x1B[2m");
          eprintln!("key collision or decode error: illegal state");
          eprintln!("  {}", state.to_fen());
          eprintln!(
            "  hint move = {}, src = {}, dst = {}",
            hint_move, hint_move.src.id(), hint_move.dst.id()
          );
          eprintln!("  hint score = {}", hint_score);
          eprint!("\x1B[22m")
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
        && !alpha.is_loss()
        && !lower_estimate.is_sentinel()
        && lower_estimate + PovScore::new(context.tuning.fp_offset + (depth as i16) * context.tuning.fp_scale) < alpha
      { futile = true; }

      if successors > 1
        && futile
        && !mv.is_gainful()
        && !mv.gives_check()
        && mv.score < context.tuning.fp_thresh
      { continue; }

      state.apply(&mv);
    }
    successors += 1;

    if expected == NodeKind::PV && raised_alpha { next_expected = NodeKind::Cut; }

    context.state_history.push((state.key, mv.is_zeroing()));

    let mut score = PovScore::SENTINEL;
    let okay = loop { // TODO replace with labelled block
      if successors > 1 {
        // Step 8c. Reduced-depth zero-window search
        if depth > 2 {
          let mut reduction : i16 = 0;
          if height > 0 && !mv.is_gainful() && successors > 3 {
            let moves_since_raise = (successors - last_raise) as f64;
            let mut r = context.tuning.lmr_base
                      + context.tuning.lmr_scale * ((depth-1) as f64).log2() * moves_since_raise.log2()
                      - context.tuning.hst_scale * (mv.score as f64);
            if !zerowind { r *= context.tuning.fw_ratio; }
            reduction = r.floor() as i16;
          }
          if reduction > 0 {
            let next_depth = std::cmp::max(0, depth as i16 - 1 - reduction) as u8;
            let subscore = main_search(
              state, next_depth, height+1, -alpha-ONE, -alpha,
              true, next_expected, context
            );
            if subscore.is_sentinel() { break false; }
            score = -subscore;
            if !(score > alpha) { break true; }
          }
        }
        // Step 8d. Full-depth zero-window search (within a full-window search)
        if !zerowind {
          let next_depth = depth - 1;
          let subscore = main_search(
            state, next_depth, height+1, -alpha-ONE, -alpha,
            true, next_expected, context
          );
          if subscore.is_sentinel() { break false; }
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
      if subscore.is_sentinel() { break false; }
      score = -subscore;
      break true;
    };

    // Step 8f. State restoration
    context.state_history.pop();
    state.undo(&mv);
    state.restore(&metadata);

    if !okay { return PovScore::SENTINEL; }

    debug_assert!(!score.is_sentinel(), "score was not set");

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
      if depth >= 2 { selector.update_history_scores(context); }
      break;
    }
  }

  // Step 9. Mate detection
  if successors == 0 {
    best_score =
      if !excluded_move.is_null() { alpha }
      else if state.incheck { PovScore::realized_loss(height) }
      else { PovScore::ZERO };
  }
  else if state.dfz == 100 {
    best_score = PovScore::ZERO;
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
      (selector.stage != Stage::Done || best_score.is_realized_winloss()) { NodeKind::Cut }
    // The last case – this is actually a PV node.
    else { NodeKind::PV };

  // Step 11. Transposition table update
  if excluded_move.is_null() {
    let table_score =
      if      best_score.is_proven_win()  { best_score + PovScore::new(height as i16) }
      else if best_score.is_proven_loss() { best_score - PovScore::new(height as i16) }
      else                                { best_score };

    let repl_move =
      if hint_score != PovScore::SENTINEL && kind == NodeKind::All {
        entry.hint_move
      }
      else {
        best_move.compress()
      };

    table_update(state.key, TableEntry {
    //key:        if state.dfz > 92 { state.key ^ state.dfz as u64 } else { state.key },
      key:        state.key as u16,
      generation: generation() as u16,
      hint_move:  repl_move,
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
  context.state_history = history;

  let mut score = resolving_search(&mut state, 0, 0, PovScore::MIN, PovScore::MAX, context);

  'iterative_deepening: for step in 1..((MAX_DEPTH+1) as u8) {
    let init_width = (score.unsigned_abs().max(140).min(500) / 4 - 25) as i16;

    let mut alpha_width = PovScore::new(init_width);
    let mut  beta_width = PovScore::new(init_width);
    let mut alpha_failures = 0;
    let mut  beta_failures = 0;
    let mut searching = true;
    let mut tentative = score;
    while searching {
      let alpha = if alpha_failures > 2 || step <= MINIMUM_DEPTH { PovScore::MIN } else { score - alpha_width };
      let  beta = if  beta_failures > 2 || step <= MINIMUM_DEPTH { PovScore::MAX } else { score +  beta_width };
      context.nominal = step;
      tentative = main_search(
        &mut state, step, 0, alpha, beta, false, NodeKind::PV, context
      );
      if abort() { break 'iterative_deepening; }
      debug_assert!(!tentative.is_sentinel());
      searching = tentative >= beta || alpha >= tentative;
      if alpha >= tentative { alpha_failures += 1; alpha_width = alpha_width * 5 / 2; }
      if tentative >= beta  {  beta_failures += 1;  beta_width =  beta_width * 5 / 2; }
    }
    score = tentative;
  }
}

fn best_move(
  limits    : Limits,
  mut state : State,
  history   : Vec<(u64, bool)>,
)
{
  let clock = unsafe { GLOB.clock.unwrap() };

  let term_width = if isatty(STDERR) { get_window_width() } else { 0 };

  if isatty(STDERR) {
    let s = if syzygy_enabled() { "   tb " } else { "" };
    eprintln!("\x1B[38;5;236m id  dp  sd   time  node   nps{s}  cu  score\x1B[39m");
  }

  // Setup

  let context = unsafe { &mut GLOB.context[0] };
  context.state_history = history;

  let mut prev_time_to_depth = 0.0;
  let mut prev_nodes_to_depth = 0;
  let mut time_ratios = [1.5, 1.5, 1.5, 1.5];
  let mut node_ratios = [1.5, 1.5, 1.5, 1.5];

  let mut last_best = Move::NULL;
  let mut stability = 0;

  let mut last_step = 0;
  let mut last_pv;
  let mut last_score;
  let mut score =
    resolving_search(&mut state, 0, 0, PovScore::MIN, PovScore::MAX, context);
  let mut best = Move::NULL;
  let mut no_change = false;

  // Iterative deepening loop

  'iterative_deepening: for step in 1..(limits.depth+1) {

    // Part 1. Search at the next depth

    last_pv = context.pv[0].clone();

    let init_width = (score.unsigned_abs().max(140).min(500) / 4 - 25) as i16;

    let mut alpha_width = PovScore::new(init_width);
    let mut  beta_width = PovScore::new(init_width);
    let mut alpha_failures = 0;
    let mut  beta_failures = 0;
    let mut searching = true;
    let mut tentative = score;
    while searching {
      let alpha = if alpha_failures > 2 || step <= MINIMUM_DEPTH { PovScore::MIN } else { score - alpha_width };
      let  beta = if  beta_failures > 2 || step <= MINIMUM_DEPTH { PovScore::MAX } else { score +  beta_width };
      context.nominal = step;
      tentative = main_search(
        &mut state, step, 0, alpha, beta, false, NodeKind::PV, context
      );
      if abort() && step > MINIMUM_DEPTH { break 'iterative_deepening; }
      debug_assert!(!tentative.is_sentinel());
      searching = tentative >= beta || alpha >= tentative;
      if alpha >= tentative { alpha_failures += 1; alpha_width = alpha_width * 5 / 2; }
      if tentative >= beta  {  beta_failures += 1;  beta_width =  beta_width * 5 / 2; }
    }
    last_score = score;
    score = tentative;
    if !context.pv[0].is_empty() { best = context.pv[0][0].clone(); }

    last_step = step;
    let time_to_depth = clock.elapsed().as_secs_f64();

    stability = if fast_eq(&best, &last_best) { stability + 1 } else { 1 };
    last_best = best.clone();

    // Part 2. Report to stderr and stdout

    let mut main_depth = 0;
    let mut  ext_depth = 0;
    for x in 0..MAX_HEIGHT { if context.m_nodes_at_height[x] != 0 { main_depth = x; } }
    for x in 0..MAX_HEIGHT { if context.r_nodes_at_height[x] != 0 {  ext_depth = x; } }

    let mut nodes_to_depth = 0;
    let mut tbhits = 0;
    unsafe {
      let num_threads = num_threads();
      for id in 0..num_threads {
        // These values might be slightly out of date, but we don't mind
        nodes_to_depth += GLOB.context[id].m_nodes_at_height.iter().sum::<usize>();
        nodes_to_depth += GLOB.context[id].r_nodes_at_height.iter().sum::<usize>();
        tbhits += GLOB.context[id].tb_hits;
      }
    }

    let nps = nodes_to_depth as f64 / time_to_depth;

    let utilization;
    if time_to_depth >= 0.25 {
      utilization = table_utilization(generation() as u16);
    }
    else {
      utilization = 0;
    }

    if isatty(STDERR) {
      eprint!("\r");  // TODO replace with ANSI escape code
      if no_change { eprint!("\x1B[A\x1B[K"); }
      no_change = score == last_score && context.pv[0] == last_pv;

      // 3 + 1 + 3 + 1 + 3 + 1 + 6 + 1 + 5 = 16
      eprint!(
        "{:3} \x1B[38;5;242m{:3} {:3}\x1B[39m {} {}",
        step, main_depth+1, ext_depth+1,
        format_time(time_to_depth), format_node_compact(nodes_to_depth)
      );
      let mut line_width = 24;

      // 1 + 5 = 6
      if time_to_depth >= 0.25 {
        eprint!(" {}", format_node_compact(nps as usize));
      }
      else {
        eprint!("   \x1B[38;5;234m──\x1B[39m ");
      }
      line_width += 6;

      // 1 + 5 = 6
      if syzygy_enabled() {
        if tbhits == 0 {
          eprint!("   \x1B[38;5;234m──\x1B[39m ");
        }
        else {
          eprint!(" {}", format_node_compact(tbhits));
        }
        line_width += 6;
      }

      // 1 + 3 = 4
      if time_to_depth >= 0.25 {
        if utilization > 999 {
          eprint!(" \x1B[38;5;242m000\x1B[39m");
        }
        else {
          eprint!(" \x1B[38;5;242m{utilization:3}\x1B[39m");
        }
      }
      else {
        eprint!("  \x1B[38;5;234m──\x1B[39m");
      }
      line_width += 4;

      // 1 + 6 = 7
      eprint!(" {:#6}", score.from(state.turn));
      line_width += 7;

      eprint!(" ");
      line_width += 1;

      let mut scratch = state.clone_empty();
      for (idx, mv) in context.pv[0].iter().enumerate() {
        let mut short = mv.disambiguate(&scratch);
        short.hi = if idx == 0 { 1 } else { 2 };
        let mv_width = 1 + short.width as u16;
        if term_width > 0 && line_width + mv_width > term_width { break; }
        eprint!(" {short:#}");
        line_width += mv_width;
        scratch.apply(mv);
      }
      eprint!("\n");
    }
    if !isatty(STDOUT) || !isatty(STDERR) {
      if time_to_depth >= 0.25 {
        print!(
          "info depth {} seldepth {} nodes {} tbhits {} hashfull {} time {:.0} nps {:.0} score {:#o}",
          step, ext_depth+1, nodes_to_depth, tbhits, utilization, time_to_depth * 1000.0, nps, score
        );
      }
      else {
        print!(
          "info depth {} seldepth {} nodes {} tbhits {} time {:.0} nps {:.0} score {:#o}",
          step, ext_depth+1, nodes_to_depth, tbhits, time_to_depth * 1000.0, nps, score
        );
      }
      if !context.pv[0].is_empty() {
        print!(" multipv 1 pv");
        for mv in context.pv[0].iter() { print!(" {:#o}", mv); }
      }
      print!("\n");
    }

    // Part 3. Check time management

    if step < MINIMUM_DEPTH { continue; }

    // We don't break above when step = MINIMUM_DEPTH so that we can write info,
    //   but with that out of the way, we need to break now

    if abort() { break; }

    // Decide whether to stop early if we're given a target (soft limit)

    if let Some(target) = limits.target {

      if time_to_depth > target { break; }

      // Update the history of time to depth ratios
      if prev_time_to_depth > 0.0 {
        time_ratios = [
          time_to_depth / prev_time_to_depth, time_ratios[0], time_ratios[1], time_ratios[2]
        ];
      }
      prev_time_to_depth = time_to_depth;
      // Remove the outlier
      //   Unfortunately, since f64 doesn't implement Ordering, we can't use max_by_key:
      //   let (idx, val) =
      //     ratios.iter().enumerate().max_by_key(|(_, x)| (*x - arith_mean).abs()).unwrap();
      let arith_mean = time_ratios.iter().sum::<f64>() * 0.25;
      let mut idx = 0;
      let mut dev = (time_ratios[0] - arith_mean).abs();
      for x in 1..4 {
        let d = (time_ratios[x] - arith_mean).abs();
        if d >= dev { idx = x; dev = d; }
      }
      let outlier = time_ratios[idx];
      time_ratios[idx] = 1.0;
      // Take the geometric mean
      let geom_mean = time_ratios.iter().product::<f64>().cbrt();
      time_ratios[idx] = outlier;
      // Estimate the next time to depth
      let estimate = time_to_depth * geom_mean;

      if let Some(hard_limit) = limits.cutoff {
        if estimate > hard_limit && stability > 1 { break; }
      }

      if estimate >= target {
        // At this point we know
        //   time-to-depth  <  target  <  estimate
        //      |<- undershoot ->|<- overshoot ->|
        let  overshoot = estimate - target;
        let undershoot = target - time_to_depth;
        if overshoot > undershoot { break; }
      }
    }

    if limits.nodes > 0 {
      let target = limits.nodes;

      if nodes_to_depth > target { break; }

      if prev_nodes_to_depth > 0 {
        node_ratios = [
          nodes_to_depth as f64 / prev_nodes_to_depth as f64,
          node_ratios[0], node_ratios[1], node_ratios[2]
        ];
      }
      prev_nodes_to_depth = nodes_to_depth;
      let arith_mean = node_ratios.iter().sum::<f64>() * 0.25;
      let mut idx = 0;
      let mut dev = (node_ratios[0] - arith_mean).abs();
      for x in 1..4 {
        let d = (node_ratios[x] - arith_mean).abs();
        if d >= dev { idx = x; dev = d; }
      }
      let outlier = node_ratios[idx];
      node_ratios[idx] = 1.0;
      let geom_mean = node_ratios.iter().product::<f64>().cbrt();
      node_ratios[idx] = outlier;
      let estimate = (nodes_to_depth as f64 * geom_mean) as usize;

      if estimate >= target {
        let  overshoot = estimate - target;
        let undershoot = target - nodes_to_depth;
        if overshoot > undershoot { break; }
      }
    }

    // Branch back to the top of the iterative deepening loop
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
    eprint!("  depth {}/{}/{}", last_step, main_depth+1, ext_depth+1);
    eprintln!(
      " \x1B[38;5;236m({:.0}% main {:.0}% leaf {:.0}% extn)\x1B[39m",
      m_nodes as f64 * 100.0 / nodes as f64,
      leaves as f64 * 100.0 / nodes as f64,
      (r_nodes - leaves) as f64 * 100.0 / nodes as f64
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
    println!("bestmove {:#o}", best);
  }

  set_abort(true);

  if limits.cutoff.is_some() {
    unsafe { GLOB.supervisor.clone().unwrap().unpark(); }
  }

  set_searching(false);
}

fn supervise(clock : Instant, cutoff : f64)
{
  unsafe {
    GLOB.supervisor = Some(std::thread::current());
  }
  loop {
    let time_to_go = cutoff - clock.elapsed().as_secs_f64();
    if !(time_to_go > 0.0) {
      set_abort(true);
      break;
    }
    std::thread::park_timeout(Duration::from_secs_f64(time_to_go));
    if abort() { break; }
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub fn worker(thread_id : usize)
{
  loop {
    while abort() {
      std::thread::park();
      if kill() { return; }
    }
    let state;
    let history;
    let limits;
    unsafe {
      state   = GLOB.state.clone();
      history = GLOB.history.clone();
      limits  = GLOB.limits.clone();
    }
    if thread_id == 0 {
      set_num_search(generation());
      best_move(limits, state, history);
    }
    else {
      support(thread_id, state, history);
    }
    unsafe { GLOB.context[thread_id].reset(); }
  }
}

pub fn start_threads(
  handles : &mut Vec<std::thread::JoinHandle<()>>,
  num_threads : usize,
)
{
  // NOTE this should never be called while searching!
  // NOTE except at startup, you should always call reap_threads first!
  // NOTE remember to set tuning after start_threads

  set_num_threads(num_threads);

  unsafe {
    let prev = GLOB.context.len();
    if num_threads > prev {
      let deficit = num_threads - prev;
      for _ in 0..deficit {
        GLOB.context.push(Context::new());
      }
    }
    if num_threads < prev {
      GLOB.context.truncate(num_threads);
    }
  }

  set_abort(true);
  set_kill(false);

  handles.clear();

  for id in 0..num_threads {
    handles.push(
      std::thread::Builder::new()
        .name(format!("search.{}", id))
        .spawn(move || { worker(id); })
        .unwrap()
    );
  }
}

pub fn reap_threads(handles : &mut Vec<std::thread::JoinHandle<()>>)
{
  set_kill(true);

  for h in handles.iter() {
    h.thread().unpark();
  }

  for h in std::mem::replace(handles, Vec::new()) {
    h.join().expect("unable to join search thread");
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// These two remaining methods are called only by the main thread (managing the
//   UCI interface) and so can never be running concurrently.

pub fn start_search(
  handles     : &Vec<std::thread::JoinHandle<()>>,
  state       : &State,
  history     : &Vec<(u64, bool)>,
  limits      : Limits,
) -> Option<std::thread::JoinHandle<()>>
{
  let clock = Instant::now();

  set_searching(true);
  increment_generation();

  let maybe_cutoff = limits.cutoff;

  unsafe {
    // TODO extremely undefined behavior
    GLOB.state   = state.clone();
    GLOB.history = history.clone();
    GLOB.limits  = limits;
    GLOB.clock   = Some(clock);
  }

  set_abort(false);

  for h in handles.into_iter() {
    h.thread().unpark();
  }

  if let Some(cutoff) = maybe_cutoff {
    return Some(
      std::thread::Builder::new()
      .name(String::from("supervisor"))
      .spawn(move || { supervise(clock, cutoff); })
      .unwrap()
    );
  }

  return None;
}

pub fn wait_search(supervisor : &mut Option<std::thread::JoinHandle<()>>)
{
  let supervisor = std::mem::replace(supervisor, None);
  if let Some(handle) = supervisor {
    handle.thread().unpark();
    handle.join().expect("unable to join supervisor");
  }

  while searching() {
    std::thread::sleep(Duration::from_millis(1));
  }
}

pub fn stop_search(supervisor : &mut Option<std::thread::JoinHandle<()>>)
{
  if generation() != num_search() {
    if isatty(STDERR) { eprintln!("note: waiting for worker to unpark"); }
    std::thread::sleep(Duration::from_millis(1));
    while generation() != num_search() {
      std::thread::sleep(Duration::from_millis(1));
    }
  }

  set_abort(true);

  wait_search(supervisor);
}
