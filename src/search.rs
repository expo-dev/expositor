use crate::algebraic::*;
use crate::cache::*;
use crate::color::*;
use crate::context::*;
use crate::constants::*;
use crate::misc::*;
use crate::limits::*;
use crate::movegen::*;
use crate::movesel::*;
use crate::movetype::*;
use crate::piece::*;
use crate::resolve::*;
use crate::score::*;
use crate::state::*;
use crate::syzygy::*;
use crate::tablebase::*;
use crate::util::*;

use std::time::{Instant, Duration};

const MINIMUM_DEPTH : u8 = 4;

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

pub static mut GENERATION : usize = 0;
pub static mut USE_PREV_GEN : bool = false;

pub static mut GLOBAL_STATISTICS : Statistics = Statistics::new();

static mut ABORT              : bool = false;
static mut SEARCH_IN_PROGRESS : bool = false;

static mut SUPERVISOR  : Option<std::thread::Thread> = None;
static mut NUM_THREADS : usize                       = 0;
static mut CONTEXT     : Vec<Context>                = Vec::new();
static mut STATISTICS  : Vec<Statistics>             = Vec::new();

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn threefold(history : &Vec<(u64, bool)>) -> bool
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

fn twofold(history : &Vec<(u64, bool)>) -> bool
{
  let current = match history.last() { Some(a) => a.0, None => return false };
  let end = history.len() - 1;
  for &(position, zeroing) in history[0..end].iter().rev() {
    if position == current { return true; }
    if zeroing { return false; }
  }
  return false;
}

fn main_search(
  state    : &mut State,
  depth    : u8,
  height   : u8,
  alpha    : i16,
  beta     : i16,
  zerowind : bool,
  // = zero window (when true, this node is not expected to be part of the principal variation)
  expected : NodeKind,
  context  : &mut Context,
  stats    : &mut Statistics,
) -> i16
{
  table_prefetch(state.key);

  context.pv[height as usize].clear();
  unsafe { if ABORT && context.nominal > MINIMUM_DEPTH { return 0; } }

  // Step 0. 50-move rule and draw-by-repetition detection
  // The condition for two-fold repetition needs to be
  //   at least "height > 1" to prevent false draw scores.
  if state.dfz > 100 { return 0; }
  if height > 3 { if twofold(&context.state_history) { return 0; } }
  else        { if threefold(&context.state_history) { return 0; } }

  // Step 1. Resolving search
  if depth == 0 { return resolving_search(state, 0, height, alpha, beta, context, stats); }

  // Step 2. Update statistics
  stats.m_nodes_at_height[height as usize] += 1;
  let mut alpha = alpha;
  let mut beta  = beta;

  // Step 3. Mate-distance pruning and internal tablebase access
  if height > 0 {
    let worst_possible = PROVEN_LOSS + (height as i16);
    let  best_possible = PROVEN_MATE - (height as i16 + 1);
    alpha = std::cmp::max(alpha, worst_possible);
    beta  = std::cmp::min( beta,  best_possible);
    if alpha >= beta { return alpha; }
  }

  let excluded_move = context.exclude[height as usize].clone();

  if height > 0 && state.rights == 0 && excluded_move.is_null() && state.dfz == 0
  {
    let men = (state.sides[W] | state.sides[B]).count_ones();
    if men == 3 { return probe_3man(state, height as i16); }
    if syzygy_active() && syzygy_support() >= men {
      if let Some(score) = probe_syzygy_wdl(&state, height as i16) {
        stats.tb_hits += 1;
        return score;
      }
    }
  }

  // Step 4. Transposition table lookup and internal iterative reduction
  let mut depth = depth;

  let mut hint_move = NULL_MOVE;
  let mut hint_score = 0;

  let entry =
    if excluded_move.is_null() {
      // We mix dfz into the key when dfz is close to 100 to prevent false
      //   positives and negatives in draw detection due to transpositions.
      table_lookup(if state.dfz > 92 { state.key ^ state.dfz as u64 } else { state.key })
    }
    else {
      NULL_ENTRY
    };

  let okay_generation;
  unsafe {
    okay_generation = USE_PREV_GEN || (entry.generation == GENERATION as u16);
  }
  if entry.hint_move != 0 && okay_generation {
    let score =
      if      entry.hint_score >= MINIMUM_TB_MATE { entry.hint_score - height as i16 }
      else if MINIMUM_TB_LOSS >= entry.hint_score { entry.hint_score + height as i16 }
      else                                        { entry.hint_score                 };
    if zerowind && entry.depth >= depth {
      if (entry.kind as u8) & (NodeKind::Cut as u8) != 0 && score >= beta  { return score; }
      if (entry.kind as u8) & (NodeKind::All as u8) != 0 && alpha >= score { return score; }
    }
    hint_move = Move::decompress(state, entry.hint_move);
    hint_score = score;
    #[cfg(debug_assertions)]
    if hint_move.is_null() {
      eprintln!("{}", state.to_fen());
      eprintln!("entry.hint_move = {:016b}", entry.hint_move);
      let src = (   entry.hint_move     & 0x003F) as i8;
      let dst = ((entry.hint_move >> 6) & 0x003F) as i8;
      eprintln!("src = {}, dst = {}", src.algebraic(), dst.algebraic());
      panic!("key collision or decode error");
    }
  }
  else if height >= 2 && depth >= 6 && excluded_move.is_null() {
    depth -= 1;
  }

  // Step 5. Snapshot saving
  let metadata = state.save();

  // Step 6. Reductions and pruning
  let mut lower_estimate = i16::MIN;
  let mut futile = false;

  if context.nominal > 4 && height > 0 && zerowind && !state.incheck {

    let ofs = (state.turn as usize) * 8;
    let pieces = state.boards[ofs + QUEEN ]
               | state.boards[ofs + ROOK  ]
               | state.boards[ofs + BISHOP]
               | state.boards[ofs + KNIGHT];
    let num_pieces = pieces.count_ones();

    let static_eval =
      if !hint_move.is_null() && entry.kind == NodeKind::PV { 0 /* this will be ignored */ }
    //else if depth > 4 { resolving_search(state, 0, height, alpha, beta, context, stats) }
      else { state.evaluate_in_game() };

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
      let null_score = -main_search(
        state, null_depth as u8, height+1, -beta, -beta+1, true,
        NodeKind::All, context, stats
      );
      state.undo_null();
      state.restore(&metadata);
      if null_score >= beta {
        // At high depths, add verification search? (regular search at low depth)
        //   Verification search is required to also equal or exceed beta to
        //   actually perform the pruning.
        if num_pieces >= 2 { return null_score; }
        depth = std::cmp::min(depth - 4, depth / 2);
      }
    }
  }
  context.null[height as usize] = false;

  // Step 7. Extensions and multicut

  //   Step 7a. Check extension
  //   NOTE should this or the original depth be recorded in the cache update?
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
    && hint_score.abs() < MINIMUM_TB_MATE
  {
    let reduction = SSE_BASE + (depth as i32)*SSE_SCALE;
    let sse_depth = ((depth as u32)*4096 - (reduction as u32)) / 4096;

    let margin = (SSE_MARGIN_OFFSET + (depth as i32)*SSE_MARGIN_SCALE) / 256;
    let margin = margin as i16;

    context.exclude[height as usize] = hint_move.clone();
    let singular_score = main_search(
      state, sse_depth as u8, height, hint_score-margin-1, hint_score-margin, true,
      expected, context, stats
    );
    context.exclude[height as usize] = NULL_MOVE;
    if singular_score >= hint_score-margin {
      if hint_score-margin >= beta { return hint_score-margin; }
    }
    else {
      extend_hint = true;
    }
  }

  // Step 8. Move iteration
  let mut selector = MoveSelector::new(
    Selectivity::Everything,
    height,
    if excluded_move.is_null() { hint_move.clone() } else { excluded_move.clone() }
  );

  let mut best_score = i16::MIN + 1;
  let mut best_move  = NULL_MOVE;

  let mut successors   = 0;
  let mut raised_alpha = false;
  let mut last_raise   = 0;

  let mut expected_child = match expected {
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
          eprintln!("{}", state.to_fen());
          eprintln!("hint move = {}", hint_move);
          panic!("key collision or decode error");
        }
        #[cfg(not(debug_assertions))] {
          state.undo(&hint_move);
          state.restore(&metadata);
          hint_move = NULL_MOVE;
          continue;
        }
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
        && lower_estimate != i16::MIN
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

    if expected == NodeKind::PV && raised_alpha { expected_child = NodeKind::Cut; }

    context.state_history.push((state.key, mv.is_zeroing()));

    let mut trace = if zerowind { IN_ZW } else { 0 };
    let mut score;
    loop {
      if successors > 1 {
        // Step 8c. Reduced-depth zero-window search
        if depth > 2 {
          let mut reduction : i8 = 0;
          if height > 0 && !mv.is_gainful() && successors > 3 {
            let moves_since_raise = (successors - last_raise) as f64;
            let mut r = LMR_BASE
                      + LMR_SCALE * ((depth-1) as f64).log2() * moves_since_raise.log2()
                      - HST_SCALE * (mv.score as f64);
            if !zerowind { r *= FW_RATIO; }
            reduction = r.floor() as i8;
          }
          if reduction > 0 {
            trace |= ZR;
            let next_depth = std::cmp::max(0, depth as i8 - 1 - reduction) as u8;
            score = -main_search(
              state, next_depth, height+1, -alpha-1, -alpha, true,
              expected_child, context, stats
            );
            if !(score > alpha) { trace |= ZR_OK; break; }
          }
        }
        // Step 8d. Full-depth zero-window search (within a full-window search)
        if !zerowind {
          trace |= ZF;
          let next_depth = depth - 1;
          score = -main_search(
            state, next_depth, height+1, -alpha-1, -alpha, true,
            expected_child, context, stats
          );
          if !(score > alpha) { trace |= ZF_OK; break; }
        }
      }
      else {
        trace |= FST;
      }
      // Step 8e. Full-depth full-window search (within a full-window search) or
      //            zero-window search (within a zero-window search)
      trace |= FF;
      let next_depth = if successors == 1 && extend_hint { depth } else { depth - 1 };
      score = -main_search(
        state, next_depth, height+1, -beta, -alpha, zerowind,
        expected_child, context, stats
      );
      break;
    }
    debug_assert!(score != i16::MIN, "score was not set");

    // Step 8f. State restoration
    context.state_history.pop();
    state.undo(&mv);
    state.restore(&metadata);

    // Step 8g. Tracing
    match trace {
      FULL_ETC_ZWRD           => { stats.full_etc_zwrd           += 1; }
      FULL_ETC_ZWRD_ZWFD      => { stats.full_etc_zwrd_zwfd      += 1; }
      FULL_ETC_ZWFD           => { stats.full_etc_zwfd           += 1; }
      FULL_ETC_ZWRD_ZWFD_FWFD => { stats.full_etc_zwrd_zwfd_fwfd += 1; }
      FULL_ETC_ZWFD_FWFD      => { stats.full_etc_zwfd_fwfd      += 1; }
      FULL_FST_FWFD           => { stats.full_fst_fwfd           += 1; }
      ZERO_ETC_ZWRD           => { stats.zero_etc_zwrd           += 1; }
      ZERO_ETC_ZWRD_FWFD      => { stats.zero_etc_zwrd_fwfd      += 1; }
      ZERO_ETC_FWFD           => { stats.zero_etc_fwfd           += 1; }
      ZERO_FST_FWFD           => { stats.zero_fst_fwfd           += 1; }
      _ => { panic!("unrecognized trace {:08b}", trace); }
    }

    // Step 8h. Folding
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

    // Step 8i. Beta cutoff
    if alpha >= beta {
      if !mv.is_gainful() {
        let killers = &mut context.killer_table[height as usize];
        if !quick_eq(&mv, &killers.0) {
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
      (selector.stage != Stage::Done || best_score.abs() >= MINIMUM_PROVEN_MATE) { NodeKind::Cut }
    // The last case – this is actually a PV node.
    else { NodeKind::PV };

  // Step 11. Transposition table update
  unsafe {
    if ABORT && context.nominal > MINIMUM_DEPTH {
      if kind == NodeKind::PV { context.pv[height as usize].insert(0, best_move); }
      return best_score;
    }
  }

  if excluded_move.is_null() {
    let table_score =
      if      best_score >= MINIMUM_TB_MATE { best_score + height as i16 }
      else if best_score <= MINIMUM_TB_LOSS { best_score - height as i16 }
      else                                  { best_score                 };

    table_update(TableEntry {
      key:        if state.dfz > 92 { state.key ^ state.dfz as u64 } else { state.key },
      generation: unsafe { GENERATION as u16 },
      hint_move:  best_move.compress(),
      hint_score: table_score,
      depth:      depth,
      kind:       kind,
    });
  }

  // Step 12. Updating statistics
  if selector.stage != Stage::GenerateMoves { stats.nodes_w_num_moves[selector.len()] += 1; }

  if hint_move.is_null() {
    stats.last_at_movenum_w_miss[successors as usize] += 1;
    if kind == NodeKind::Cut { stats.cut_at_movenum_w_miss[successors as usize] += 1; }
    if kind == NodeKind::PV  {  stats.pv_at_movenum_w_miss[last_raise as usize] += 1; }
  }
  else {
    stats.last_at_movenum_w_hit[successors as usize] += 1;
    if kind == NodeKind::Cut { stats.cut_at_movenum_w_hit[successors as usize] += 1; }
    if kind == NodeKind::PV  {  stats.pv_at_movenum_w_hit[last_raise as usize] += 1; }
  }

  // Step 13. Returning
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
  let statistics = unsafe { &mut STATISTICS[thread_id] };
  let context    = unsafe { &mut    CONTEXT[thread_id] };
  statistics.reset();
  context.reset();
  context.state_history = history;

  let mut score = resolving_search(&mut state, 0, 0, i16::MIN+1, i16::MAX, context, statistics);

  'iterative_deepening: for step in 1..65 {
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
      let alpha = if alpha_failures >= 2 { i16::MIN+1 } else { score - alpha_width };
      let  beta = if  beta_failures >= 2 { i16::MAX   } else { score +  beta_width };
      context.nominal = step;
      tentative = main_search(
        &mut state, step, 0, alpha, beta, false, NodeKind::PV, context, statistics
      );
      unsafe {
        if ABORT { break 'iterative_deepening; }
      }
      searching = tentative >= beta || alpha >= tentative;
      if alpha >= tentative { alpha_failures += 1; alpha_width *= 2; }
      if tentative >= beta  {  beta_failures += 1;  beta_width *= 2; }
    }
    score = tentative;
  }
}

fn best_move(
  mut state : State,
  history   : Vec<(u64, bool)>,
  limits    : Limits,
)
{
  // Setup

  let statistics = unsafe { &mut STATISTICS[0] };
  let context    = unsafe { &mut    CONTEXT[0] };
  statistics.reset();
  context.reset();
  context.state_history = history;

  let mut prev_time_to_depth = 0.0;
  let mut ratios = [1.5, 1.5, 1.5, 1.5];
  let mut last_best = NULL_MOVE;
  let mut stability = 0;
  let mut wgt_stbly : u16 = 0;
  let mut wgt_actv  : u16 = 0;

  let mut last_step = 0;
  let mut last_pv;
  let mut last_score;
  let mut score =
    resolving_search(&mut state, 0, 0, i16::MIN+1, i16::MAX, context, statistics);
  let mut best = NULL_MOVE;

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
      let alpha = if alpha_failures >= 2 { i16::MIN+1 } else { score - alpha_width };
      let  beta = if  beta_failures >= 2 { i16::MAX   } else { score +  beta_width };
      context.nominal = step;
      tentative = main_search(
        &mut state, step, 0, alpha, beta, false, NodeKind::PV, context, statistics
      );
      unsafe {
        if ABORT && step > MINIMUM_DEPTH { break 'iterative_deepening; }
      }
      searching = tentative >= beta || alpha >= tentative;
      if alpha >= tentative { alpha_failures += 1; alpha_width *= 2; }
      if tentative >= beta  {  beta_failures += 1;  beta_width *= 2; }
    }
    last_score = score;
    score = tentative;
    best = if context.pv[0].is_empty() { NULL_MOVE } else { context.pv[0][0].clone() };
    last_step = step;
    let time_to_depth = clock.elapsed().as_secs_f64();

    if quick_eq(&best, &last_best) {
      stability += 1;
      wgt_stbly += step as u16;
    }
    else {
      stability = 1;
      wgt_stbly = step as u16;
      wgt_actv += step as u16;
    };
    last_best = best.clone();

    // Part 2. Report to stderr and stdout

    let mut ext_depth = 0;
    for x in 0..128 { if statistics.r_nodes_at_height[x] != 0 { ext_depth = x; } }

    let mut nodes  = 0;
    let mut tbhits = 0;
    unsafe {
      let num_threads = NUM_THREADS;
      for id in 0..num_threads {
        // These values might be slightly out of date, but we don't mind
        nodes  += STATISTICS[id].m_nodes_at_height.iter().sum::<usize>();
        nodes  += STATISTICS[id].r_nodes_at_height.iter().sum::<usize>();
        tbhits += STATISTICS[id].tb_hits;
      }
    }

    let nps = nodes as f64 / time_to_depth;

    if isatty(STDERR) {
      if score == last_score && context.pv[0] == last_pv {
        eprint!("{:>2}\r", step);
      }
      else {
        let rectified = if state.turn == Color::Black { -score } else { score };
        eprint!(
          "{:2} \x1B[2m{}/{}\x1B[22m {:6.3} \x1B[2m{:6}\x1B[22m \x1B[1m{:>6}\x1B[22m",
          step, alpha_failures, beta_failures,
          time_to_depth, nodes / 1000, format_score(rectified)
        );
        for mv in context.pv[0].iter() { eprint!(" {}", mv); }
        eprint!("\n");
      }
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

      unsafe { if ABORT { break; } }

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

        let s = 0.002_469_135_802_469_135_8;
        let t = 0.333_333_333_333_333_333_3;
        let downfactor =
          if stability == 1       { 1.5 }
          else if wgt_stbly <  30 { 1.0 }
          else if wgt_stbly < 300 { 1.0 - s * (wgt_stbly - 30) as f64 }
          else                    { t };
        let upfactor =
          if      wgt_actv < 10 { 1.0 }
          else if wgt_actv < 60 { 1.0 + 0.02 * (wgt_actv - 10) as f64 }
          else                  { 2.0 };
        let target = target * upfactor * downfactor;

        if estimate - target > target - time_to_depth { break; }
      }
    }
  }

  // Report final summary and bestmove

  let duration = clock.elapsed().as_secs_f64();

  let mut main_depth = 0;
  let mut  ext_depth = 0;
  for x in 0..128 { if statistics.m_nodes_at_height[x] != 0 { main_depth = x; } }
  for x in 0..128 { if statistics.r_nodes_at_height[x] != 0 {  ext_depth = x; } }

  let mut m_nodes = 0;
  let mut r_nodes = 0;
  let mut leaves  = 0;
  unsafe {
    let num_threads = NUM_THREADS;
    for id in 0..num_threads {
      // These values might be slightly out of date, but we don't mind
      m_nodes += STATISTICS[id].m_nodes_at_height.iter().sum::<usize>();
      r_nodes += STATISTICS[id].r_nodes_at_height.iter().sum::<usize>();
      leaves  += STATISTICS[id].r_nodes_at_length[0];
    }
  }

  let nodes = m_nodes + r_nodes;
  let nps = nodes as f64 / duration;

  if isatty(STDERR) {
    eprintln!("depth {}/{}/{}", last_step, main_depth+1, ext_depth+1);
    eprintln!(
      "\x1B[2m{:6} knode main {:2.0}%\x1B[22m",
      m_nodes / 1000, m_nodes as f64 * 100.0 / nodes as f64
    );
    eprintln!(
      "\x1B[2m{:6} knode leaf {:2.0}%\x1B[22m",
      leaves / 1000, leaves as f64 * 100.0 / nodes as f64
    );
    eprintln!(
      "\x1B[2m{:6} knode extn {:2.0}%\x1B[22m",
      (r_nodes-leaves) / 1000, (r_nodes-leaves) as f64 * 100.0 / nodes as f64
    );
    eprintln!("{:6} knode", nodes / 1000);
    eprintln!("{:6.0} knode/s", nps / 1000.0);
    let num_threads = unsafe { NUM_THREADS };
    if num_threads > 1 { eprintln!("{:6.0} kn/s/th", nps / (num_threads as f64) / 1000.0); }
    eprintln!("{:6.3} seconds", duration);
  }
  if !isatty(STDOUT) || !isatty(STDERR) {
    println!("bestmove {}", best.algebraic());
  }

  // Stop the search if it hasn't been stopped already

  unsafe { if !ABORT { ABORT = true; if let Some(thread) = &SUPERVISOR { thread.unpark(); } } }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn supervise(
  state       : State,
  history     : Vec<(u64, bool)>,
  limits      : Limits,
  num_threads : usize,
)
{
  unsafe {
    SUPERVISOR  = Some(std::thread::current());
    NUM_THREADS = num_threads;
  }

  let time_limit = limits.cutoff;
  let clock = Instant::now();

  unsafe {
    if CONTEXT.len() < num_threads {
      let deficit = num_threads - CONTEXT.len();
      for _ in 0..deficit { CONTEXT.push(Context::new()); }
    }
    if STATISTICS.len() < num_threads {
      let deficit = num_threads - STATISTICS.len();
      for _ in 0..deficit { STATISTICS.push(Statistics::new()); }
    }
    ABORT = false;
  }

  let mut handles = Vec::new();
  {
    let state = state.clone();
    let history = history.clone();
    handles.push(
      std::thread::Builder::new()
        .name(String::from("search.0"))
        .spawn(move || { best_move(state, history, limits); })
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
      unsafe {
        let time_to_go = stop_time - clock.elapsed().as_secs_f64();
        if !(time_to_go > 0.0) {
          ABORT = true;
          break;
        }
        std::thread::park_timeout(Duration::from_secs_f64(time_to_go));
        if ABORT { break; }
      }
    }
  }

  for h in handles { h.join().expect("unable to join search thread"); }

  unsafe {
    for id in 0..num_threads { GLOBAL_STATISTICS.add(&STATISTICS[id]); }
    SEARCH_IN_PROGRESS = false;
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// These three remaining methods are called only by the main thread
//   (managing the UCI interface) and so can never be running concurrently.

pub fn search_in_progress() -> bool { return unsafe { SEARCH_IN_PROGRESS }; }

pub fn start_search(
  state       : &State,
  history     : &Vec<(u64, bool)>,
  limits      : Limits,
  num_threads : usize,
)
{
  let num_threads = std::cmp::max(num_threads, 1);
  unsafe {
    GENERATION += 1;
    SEARCH_IN_PROGRESS = true;
  }
  let state = state.clone();
  let history = history.clone();
  std::thread::Builder::new()
    .name(String::from("supervisor"))
    .spawn(move || { supervise(state, history, limits, num_threads); })
    .unwrap();
}

// NOTE there are two possible race conditions while using stop: first, the supervisor has been
//   started but has not parked yet. This will not break the program but means the search will
//   not be stopped.
//
//   Second, SUPERVISOR is already set to Some(...), but the current supervisor has not yet set
//   SUPERVISOR to itself (and so we're grabbing the previous supervisor). I don't believe this
//   will break the program, but it means the search will not be stopped.
//
pub fn stop_search()
{
  unsafe { ABORT = true; if let Some(thread) = &SUPERVISOR { thread.unpark(); } }
}
