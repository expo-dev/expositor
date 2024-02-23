use crate::color::Color::*;
use crate::context::Context;
use crate::misc::NodeKind;
use crate::movegen::Selectivity::Everything;
use crate::movetype::Move;
use crate::piece::Kind::Queen;
use crate::global::*;
use crate::rand::{Rand, RandDist, init_rand};
use crate::resolve::{resolving_search, resolving_search_leaves};
use crate::score::*;
use crate::state::{State, MiniState};
use crate::search::{MINIMUM_DEPTH, MAX_DEPTH, main_search, threefold};
use crate::util::{SetU64, isatty, STDERR};

use std::collections::HashSet;
use std::io::{BufWriter, Write};
use std::fs::{File, OpenOptions};
use std::time::{Instant, Duration};
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Acquire, Release};

// TODO switch everything to game_eval (including resolving_search_leaves)
//   so that the HCE can be used

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// These constants need to be recalculated empirically whenever
//   DFLT_POSN_PARAMS or DFLT_MOVE_PARAMS are changed.
const AVG_POSN_EMIT_PER_GAME : f64 =  200.0;
const AVG_MOVE_EMIT_PER_GAME : f64 = 2500.0;

// Suppose there are k threads selecting openings (and playing out games from them) and that
//   these k threads cannot communicate. How do we avoid collisions (book exits being picked by
//   more than one thread) while also having randomized and uniform coverage? Broadly speaking,
//   we simply make the space of possible book exits to pick from large enough that the chance
//   of a collision is low.
// Suppose we want to generate g book exits (i.e. play out g games) in total, or g / k games per
//   thread. Let s be a multiplier, defined so that that the number of possible book exits to
//   pick from is g × s. Then the fraction of the space that each thread will explore is
//
//   p = (g / k) ÷ (g × s) = 1 / (k × s).
//
//   For any particular thread, and any particular book exit, this is the probability that the
//   thread will pick that book exit. For convenience, let's also define
//
//   q = 1 − p,
//
//   which is the probability that a particular thread does NOT pick a particular book exit.
// The probability that more than one thread picks the same book exit is then
//
//   c = 1 − q^k − k × p × q^(k−1).
//
//   In words, this is the probability that a particular book exit is not picked at all
//     (by any thread) or is picked by only one thread.
// Quite happily, c rapidly converges to a limit as k increases! Taking a look at the graph
//   [https://www.desmos.com/calculator/z2flrwnsms] we can see that c is 0.01 between s = 6
//   and s = 7; in other words, for each book exit, the probability that the book exit is
//   duplicated is less than 1% when s = 7, and this is true independent of the number of
//   threads, k, and the total number of games we want to play, g. To make things nice and
//   round, we pick s = 8 whenever k > 1.

pub struct OpeningParams {
  pub gen_moves     : bool, // emit moves for the policy network instead of pos'ns for the NNUE
  pub num_procs     : u8,   // this is the size of the cohort (see the discussion above)
  pub total_todo    : usize,// this many pos'ns or moves will be emitted in total by the cohort
  pub rand_startpos : bool, // randomize the starting position before selecting an opening
  pub branch_target : f64,  // consider this many moves on average each ply of opening selection
}

pub struct SelfplayParams {
  // Scoring Parameters
  pub viable_score : u16, // skip openings with exit scores greater than this
  pub search_time  : f64, // search positions for this many seconds
  pub max_score    : u16, // don't emit positions or emit moves for positions with scores
  // End Conditions            greater than this
  pub draw_onset : u16, // adjudicate the game as drawn if this many halfmoves have been played
  pub draw_len   : u16, //   and, for this many consecutive ply,
  pub draw_score : u16, //   the absolute score of the position has been this or less
  pub rsgn_len   : u16, // adjudicate the game as won or lost if, for this many consecutive ply,
  pub rsgn_score : u16, //   the absolute score is greater than this
}

const MATE : u16 = PROVEN_MATE as u16;

pub const DFLT_POSN_PARAMS : SelfplayParams =
  SelfplayParams {
    viable_score:  2_00,
    search_time:   0.25,
    max_score:    15_00,
    draw_onset:   80,
    draw_len:     20,
    draw_score:   10,
    rsgn_len:      6,
    rsgn_score:   15_00
  };

pub const DFLT_MOVE_PARAMS : SelfplayParams =
  SelfplayParams {
    viable_score:  2_00,
    search_time:   0.15,
    max_score:     MATE,
    draw_onset:   40,
    draw_len:     10,
    draw_score:   25,
    rsgn_len:      6,
    rsgn_score:    MATE
  };

const QUIET_THRESHOLD : i16 = 5;

static mut SEARCH_COUNTER : AtomicUsize = AtomicUsize::new(0);

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn watchdog_loop()
{
  let mut before = unsafe { SEARCH_COUNTER.load(Acquire) };
  loop {
    let mut after;

    let clock = Instant::now();
    std::thread::park_timeout(Duration::from_secs(60));
    loop {
      // intentional wakeup
      after = unsafe { SEARCH_COUNTER.load(Acquire) };
      if after == usize::MAX { return; }
      // spurious wakeups
      let remaining = 60.0 - clock.elapsed().as_secs_f64();
      if remaining < 1.0 { break; }
      std::thread::park_timeout(Duration::from_secs_f64(remaining));
    }
    if after == before { set_abort(true); }
    before = after;
  }
}

pub fn generate_training_data(
  path    : &str,
  args    : &OpeningParams,
  initial : &State,
  seed    : u64,
) -> std::io::Result<(usize, usize)>
{
  let conversion = if args.gen_moves { AVG_MOVE_EMIT_PER_GAME } else { AVG_POSN_EMIT_PER_GAME };
  let num_games = args.total_todo as f64 / conversion;
  let multiplier = if args.num_procs > 1 { 8.0 } else { 1.0 };
  let space_size = num_games * multiplier;
  let reqd_openings = if args.rand_startpos { space_size / 324.0 } else { space_size };

  let opening_length = reqd_openings.log(args.branch_target).round() as u8;
  let branching_factor = reqd_openings.powf((opening_length as f64).recip());
  let num_todo = (args.total_todo as f64 / args.num_procs as f64).ceil() as usize;

  if isatty(STDERR) {
    eprintln!("  Opening length:   {}", opening_length);
    eprintln!("  Branching factor: {:.1}", branching_factor);
  }
  assert!(branching_factor >= 3.0);

  init_rand(seed);
  let mut out = BufWriter::new(OpenOptions::new().append(true).create(true).open(path)?);
  let mut openings = SetU64::new();
  let mut num_emitted = 0;

  // Start the watchdog
  let watchdog = std::thread::spawn(watchdog_loop);

  let params = if args.gen_moves { DFLT_MOVE_PARAMS } else { DFLT_POSN_PARAMS };
  let mut context = Context::new();

  'explore: while num_emitted < num_todo {
    context.reset();
    let mut state =
      if args.rand_startpos { State::new324(u32::rand() as usize % 324) }
      else { initial.clone_empty() };
    state.initialize_nnue();

    for /* ply */ _ in 0..opening_length {
      let metadata = state.save();
      let (mut early_moves, mut late_moves) = state.legal_moves(Everything);
      early_moves.append(&mut late_moves);
      let mut viable_moves = Vec::with_capacity(32);
      for mut mv in early_moves {
        state.apply(&mv);
        let score =
          -resolving_search(&mut state, 0, 0, LOWEST_SCORE, HIGHEST_SCORE, &mut context);
        state.undo(&mv);
        state.restore(&metadata);
        if score.unsigned_abs() > params.viable_score { continue; }
        // if ply+1 >= opening_length && score.unsigned_abs() < 50 { continue; }
        let mut s = score >> 1;
        if s >  127 { s =  127; }
        if s < -128 { s = -128; }
        mv.score = s as i8;
        viable_moves.push(mv);
      }
      let num_viable = viable_moves.len();
      if num_viable == 0 { continue 'explore; }
      let b = (branching_factor + f64::triangular()).round() as usize;
      let n = std::cmp::min(num_viable, b);
      // Take the n best moves...
      for x in 0..n {
        let mut sel_index = x;
        let mut sel_score = viable_moves[x].score;
        for y in (x+1)..num_viable {
          if viable_moves[y].score > sel_score {
            sel_index = y;
            sel_score = viable_moves[y].score;
          }
        }
        if sel_index != x { viable_moves.swap(x, sel_index); }
      }
      // ... and randomly select one.
      let mv = &viable_moves[u32::rand() as usize % n];
      state.apply(mv);
    }
    let visited = !openings.insert(state.key);
    if visited { continue; }
    num_emitted +=
      if args.gen_moves { selfplay_moves(&mut out, &state, &params)? }
      else              { selfplay_posns(&mut out, &state, &params)? };
  }

  // Stop the watchdog
  unsafe { SEARCH_COUNTER.store(usize::MAX, Release); }
  watchdog.thread().unpark();
  watchdog.join().expect("unable to join thread");

  return Ok((openings.len(), num_emitted));
}

fn selfplay_posns(
  out     : &mut BufWriter<File>,
  initial : &State,
  params  : &SelfplayParams,
) -> std::io::Result<usize>
{
  // NOTE make sure that Hash, Persist, and SyzygyPath are set to the
  //   values you want, since those values will be used during search.

  let mut num_emitted = 0;

  let mut emitted   = HashSet::<u64>::new();
  let mut positions = Vec::<MiniState>::new();
  let mut leaves    = Vec::<(State, i16)>::new();

  let mut state = initial.clone_truncated();
  let mut context = Context::new();

  let mut halfmoves_played : u16 = 0;
  let mut draw_run : u16 = 0;
  let mut rsgn_run : u16 = 0;
  let outcome;
  loop {
    // We first obtain an estimate to determine the initial aspiration window
    //   and, while we're at it, gather some leaves.
    increment_generation();
    let static_eval = (state.evaluate() * 100.0).round() as i16;
    let q_eval = resolving_search_leaves(
      &mut state, 0, 0, LOWEST_SCORE, HIGHEST_SCORE, &mut context, &mut leaves
    );
    let (score, mv) = search_score(&mut state, &mut context, q_eval, params.search_time);
    let score = match state.turn { White => score, Black => -score };
    if mv.is_null() { outcome = Outcome::Draw; break; }
    if score == PROVEN_LOSS {
      outcome = match state.turn {
        White => Outcome::Black,
        Black => Outcome::White
      }; break;
    }

    // If the score is within range and this is a quiet position, add it to the list of
    //   game positions that we'll emit (that are both scored and marked with an outcome).
    if score.unsigned_abs() <= params.max_score
    && (static_eval - q_eval).abs() <= QUIET_THRESHOLD {
      emitted.insert(state.key);
      let mut mini = MiniState::from(&state);
      mini.score = score;
      positions.push(mini);
    }

    state.apply(&mv);
    context.state_history.push((state.key, mv.is_zeroing()));
    halfmoves_played += 1;

    // eprintln!(
    //   "{:3} {:7} {:+5} {:+5} {:+5}",
    //   halfmoves_played, &mv, score, q_eval, static_eval
    // );

    if state.dfz > 100 || threefold(&context.state_history) {
      outcome = Outcome::Draw;
      break;
    }

    draw_run = if score.unsigned_abs() > params.draw_score { 0 } else { draw_run + 1 };
    rsgn_run = if score.unsigned_abs() > params.rsgn_score { rsgn_run + 1 } else { 0 };

    if draw_run >= params.draw_len && halfmoves_played >= params.draw_onset {
      outcome = Outcome::Draw;
      break;
    }
    if rsgn_run >= params.rsgn_len {
      outcome = if score < 0 { Outcome::Black } else { Outcome::White };
      break;
    }
  }

  for mut mini in positions {
    mini.set_outcome(outcome);
    out.write(mini.to_quick().as_bytes())?;
    out.write(b"\n")?;
    num_emitted += 1;
  }

  for (mut leaf, q_eval) in leaves {
    if !emitted.insert(leaf.key) { continue; }
    context.state_history.clear();
    let (score, _) = search_score(&mut leaf, &mut context, q_eval, params.search_time);
    if score.unsigned_abs() > params.max_score { continue; }
    let mut mini = MiniState::from(&leaf);
    mini.score = match leaf.turn { White => score, Black => -score };
    out.write(mini.to_quick().as_bytes())?;
    out.write(b"\n")?;
    num_emitted += 1;
  }

  out.flush()?;
  return Ok(num_emitted);
}

fn saturate(x : i32) -> i16
{
  let x = std::cmp::min(x, PROVEN_MATE as i32);
  let x = std::cmp::max(x, PROVEN_LOSS as i32);
  return x as i16;
}

fn selfplay_moves(
  out     : &mut BufWriter<File>,
  initial : &State,
  params  : &SelfplayParams,
) -> std::io::Result<usize>
{
  #![allow(non_upper_case_globals)]

  // Training data for the policy network needs to include positions that are
  //   NOT quiet (although we are generally only interested in quiet moves),
  //   since the primary purpose of the policy network is to adjust the ordering
  //   and reduction of quiet moves but the policy network will be queried in
  //   any kind of position (quiet or not).
  // We may, however, want to restrict our attention to positions where a
  //   capture isn't the best move (this may incidentally be one's definition
  //   of a quiet position, but I consider that to be a practical matter rather
  //   than a theoretical one) – the justification being that this is when the
  //   ordering and reduction of quiet moves matters most.
  // If we don't restrict our attention to such positions, we may want to
  //   measure centipawn loss for quiet moves relative to the best quiet move
  //   rather than the best move overall.

  // Skip positions where the best move is a capture.
  const only_quiet_best : bool = false;

  // Emit only quiet moves.
  const emit_only_quiet : bool = false;

  // NOTE make sure that Hash, Persist, and SyzygyPath are set to the values you
  //   want, since those values will be used during search.

  let mut num_emitted = 0;

  let mut state = initial.clone_truncated();
  let mut context = Context::new();

  let mut halfmoves_played : u16 = 0;
  let mut draw_run : u16 = 0;
  let mut rsgn_run : u16 = 0;

  // (kind + rectified src + recitified dst, score)
  let mut move_scores : Vec<(u16, i16)> = Vec::new();

  let mut prev_score : i16 = 0;

  loop {
    increment_generation();
    let q_eval = resolving_search(&mut state, 0, 0, LOWEST_SCORE, HIGHEST_SCORE, &mut context);
    let (pov, mv) = search_score(&mut state, &mut context, q_eval, params.search_time);
    if mv.is_null() || pov == PROVEN_LOSS { break; }

    let score = match state.turn { White => pov, Black => -pov };
    // eprintln!("{:3} {:7} {:+5}", halfmoves_played, &mv, score);

    if prev_score.unsigned_abs() <= params.max_score
      && !(only_quiet_best && mv.is_gainful())
    {
      let color = state.turn;

      move_scores.clear();

      let metadata = state.save();
      let (early_moves, late_moves) = state.legal_moves(Everything);
      let maybe_early =
        if emit_only_quiet { Vec::new().into_iter() } else { early_moves.into_iter() };

      for nonsense in maybe_early.chain(late_moves.into_iter()) {
        if nonsense.is_promotion() && nonsense.promotion.kind() != Queen { continue; }

        state.apply(&nonsense);

        let q_eval =
          resolving_search(&mut state, 0, 0, LOWEST_SCORE, HIGHEST_SCORE, &mut context);
        let (hyp, _) = search_score(&mut state, &mut context, q_eval, params.search_time);
        let hyp = match state.turn { White => hyp, Black => -hyp };
        // eprintln!("    \x1B[2m{:7} {:+5}\x1B[22m", &nonsense, hyp);

        let kind = nonsense.piece.kind() as u16;
        let src = nonsense.src as usize;
        let src = match color { White => src, Black => crate::misc::vmirror(src) } as u16;
        let dst = nonsense.dst as usize;
        let dst = match color { White => dst, Black => crate::misc::vmirror(dst) } as u16;
        let m = (kind << 12) | (src << 6) | dst;
        move_scores.push((m, hyp));

        state.undo(&nonsense);
        state.restore(&metadata);
      }

      let emit = !move_scores.is_empty();

      if emit {
        let mut mini = MiniState::from(&state);
        mini.score = score;
        out.write(mini.to_quick().as_bytes())?;
      }

      for (m, h) in move_scores.iter() {
        out.write(&[b' '])?;
        use crate::quick::enc;
        let u = *h as u16;
        let u0 = ( u        & 63) as u8;
        let u1 = ((u >>  6) & 63) as u8;
        let u2 = ((u >> 12) & 63) as u8;
        out.write(&[enc(u0), enc(u1), enc(u2)])?;
        let k = ((m >> 12) &  7) as u8;
        let s = ((m >>  6) & 63) as u8;
        let d = ( m        & 63) as u8;
        out.write(&[enc(k), enc(s), enc(d)])?;
        num_emitted += 1;
      }

      if emit { out.write(b"\n")?; }
    }

    prev_score = score;

    state.apply(&mv);
    context.state_history.push((state.key, mv.is_zeroing()));
    halfmoves_played += 1;

    if state.dfz > 100 || threefold(&context.state_history) { break; }

    draw_run = if score.unsigned_abs() > params.draw_score { 0 } else { draw_run + 1 };
    rsgn_run = if score.unsigned_abs() > params.rsgn_score { rsgn_run + 1 } else { 0 };

    if draw_run >= params.draw_len && halfmoves_played >= params.draw_onset { break; }
    if rsgn_run >= params.rsgn_len { break; }
  }

  out.flush()?;
  return Ok(num_emitted);
}

fn search_score(
  state   : &mut State,
  context : &mut Context,
  guess   : i16,
  target  : f64,
) -> (i16, Move)
{
  set_abort(false);
  unsafe {
    // This doesn't need to be an atomic increment,
    //   since no other thread writes to counter.
    let counter = SEARCH_COUNTER.load(Acquire);
    SEARCH_COUNTER.store(counter+1, Release);
  }

  context.reset_search();
  context.reset_stats();

  let mut prev_time_to_depth = 0.0;
  let mut ratios = [1.5, 1.5, 1.5, 1.5];

  let mut score = guess;
  let mut best = Move::NULL;

  let clock = Instant::now();
  for step in 1..((MAX_DEPTH+1) as u8) {
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
      let alpha = if alpha_failures >= 2 {  LOWEST_SCORE } else { score - alpha_width };
      let  beta = if  beta_failures >= 2 { HIGHEST_SCORE } else { score +  beta_width };
      context.nominal = step;
      tentative = main_search(state, step, 0, alpha, beta, false, NodeKind::PV, context);
      if tentative == INVALID_SCORE {
        // GLOB.abort should already be set, since the only reason we expect a search to return
        //   an invalid score is because the watchdog set GLOB.abort and the search exited.
        assert!(abort());
        let elapsed = clock.elapsed().as_secs_f64();
        let absscore = score.unsigned_abs();
        if 1_00 < absscore && absscore < 10_00 {
          eprintln!("invalid score: search timeout");
          eprintln!("  {}", state.to_fen());
          eprintln!("  step = {step}, clock = {elapsed:.1}");
          eprintln!(
            "  prev score = {}, best move = {}",
            format_score(score), best.in_context(state)
          );
        }
        assert!(step >= MINIMUM_DEPTH);
        return (score, best);
      }
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
  set_abort(true);
  return (score, best);
}

pub fn approx_time_search(
  state   : &mut State,
  context : &mut Context,
  guess   : i16,
  target  : f64,
) -> (i16, Move)
{
  return search_score(state, context, guess, target);
}
