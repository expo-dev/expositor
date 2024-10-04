use crate::color::Color::*;
use crate::context::Context;
use crate::misc::NodeKind;
use crate::movegen::Selectivity::Everything;
use crate::movetype::Move;
use crate::global::*;
use crate::rand::{Rand, RandDist, init_rand};
use crate::resolve::{resolving_search, resolving_search_leaves};
use crate::score::{PovScore, CoOutcome};
use crate::state::{State, MiniState};
use crate::search::{MINIMUM_DEPTH, MAX_DEPTH, main_search, threefold};
use crate::util::{SetU64, isatty, STDERR};

// use std::collections::HashSet;
use std::io::{BufWriter, Write};
use std::fs::{File, OpenOptions};
use std::time::{Instant, Duration};
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Acquire, Release};

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// This constant needs to be determined empirically whenever DFLT_PARAMS is changed.
const AVG_EMIT_PER_GAME : f64 = 100.0;

// Suppose there are t threads selecting openings (and playing out games from them) and that
//   these t threads cannot communicate. How do we avoid collisions (book exits being picked by
//   more than one thread) while also having randomized and uniform coverage? Broadly speaking,
//   we simply make the space of possible book exits to pick from large enough that the chance
//   of a collision is low.
// Suppose we want to generate g book exits (i.e. play out g games) in total, or g / t games per
//   thread. Let s be a multiplier, defined so that that the number of possible book exits to
//   pick from is g × s. Then the fraction of the space g × s that each thread will explore is
//
//   p = (g / t) ÷ (g × s) = 1 / (t × s).
//
//   For any particular thread, and any particular book exit, this is the probability that the
//   thread will pick that book exit. For convenience, let's also define
//
//   q = 1 − p,
//
//   which is the probability that a particular thread does NOT pick a particular book exit.
// The probability that a particular book exit is picked by any thread is
//
//   a = 1 - q^t
//
// and the probability that a particular book exit is picked by exactly one thread is
//
//   u = t × p × q^(t−1).
//
// Then the probability that a particular book exit is picked by exactly one thread given that
//   it was picked is u / a.
// Quite happily, u/a rapidly converges to a limit as t increases!
// Taking a look at the graph [https://www.desmos.com/calculator/ubf3lhq5y9] we can see that
//   u/a is 0.95 around s = 10; in other words, for each book exit, the probability that the
//   book exit is duplicated is less than 5% when s = 10, and this is true independent of the
//   number of threads, k, and the total number of games we want to play, g. To make things
//   nice and round, we pick s = 10 whenever k > 1.

pub struct OpeningParams {
  pub num_procs     : u8,   // this is the size of the cohort (see the discussion above)
  pub total_todo    : usize,// this many pos'ns or moves will be emitted in total by the cohort
  pub rand_startpos : bool, // randomize the starting position before selecting an opening
  pub branch_target : f64,  // consider this many moves on average each ply of opening selection
}

pub struct SelfplayParams {
  // Scoring Parameters
  pub search_nodes : u32, // search positions for this many nodes
  pub viable_score : u16, // skip openings with exit scores greater than this
  pub max_score    : u16, // don't emit positions for positions with scores greater than this
  // End Conditions
  pub draw_onset : u16, // adjudicate the game as drawn if this many halfmoves have been played
  pub draw_len   : u16, //   and, for this many consecutive ply,
  pub draw_score : u16, //   the absolute score of the position has been this or less
  pub rsgn_len   : u16, // adjudicate the game as won or lost if, for this many consecutive
  pub rsgn_score : i16, //   moves, a side considers its score less than this
  // Endgame Positions
  pub include_end : bool, // disable resignation and ignore the score limit for endgame pos'ns
}

//   5 knode ~ 0.0025 sec = 400/s
//  10 knode ~ 0.005  sec = 200/s
//  20 knode ~ 0.010  sec = 100/s
//  50 knode ~ 0.025  sec =  40/s
// 100 knode ~ 0.05   sec =  20/s
// 200 knode ~ 0.10   sec =  10/s
// 500 knode ~ 0.25   sec =   4/s

pub const DFLT_PARAMS : SelfplayParams =
  SelfplayParams {
    search_nodes: 20_000,
    viable_score:  2_00,
    max_score:    10_00,
    draw_onset:   50,
    draw_len:     10,
    draw_score:   25,
    rsgn_len:      3,
    rsgn_score:  -10_00,
    include_end: true
  };

const QUIET_THRESHOLD : u16 = 5;

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
  let num_games = args.total_todo as f64 / AVG_EMIT_PER_GAME;
  let multiplier = if args.num_procs > 1 { 10.0 } else { 1.0 };
  let space_size = num_games * multiplier;
  let reqd_openings = if args.rand_startpos { space_size / 324.0 } else { space_size };

  let open_len_0 = reqd_openings.log(args.branch_target).floor() as u8;
  let brch_fct_0 = reqd_openings.powf((open_len_0 as f64).recip());

  let open_len_1 = open_len_0 + 1;
  let brch_fct_1 = reqd_openings.powf((open_len_1 as f64).recip());

  if isatty(STDERR) {
    eprintln!("  Opening length:   {open_len_0} to {open_len_1}");
    eprintln!("  Branching factor: {brch_fct_0:.1} to {brch_fct_1:.1}");
  }
  assert!(brch_fct_0 >= 3.0);
  assert!(brch_fct_1 >= 3.0);

  init_rand(seed);
  let mut out = BufWriter::new(OpenOptions::new().append(true).create(true).open(path)?);
  let mut openings = SetU64::new();
  let mut num_emitted = 0;

  // Start the watchdog
  let watchdog = std::thread::spawn(watchdog_loop);

  let params = DFLT_PARAMS;
  let mut context = Context::new();

  let num_todo = (args.total_todo as f64 / args.num_procs as f64).ceil() as usize;

  let mut opening_count = 0;

  'explore: while num_emitted < num_todo {
    let parity = opening_count % 2;
    let opening_length   = match parity { 0 => open_len_0, _ => open_len_1 };
    let branching_factor = match parity { 0 => brch_fct_0, _ => brch_fct_1 };
    context.reset();
    let mut state =
      if args.rand_startpos { State::new324(u32::rand() as usize % 324) }
      else { initial.clone_empty() };
    state.initialize_nnue();

    for /* ply */ _ in 0..opening_length {
      let metadata = state.save();
      let mut viable_moves = Vec::with_capacity(32);
      for mut mv in state.legal_moves(Everything) {
        state.apply(&mv);
        let score =
          -resolving_search(&mut state, 0, 0, PovScore::MIN, PovScore::MAX, &mut context);
        state.undo(&mv);
        state.restore(&metadata);
        if score.unsigned_abs() > params.viable_score { continue; }
        // if ply+1 >= opening_length && score.unsigned_abs() < params.dead_score { continue; }
        let mut s = (score / 2).as_i16();
        if s >  127 { s =  127; }
        if s < -127 { s = -127; }
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
    num_emitted += selfplay(&mut out, &state, &params)?;
    opening_count += 1;
  }

  // Stop the watchdog
  unsafe { SEARCH_COUNTER.store(usize::MAX, Release); }
  watchdog.thread().unpark();
  watchdog.join().expect("unable to join thread");

  return Ok((openings.len(), num_emitted));
}

// It’s generally best to train on quiet positions exclusively. There are
// many ways to define “quiet”; some of these may be equivalent depending
// on the engine. For example, a quiet position is a position...
// ‣ that, as the root of a quiescing search, has no children.
// ‣ in which the best move is not a capture.
// ‣ in which there are no viable tactics.
// ‣ whose static eval is exactly equal to result of a quiescing search
//   from the position.
// ‣ whose static eval is within a few centipawn of the result of a quiescing
//   search from the position.
// There may be added to any of these definitions the stipulation that a
// position is not quiet if the side to move is in check.

fn selfplay(
  out     : &mut BufWriter<File>,
  initial : &State,
  params  : &SelfplayParams,
) -> std::io::Result<usize>
{
  // NOTE make sure that Hash, Persist, and SyzygyPath are set to the
  //   values you want, since those values will be used during search.

  let mut num_emitted = 0;

  let mut emitted   = SetU64::new();
  let mut positions = Vec::<MiniState>::new();
  let mut leaves    = Vec::<State>::new();

  let mut state = initial.clone_truncated();
  let mut context = Context::new();

  let mut halfmoves_played : u16 = 0;
  let mut draw_run :  u16     =  0;
  let mut rsgn_run : [u16; 2] = [0; 2];
  let outcome;
  loop {
    // We first obtain an estimate to determine the initial aspiration window
    //   and, while we're at it, gather some leaves.
    increment_generation();
    let mut leaf_buffer = Vec::<State>::new();
    let static_eval = state.game_eval();
    context.reset_search();
    let q_eval = resolving_search_leaves(
      &mut state, 0, 0, PovScore::MIN, PovScore::MAX, &mut context,
      &mut leaf_buffer
    );
    if q_eval.is_checkmate() {
      outcome = match state.turn {
        White => CoOutcome::Black,
        Black => CoOutcome::White
      };
      break;
    }
    let (score, mv) = search_score(
      &mut state, &mut context, q_eval, params.search_nodes as usize
    );
    if mv.is_null() {
      // There are no legal moves but it isn’t mate, therefore it’s stalemate,
      //   three-fold repetition, or fifty moves since a zeroing move.
      if !score.is_zero() {
        eprintln!("error: move is null but score is nonzero");
        eprintln!("  {}", state.to_fen());
        eprintln!("  {}", score);
      }
      outcome = CoOutcome::Draw;
      break;
    }

    // If the score is within range and this is a quiet position, add it to the list of
    //   game positions that we'll emit (that are both scored and marked with an outcome).

    leaves.append(&mut leaf_buffer);

    let allow;
    if params.include_end {
      use crate::piece::Piece::*;
      let material_score =
          (state.boards[WhiteQueen ].count_ones() as i16 - state.boards[BlackQueen ].count_ones() as i16) * 10
        + (state.boards[WhiteRook  ].count_ones() as i16 - state.boards[BlackRook  ].count_ones() as i16) * 5
        + (state.boards[WhiteBishop].count_ones() as i16 - state.boards[BlackBishop].count_ones() as i16) * 3
        + (state.boards[WhiteKnight].count_ones() as i16 - state.boards[BlackKnight].count_ones() as i16) * 3
        + (state.boards[WhitePawn  ].count_ones() as i16 - state.boards[BlackPawn  ].count_ones() as i16);
      allow = material_score.unsigned_abs() <= 3;
    }
    else {
      allow = false;
    }
    if score.unsigned_abs() <= if allow { 100_00 } else { params.max_score }
    && (static_eval - q_eval).unsigned_abs() <= QUIET_THRESHOLD
    && !state.incheck
    {
      emitted.insert(state.key);
      if let Some(mut mini) = MiniState::from(&state) {
        mini.score = score.from(state.turn);
        positions.push(mini);
      }
    }

    draw_run = if score.unsigned_abs() > params.draw_score { 0 } else { draw_run + 1 };
    rsgn_run[state.turn] =
      if score < PovScore::new(params.rsgn_score) { rsgn_run[state.turn] + 1 } else { 0 };

    state.apply(&mv);
    context.state_history.push((state.key, mv.is_zeroing()));
    halfmoves_played += 1;

    // eprintln!(
    //   "{:3} {:7} {:+5} {:+5} {:+5}",
    //   halfmoves_played, &mv, score, q_eval, static_eval
    // );

    if state.dfz > 100 || threefold(&context.state_history) {
      outcome = CoOutcome::Draw;
      break;
    }
    if draw_run >= params.draw_len && halfmoves_played >= params.draw_onset {
      outcome = CoOutcome::Draw;
      break;
    }
    if !params.include_end && rsgn_run[!state.turn] >= params.rsgn_len {
      outcome = match state.turn { White => CoOutcome::White, Black => CoOutcome::Black };
      break;
    }
  }

  for mut mini in positions {
    mini.set_outcome(outcome);
    out.write(mini.to_quick().as_bytes())?;
    out.write(b"\n")?;
    num_emitted += 1;
  }

  for mut leaf in leaves {
    if !emitted.insert(leaf.key) { continue; }

    let static_eval = leaf.game_eval();

    context.state_history.clear();
    context.reset_search();
    let q_eval = resolving_search(
      &mut leaf, 0, 0, PovScore::MIN, PovScore::MAX, &mut context,
    );

    if (static_eval - q_eval).unsigned_abs() > QUIET_THRESHOLD { continue; }
    if leaf.incheck { continue; }

    // We don’t include stalemates.
    if leaf.legal_moves(Everything).length() == 0 { continue; }

    context.state_history.clear();
    let (score, _) = search_score(
      &mut leaf, &mut context, q_eval, params.search_nodes as usize
    );
    let allow;
    if params.include_end {
      use crate::piece::Piece::*;
      let material_score =
          (leaf.boards[WhiteQueen ].count_ones() as i16 - leaf.boards[BlackQueen ].count_ones() as i16) * 10
        + (leaf.boards[WhiteRook  ].count_ones() as i16 - leaf.boards[BlackRook  ].count_ones() as i16) * 5
        + (leaf.boards[WhiteBishop].count_ones() as i16 - leaf.boards[BlackBishop].count_ones() as i16) * 3
        + (leaf.boards[WhiteKnight].count_ones() as i16 - leaf.boards[BlackKnight].count_ones() as i16) * 3
        + (leaf.boards[WhitePawn  ].count_ones() as i16 - leaf.boards[BlackPawn  ].count_ones() as i16);
      allow = material_score.unsigned_abs() <= 3;
    }
    else {
      allow = false;
    }
    if score.unsigned_abs() > if allow { 100_00 } else { params.max_score } { continue; }
    if let Some(mut mini) = MiniState::from(&leaf) {
      mini.score = score.from(leaf.turn);
      out.write(mini.to_quick().as_bytes())?;
      out.write(b"\n")?;
      num_emitted += 1;
    }
  }

  out.flush()?;
  return Ok(num_emitted);
}

pub fn sample_syzygy(path : &str, seed : u64, samples : usize) -> std::io::Result<()>
{
  use crate::movegen::Selectivity::Everything;
  use crate::piece::Piece::*;
  use crate::syzygy::*;

  // Motivated by this position, which Expositor scores as +5 pawns.
  //
  //   7k/8/7K/6p1/8/1B5P/8/8 w - - 70 104
  //    ·  ·  ·  ·  ·  ·  ·  k
  //    ·  ·  ·  ·  ·  ·  ·  ·
  //    ·  ·  ·  ·  ·  ·  ·  K
  //    ·  ·  ·  ·  ·  ·  p  ·
  //    ·  ·  ·  ·  ·  ·  ·  ·
  //    ·  B  ·  ·  ·  ·  ·  P
  //    ·  ·  ·  ·  ·  ·  ·  ·
  //
  // The position appears favorable because the white pawn can capture
  //   and switch onto the g file (and then the promotion square is the
  //   same color as the bishop). However, you end up in a position like
  //   7k/8/7K/8/2B3P1/8/8/8 b, which is a stalemate.
  //
  // And positions like these, which she similarly does not understand.
  //   8/8/8/3b4/8/7p/7P/4k1K1 b - - 60 98
  //   8/8/8/3b4/8/7p/4k2P/6K1 b - - 64 100
  //   7k/7P/K7/7P/8/8/1p6/1B6 b - - 28 110
  //   8/8/8/7p/3b3P/6k1/8/7K b - - 73 173
  //   1b6/1P6/8/5k2/8/8/6Kp/8 b - - 51 123
  //   7k/8/4B1pK/8/7P/8/8/8 w - - 75 181
  //   7k/8/8/6KP/8/3B4/8/8 w - - 76 172
  //   7k/8/4B2K/7p/7P/8/8/8 w - - 72 117

  assert!(syzygy_enabled());
  let support = syzygy_support();

  let mut out = BufWriter::new(
    OpenOptions::new().append(true).create(true).open(path)?
  );

  init_rand(seed);

  let mut num_draw : usize = 0;
  let mut num_mate : usize = 0;

  while num_draw + num_mate < samples {
    // 5-man
    //   KBPvK  KBvKP  KBPvKP
    //   50%    12.5%  37.5%
    // 6-man
    //   KBPvK  KBvKP  KBPvKP  KBPPvKP
    //   37.5%  12.5%  37.5%   12.5%

    let mut composite = 0;
    let mut state = State {
      sides:   [   0;  2],
      boards:  [   0; 16],
      squares: [Null; 64],
      rights:      0,
      enpass:     -1,
      incheck: false,
      turn:    Black,
      dfz:         0,
      ply:         0,
      key:         0,
      s1: Vec::new(),
    };

    let wk = (u32::rand() % 64) as usize;
    composite               |= 1u64 << wk;
    state.sides[White]      |= 1u64 << wk;
    state.boards[WhiteKing] |= 1u64 << wk;
    state.squares[wk] = WhiteKing;

    let mut bk = (u32::rand() % 64) as usize;
    let illegal = composite | crate::dest::king_destinations(wk);
    while illegal & (1u64 << bk) != 0 { bk = (u32::rand() % 64) as usize; }
    composite               |= 1u64 << bk;
    state.sides[Black]      |= 1u64 << bk;
    state.boards[BlackKing] |= 1u64 << bk;
    state.squares[bk] = BlackKing;

    let mut wb = (u32::rand() % 64) as usize;
    while composite & (1u64 << wb) != 0 { wb = (u32::rand() % 64) as usize; }
    composite                 |= 1u64 << wb;
    state.sides[White]        |= 1u64 << wb;
    state.boards[WhiteBishop] |= 1u64 << wb;
    state.squares[wb] = WhiteBishop;

    composite |= 0x_ff_00_00_00_00_00_00_ff;

    let mut wp = (u32::rand() % 64) as usize;
    while composite & (1u64 << wp) != 0 { wp = (u32::rand() % 64) as usize; }
    composite               |= 1u64 << wp;
    state.sides[White]      |= 1u64 << wp;
    state.boards[WhitePawn] |= 1u64 << wp;
    state.squares[wp] = WhitePawn;

    if support >= 5 {
      let pick = u32::rand();
      if pick & 1 != 0 {
        let mut bp = (u32::rand() % 64) as usize;
        while composite & (1u64 << bp) != 0 { bp = (u32::rand() % 64) as usize; }
        composite               |= 1u64 << bp;
        state.sides[Black]      |= 1u64 << bp;
        state.boards[BlackPawn] |= 1u64 << bp;
        state.squares[bp] = BlackPawn;
        if pick & 6 != 0 {
          let mut wp = (u32::rand() % 64) as usize;
          while composite & (1u64 << wp) != 0 { wp = (u32::rand() % 64) as usize; }
          state.sides[White]      |= 1u64 << wp;
          state.boards[WhitePawn] |= 1u64 << wp;
          state.squares[wp] = WhitePawn;
        }
      }
    }

    if state.in_check(White) { continue; }
    if state.in_check(Black) { continue; }

    for _ in 0..2 {
      state.turn = !state.turn;
      let legal_moves = state.legal_moves(Everything);
      if legal_moves.early_len > 0 { continue; }
      if let Some(score) = probe_syzygy_wdl(&state, 0) {
        if score.is_zero() {
          num_draw += 1;
        }
        else {
          if num_draw > 1 {
            let ratio = num_mate as f32 / num_draw as f32;
            // target is 1:1
            if ratio > 1.125 { continue; }
            if ratio > 0.875 {
              let thresh = ((ratio - 0.875) * 1024.0) as u32;
              if u32::rand() % 256 < thresh { continue; }
            }
            num_mate += 1;
          }
        }
        let mut mini = MiniState::from(&state).unwrap();
        mini.score = score.from(state.turn);
        out.write(mini.to_quick().as_bytes())?;
        out.write(b"\n")?;
      }
    }
  }

  eprintln!("{num_draw:9}");
  eprintln!("{num_mate:9}");

  out.flush()?;
  return Ok(());
}

fn search_score(
  state   : &mut State,
  context : &mut Context,
  guess   : PovScore,
  target  : usize,
) -> (PovScore, Move)
{
  use crate::syzygy::{syzygy_support, probe_syzygy_move};

  let units = (state.sides[White] | state.sides[Black]).count_ones();
  if state.rights == 0 && syzygy_enabled() && syzygy_support() >= units {
    if let Some(pair) = probe_syzygy_move(state) { return pair; }
  }

  set_abort(false);
  unsafe {
    // This doesn't need to be an atomic increment,
    //   since no other thread writes to counter.
    let counter = SEARCH_COUNTER.load(Acquire);
    SEARCH_COUNTER.store(counter+1, Release);
  }

  context.reset_search();
  context.reset_stats();

  let mut prev_nodes_to_depth = 0;
  let mut ratios = [1.5, 1.5, 1.5, 1.5];

  let mut score = guess;
  let mut best = Move::NULL;

  let clock = Instant::now();
  for step in 1..((MAX_DEPTH+1) as u8) {
    let init_width = (score.unsigned_abs().max(140).min(500) / 4 - 25) as i16;

    let mut alpha_width = PovScore::new(init_width);
    let mut beta_width  = PovScore::new(init_width);
    let mut alpha_failures = 0;
    let mut  beta_failures = 0;
    let mut searching = true;
    let mut tentative = score;
    while searching {
      let alpha = if alpha_failures > 2 || step <= MINIMUM_DEPTH { PovScore::MIN } else { score - alpha_width };
      let  beta = if  beta_failures > 2 || step <= MINIMUM_DEPTH { PovScore::MAX } else { score +  beta_width };
      context.nominal = step;
      tentative = main_search(state, step, 0, alpha, beta, false, NodeKind::PV, context);
      if tentative.is_sentinel() {
        // GLOB.abort should already be set, since the only reason we expect a search to return
        //   an invalid score is because the watchdog set GLOB.abort and the search exited.
        assert!(abort());
        let elapsed = clock.elapsed().as_secs_f64();
        let usc = score.unsigned_abs();
        if 1_00 < usc && usc < 10_00 {
          eprintln!("warning: search timeout");
          eprintln!("  {}", state.to_fen());
          eprintln!("  step = {step}, clock = {elapsed:.1}");
          eprintln!(
            "  prev score = {}, best move = {}",
            score, best.disambiguate(state)
          );
        }
        assert!(step >= MINIMUM_DEPTH);
        return (score, best);
      }
      searching = tentative >= beta || alpha >= tentative;
      if alpha >= tentative { alpha_failures += 1; alpha_width = alpha_width * 5 / 2; }
      if tentative >= beta  {  beta_failures += 1;  beta_width =  beta_width * 5 / 2; }
    }
    score = tentative;
    if !context.pv[0].is_empty() { best = context.pv[0][0].clone(); }
    let nodes_to_depth = context.m_nodes_at_height.iter().sum::<usize>()
                       + context.r_nodes_at_height.iter().sum::<usize>();

    if step < MINIMUM_DEPTH { continue; }

    if nodes_to_depth > target { break; }

    if prev_nodes_to_depth > 0 {
      ratios = [
        nodes_to_depth as f64 / prev_nodes_to_depth as f64, ratios[0], ratios[1], ratios[2]
      ];
    }
    prev_nodes_to_depth = nodes_to_depth;
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
    let estimate = (nodes_to_depth as f64 * geom_mean) as usize;

    if estimate < target { continue; }

    // At this point we know
    //   nodes-to-depth   <    target    <    estimate
    //          |<- undershoot ->|<- overshoot ->|
    let  overshoot = estimate - target;
    let undershoot = target - nodes_to_depth;
    if overshoot > undershoot { break; }
  }
  set_abort(true);
  return (score, best);
}

pub fn approx_node_search(
  state   : &mut State,
  context : &mut Context,
  guess   : PovScore,
  target  : usize,
) -> (PovScore, Move)
{
  return search_score(state, context, guess, target);
}
