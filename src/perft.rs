use crate::movegen::Selectivity::Everything;
// use crate::policy::{PolicyBuffer, POLICY};
use crate::state::State;

use std::io::Write;
use std::time::Instant;

const HORIZON_BULK_COUNTING : bool = false;

static mut PERFT_NODES : [usize; 16] = [0; 16];

pub fn reset_perft()
{
  for x in 0..16 { unsafe { PERFT_NODES[x] = 0; } };
}

fn perft(state : &mut State, depth : i8, height : usize)
{
  unsafe { PERFT_NODES[height] += 1; }
  if depth < 0 { return; }
  let metadata = state.save();
  let legal_moves = state.legal_moves(Everything);
  if HORIZON_BULK_COUNTING && depth == 0 {
    unsafe {
      PERFT_NODES[height+1] += legal_moves.early_len as usize + legal_moves.late_len as usize;
    }
    return;
  }
  for mv in legal_moves {
    state.apply(&mv);
    perft(state, depth-1, height+1);
    state.undo(&mv);
    state.restore(&metadata);
  }
}

pub fn run_perft()
{
  let mut total_nodes : usize = 0;
  let mut total_time  : f64 = 0.0;

  let mut mismatches = 0;
  for t in 0..PERFT_TESTS.len() {
    let mut state = State::from_fen(PERFT_TESTS[t].0).unwrap();
    // there's no need to initialize the NNUE

    eprintln!("  Position {}", t);
    eprint!("    Running...");
    let _ = std::io::stderr().flush();

    let correct = &PERFT_TESTS[t].1;
    let depth = correct.iter().position(|&n| n == 0).unwrap_or(12) - 2;

    reset_perft();
    let timer = Instant::now();
    perft(&mut state, depth as i8, 0);
    let elapsed = timer.elapsed();

    eprint!("\r\x1B[K");
    let mut count = 0;
    for x in 0..depth+2 {
      let n = unsafe { PERFT_NODES[x] };
      count += n;
      if x == 0 { continue; }
      if n == correct[x] {
        eprintln!("    {} \x1B[92m{}\x1B[39m", x, n);
      }
      else {
        eprintln!("    {} \x1B[91m{}\x1B[39m", x, n);
        mismatches += 1;
      }
    }
    eprintln!("    {} knode", count / 1000);
    let nps = count as f64 / elapsed.as_secs_f64();
    eprintln!("    {:.0} knode/s", nps / 1000.0);

    total_nodes += count;
    total_time  += elapsed.as_secs_f64();
  }
  eprintln!("  Total");
  eprintln!("    {} knode", total_nodes / 1000);
  let nps = total_nodes as f64 / total_time;
  eprintln!("    {:.0} knode/s", nps / 1000.0);
  match mismatches {
    0 => { eprintln!("  No mismatches");             }
    1 => { eprintln!( "  1 mismatch");               }
    _ => { eprintln!("  {} mismatches", mismatches); }
  }
}

/*
fn polyp(state : &mut State, buf : &mut PolicyBuffer, depth : u8) -> usize
{
  unsafe { POLICY.initialize(state, buf); }
  if depth == 0 { return 1; }
  let mut count = 1;
  let metadata = state.save();
  for mv in state.legal_moves(Everything) {
    state.apply(&mv);
    count += polyp(state, buf, depth-1);
    state.undo(&mv);
    state.restore(&metadata);
  }
  return count;
}

pub fn run_polyp()
{
  let mut total_nodes : usize = 0;
  let mut total_time  : f64 = 0.0;

  let mut buf = PolicyBuffer::zero();

  for t in 0..PERFT_TESTS.len() {
    let mut state = State::from_fen(PERFT_TESTS[t].0).unwrap();

    eprintln!("  Position {}", t);
    eprint!("    Running...");
    let _ = std::io::stderr().flush();

    let correct = &PERFT_TESTS[t].1;
    let depth = correct.iter().position(|&n| n > 4096).unwrap_or(4) - 1;

    let timer = Instant::now();
    let count = polyp(&mut state, &mut buf, depth as u8);
    let elapsed = timer.elapsed();

    eprint!("\r\x1B[K");
    eprintln!("    {} node", count);
    let nps = count as f64 / elapsed.as_secs_f64();
    eprintln!("    {:.1} node/s", nps);

    total_nodes += count;
    total_time  += elapsed.as_secs_f64();
  }
  eprintln!("  Total");
  eprintln!("    {} knode", total_nodes);
  let nps = total_nodes as f64 / total_time;
  eprintln!("    {:.1} node/s", nps);
}
*/

static PERFT_TESTS : [(&str, [usize; 12]); 58] = [
  ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"            , [1, 20,  400,   8902,  197281,   4865609, 119060324,         0,         0,         0,         0,         0]),
  ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [1, 48, 2039,  97862, 4085603, 193690690,         0,         0,         0,         0,         0,         0]),
  ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"                           , [1, 14,  191,   2812,   43238,    674624,  11030083, 178633661,         0,         0,         0,         0]),
  ("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1"    , [1,  6,  264,   9467,  422333,  15833292,         0,         0,         0,         0,         0,         0]),
  ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1"           , [1, 44, 1486,  62379, 2103487,  89941194,         0,         0,         0,         0,         0,         0]),
  ("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1"                             , [1, 24,  496,   9483,  182838,   3605103,  71179139,         0,         0,         0,         0,         0]),
  ("rnb1kbnr/pp1pp1pp/1qp2p2/8/Q1P5/N7/PP1PPPPP/1RB1KBNR b Kkq - 0 1"    , [1, 28,  741,  21395,  583456,  17251342,         0,         0,         0,         0,         0,         0]),
  ("1Q5Q/8/3p1p2/2RpkpR1/2PpppP1/2QPQPB1/8/4K3 b - - 0 1"                , [1,  3,  202,   1130,   74235,    567103,  35635077,         0,         0,         0,         0,         0]),
  ("8/PPP4k/8/8/8/8/4Kppp/8 w - - 0 1"                                   , [1, 18,  290,   5044,   89363,   1745545,  34336777,         0,         0,         0,         0,         0]),
  ("r3r1k1/1pq2pp1/2p2n2/1PNn4/2QN2b1/6P1/3RPP2/2R3KB b - - 0 1"         , [1, 56, 2594, 137198, 6391735, 323787902,         0,         0,         0,         0,         0,         0]),
  ("8/p1p1p3/8/1P1P2k1/1K2p1p1/8/3P1P1P/8 w - - 0 1"                     , [1, 15,  219,   3086,   44013,    613007,   8561152, 118590233,         0,         0,         0,         0]),
  ("r3k2r/8/8/8/3pPp2/8/8/R3K1RR b KQkq e3 0 1"                          , [1, 29,  829,  20501,  624871,  15446339,         0,         0,         0,         0,         0,         0]),
  ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"    , [1,  6,  264,   9467,  422333,  15833292,         0,         0,         0,         0,         0,         0]),
  ("8/7p/p5pb/4k3/P1pPn3/8/P5PP/1rB2RK1 b - d3 0 1"                      , [1,  5,  117,   3293,   67197,   1881089,  38633283,         0,         0,         0,         0,         0]),
  ("8/3K4/2p5/p2b2r1/5k2/8/8/1q6 b - - 0 1"                              , [1, 50,  279,  13310,   54703,   2538084,  10809689, 493407574,         0,         0,         0,         0]),
  ("rnbqkb1r/ppppp1pp/7n/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 1"       , [1, 31,  570,  17546,  351806,  11139762, 244063299,         0,         0,         0,         0,         0]),
  ("8/p7/8/1P6/K1k3p1/6P1/7P/8 w - - 0 1"                                , [1,  5,   39,    237,    2002,     14062,    120995,    966152,   8103790,  72427818,         0,         0]),
  ("r3k2r/p6p/8/B7/1pp1p3/3b4/P6P/R3K2R w KQkq - 0 1"                    , [1, 17,  341,   6666,  150072,   3186478,  77054993,         0,         0,         0,         0,         0]),
  ("8/5p2/8/2k3P1/p3K3/8/1P6/8 b - - 0 1"                                , [1,  9,   85,    795,    7658,     72120,    703851,   6627106,  64451405,         0,         0,         0]),
  ("r3k2r/pb3p2/5npp/n2p4/1p1PPB2/6P1/P2N1PBP/R3K2R w KQkq - 0 1"        , [1, 33,  946,  30962,  899715,  29179893,         0,         0,         0,         0,         0,         0]),
  ("rnbqkbnr/ppN5/3ppppp/8/B3P3/8/PPPP1PPP/R1BQK1NR b KQkq - 0 1"        , [1,  2,   70,   1454,   51654,   1171698,  42212557,         0,         0,         0,         0,         0]),
  ("k7/8/8/8/8/8/3b1P2/4K3 w - - 0 1"                                    , [1,  4,   39,    244,    2653,     17386,    202238,   1390017,  17262346, 119631160,         0,         0]),
  ("r3k2r/1bp2pP1/5n2/1P1Q4/1pPq4/5N2/1B1P2p1/R3K2R b KQkq c3 0 1"       , [1, 60, 2608, 113742, 4812099, 202902054,         0,         0,         0,         0,         0,         0]),
  ("rrrrkr1r/rr1rr3/8/8/8/8/8/6KR w k - 0 1"                             , [1,  8,  388,   4827,  266260,   2955677, 174627748,         0,         0,         0,         0,         0]),
  ("3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1"                                   , [1, 18,   92,   1670,   10138,    185429,   1134888,  20757544, 130459988,         0,         0,         0]),
  ("8/8/8/8/k1p4R/8/3P4/3K4 w - - 0 1"                                   , [1, 18,   92,   1670,   10138,    185429,   1134888,  20757544, 130459988,         0,         0,         0]),
  ("8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1"                                  , [1, 13,  102,   1266,   10276,    135655,   1015133,  14047573, 102503850,         0,         0,         0]),
  ("8/b2p2k1/8/2P5/8/4K3/8/8 b - - 0 1"                                  , [1, 13,  102,   1266,   10276,    135655,   1015133,  14047573, 102503850,         0,         0,         0]),
  ("8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1"                                 , [1, 15,  126,   1928,   13931,    206379,   1440467,  21190412, 144302151,         0,         0,         0]),
  ("8/5k2/8/2Pp4/2B5/1K6/8/8 w - d6 0 1"                                 , [1, 15,  126,   1928,   13931,    206379,   1440467,  21190412, 144302151,         0,         0,         0]),
  ("5k2/8/8/8/8/8/8/4K2R w K - 0 1"                                      , [1, 15,   66,   1198,    6399,    120330,    661072,  12762196,  73450134,         0,         0,         0]),
  ("4k2r/8/8/8/8/8/8/5K2 b k - 0 1"                                      , [1, 15,   66,   1198,    6399,    120330,    661072,  12762196,  73450134,         0,         0,         0]),
  ("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1"                                      , [1, 16,   71,   1286,    7418,    141077,    803711,  15594314,  91628014,         0,         0,         0]),
  ("r3k3/8/8/8/8/8/8/3K4 b q - 0 1"                                      , [1, 16,   71,   1286,    7418,    141077,    803711,  15594314,  91628014,         0,         0,         0]),
  ("r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1"                           , [1, 26, 1141,  27826, 1274206,  31912360,         0,         0,         0,         0,         0,         0]),
  ("r3k2r/7b/8/8/8/8/1B4BQ/R3K2R b KQkq - 0 1"                           , [1, 26, 1141,  27826, 1274206,  31912360,         0,         0,         0,         0,         0,         0]),
  ("r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1"                            , [1, 44, 1494,  50509, 1720476,  58773923,         0,         0,         0,         0,         0,         0]),
  ("r3k2r/8/5Q2/8/8/3q4/8/R3K2R w KQkq - 0 1"                            , [1, 44, 1494,  50509, 1720476,  58773923,         0,         0,         0,         0,         0,         0]),
  ("2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1"                                   , [1, 11,  133,   1442,   19174,    266199,   3821001,  60651209,         0,         0,         0,         0]),
  ("3K4/8/8/8/8/8/4p3/2k2R2 b - - 0 1"                                   , [1, 11,  133,   1442,   19174,    266199,   3821001,  60651209,         0,         0,         0,         0]),
  ("8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1"                                 , [1, 29,  165,   5160,   31961,   1004658,   6334638, 197013195,         0,         0,         0,         0]),
  ("5K2/8/1Q6/2N5/8/1p2k3/8/8 w - - 0 1"                                 , [1, 29,  165,   5160,   31961,   1004658,   6334638, 197013195,         0,         0,         0,         0]),
  ("4k3/1P6/8/8/8/8/K7/8 w - - 0 1"                                      , [1,  9,   40,    472,    2661,     38983,    217342,   3742283,  20625698, 397481663,         0,         0]),
  ("8/k7/8/8/8/8/1p6/4K3 b - - 0 1"                                      , [1,  9,   40,    472,    2661,     38983,    217342,   3742283,  20625698, 397481663,         0,         0]),
  ("8/P1k5/K7/8/8/8/8/8 w - - 0 1"                                       , [1,  6,   27,    273,    1329,     18135,     92683,   1555980,   8110830, 153850274,         0,         0]),
  ("8/8/8/8/8/k7/p1K5/8 b - - 0 1"                                       , [1,  6,   27,    273,    1329,     18135,     92683,   1555980,   8110830, 153850274,         0,         0]),
  ("K1k5/8/P7/8/8/8/8/8 w - - 0 1"                                       , [1,  2,    6,     13,      63,       382,      2217,     15453,     93446,    998319,   5966690,  85822924]),
  ("8/8/8/8/8/p7/8/k1K5 b - - 0 1"                                       , [1,  2,    6,     13,      63,       382,      2217,     15453,     93446,    998319,   5966690,  85822924]),
  ("8/k1P5/8/1K6/8/8/8/8 w - - 0 1"                                      , [1, 10,   25,    268,     926,     10857,     43261,    567584,   2518905,  37109897,         0,         0]),
  ("8/8/8/8/1k6/8/K1p5/8 b - - 0 1"                                      , [1, 10,   25,    268,     926,     10857,     43261,    567584,   2518905,  37109897,         0,         0]),
  ("8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1"                                   , [1, 37,  183,   6559,   23527,    811573,   3114998, 104644508,         0,         0,         0,         0]),
  ("8/5k2/8/5N2/5Q2/2K5/8/8 w - - 0 1"                                   , [1, 37,  183,   6559,   23527,    811573,   3114998, 104644508,         0,         0,         0,         0]),
  ("3k4/8/8/2Pp3r/2K5/8/8/8 w - d6 0 1"                                  , [1,  7,  112,    737,   12703,     84180,   1479073,   9505528, 167888358,         0,         0,         0]),
  ("1RR4K/3P4/8/8/8/8/3p4/4rr1k w - - 0 1"                               , [1, 26,  675,  18467,  521707,  14500164,         0,         0,         0,         0,         0,         0]),
  ("1RR4K/3P4/8/8/8/8/3p4/4rr1k b - - 1 1"                               , [1, 26,  628,  18327,  488720,  14474069,         0,         0,         0,         0,         0,         0]),
  ("1RR5/7K/3P4/8/8/3p4/7k/4rr2 w - - 0 1"                               , [1, 26,  633,  16637,  446703,  11559989, 310492012,         0,         0,         0,         0,         0]),
  ("1RR5/7K/3P4/8/8/3p4/7k/4rr2 b - - 1 1"                               , [1, 26,  617,  16901,  434106,  11853938,         0,         0,         0,         0,         0,         0]),
  ("rnb1kbnr/ppp1pppp/8/3p4/1P6/P2P3q/2P1PPP1/RNBQKBNR b KQkq - 0 1"     , [1, 40, 1091,  41284, 1149593,  42490376,         0,         0,         0,         0,         0,         0]),
];
