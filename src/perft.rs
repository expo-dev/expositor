use crate::movegen::*;
use crate::state::*;
use crate::test::*;

use std::io::Write;
use std::time::Instant;

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
  let mut early_moves = Vec::with_capacity(16);
  let mut  late_moves = Vec::with_capacity(32);
  state.generate_legal_moves(Selectivity::Everything, &mut early_moves, &mut late_moves);
  for mv in early_moves {
    state.apply(&mv);
    perft(state, depth-1, height+1);
    state.undo(&mv);
    state.restore(&metadata);
  }
  for mv in late_moves {
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
    state.initialize_nnue();

    eprintln!("  Position {}", t);
    eprint!("    Running...");
    let _ = std::io::stderr().flush();

    let correct = &PERFT_TESTS[t].1;
    let depth = correct.iter().position(|&n| n == 0).unwrap_or(12) - 2;

    reset_perft();
    let timer = Instant::now();
    perft(&mut state, depth as i8, 0);
    let duration = timer.elapsed();

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
    let nps = count as f64 / duration.as_secs_f64();
    eprintln!("    {:.0} knode/s", nps / 1000.0);

    total_nodes += count;
    total_time  += duration.as_secs_f64();
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
