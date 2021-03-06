use crate::algebraic::*;
use crate::cache::*;
use crate::color::*;
use crate::context::*;
use crate::default::*;
use crate::import::*;
use crate::limits::*;
use crate::movegen::*;
use crate::movesel::*;
use crate::movetype::*;
use crate::nnue::*;
use crate::perft::*;
use crate::regress::*;
use crate::resolve::*;
use crate::search::*;
use crate::show::*;
use crate::test::*;
use crate::training::*;
use crate::state::*;
use crate::util::*;

use std::io::Write;
use std::time::Instant;

macro_rules! ttyeprintln {
  ($($x:expr),*) => {
    if isatty(STDERR) { eprintln!($($x),*); }
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub const cache_size_default      : usize = 67108864;  // 64 MiB ~ 4 million entries
pub const search_threads_default  : usize = 1;
pub const search_overhead_default : usize = 10;

#[cfg(    debug_assertions) ] pub const use_prev_gen_default : bool = false;
#[cfg(not(debug_assertions))] pub const use_prev_gen_default : bool = true;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub fn uci() -> std::io::Result<()>
{
  unsafe {
    SEARCH_NETWORK = DEFAULT_NETWORK;
    USE_PREV_GEN = use_prev_gen_default;
  }

  let mut root     = State::new();
  let mut history  = Vec::new();
  let mut previous = NULL_MOVE;
  let mut current  = NULL_MOVE;

  root.initialize_nnue();

  let mut search_overhead = search_overhead_default;
  let mut search_threads  = search_threads_default;

  let stdin = std::io::stdin();
  let mut buf = String::new();
  loop {
    buf.clear();
    let inp_len = stdin.read_line(&mut buf)?;
    if inp_len == 0 { break; }  // EOF
    let mut inp = buf.trim().split_ascii_whitespace();
    let cmd = match inp.next() { Some(cmd) => cmd, None => continue };
    match cmd {
      // UCI ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      "uci" => {
        println!("id name Expositor {}", VERSION);
        println!("id author Kade");
        println!("option name Hash type spin default {} min 1 max 65536", cache_size_default >> 20);
        println!("option name Threads type spin default {} min 1 max 240", search_threads_default);
        println!("option name Overhead type spin default {} min 0 max 1000", search_overhead_default);
        println!("option name Persist type check default {}", use_prev_gen_default);
        println!("uciok");
      }

      "setoption" => {
        if search_in_progress() {
          ttyeprintln!("error: search in progress");
          continue;
        }

        if inp.next().unwrap_or("") != "name" {
          ttyeprintln!("error: \"setoption\" must be followed by \"name\"");
          continue;
        }

        let mut ws = Vec::new();
        while let Some(word) = inp.next() {
          if word == "value" { break; } else { ws.push(word); }
        }
        let opt = ws.join(" ");
        let val = inp.intersperse(" ").collect::<String>();

        match opt.as_str() {
          "Hash" => {
            if let Ok(mb) = val.parse::<usize>() {
              if mb > 65536 || 1 > mb {
                ttyeprintln!("error: invalid value");
                continue;
              }
              initialize_cache(mb << 20);
            }
            else { ttyeprintln!("error: invalid or missing value"); }
          }

          "Threads" => {
            if let Ok(th) = val.parse::<usize>() {
              if th > 240 || 1 > th {
                ttyeprintln!("error: invalid value");
                continue;
              }
              search_threads = th;
            }
            else { ttyeprintln!("error: invalid or missing value"); }
          }

          "Overhead" => {
            if let Ok(oh) = val.parse::<usize>() {
              search_overhead = oh;
            }
            else { ttyeprintln!("error: invalid or missing value"); }
          }

          "Persist" => {
            match val.to_lowercase().as_str() {
              "true" | "t" | "yes" | "y" => unsafe { USE_PREV_GEN = true;  }
              "false" | "f" | "no" | "n" => unsafe { USE_PREV_GEN = false; }
              _ => { ttyeprintln!("error: invalid or missing value"); }
            }
          }

          _ => { ttyeprintln!("error: unrecognized option"); }
        }
      }

      "ucinewgame" => {
        previous = NULL_MOVE;
        current  = NULL_MOVE;
        history = Vec::new();
      }

      "isready" => {
        println!("readyok");
      }

      // Position  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      "position" => {
        while let Some(token) = inp.next() {
          match token {
            "startpos" => {
              root = State::new();
              root.initialize_nnue();
              previous = NULL_MOVE;
              current  = NULL_MOVE;
              history = Vec::new();
            }
            "fen" => {
              match State::from_fen_fields(&mut inp) {
                Ok(new) => {
                  root = new;
                  root.initialize_nnue();
                  previous = NULL_MOVE;
                  current  = NULL_MOVE;
                  history = Vec::new();
                }
                Err(msg) => {
                  ttyeprintln!("error: {}", msg);
                  break;
                }
              }
            }
            "pgn" => {
              match parse_pgn_tokens(&root, &mut inp, false) {
                Ok(movelist) => {
                  for mv in movelist {
                    root.apply(&mv);
                    history.push((root.key, mv.is_capture()));
                    previous = current;
                    current = mv;
                  }
                }
                Err(msg) => {
                  ttyeprintln!("error: {}", msg);
                  break;
                }
              }
            }
            "moves" => {
              continue;
            }
            _ => {
              match parse_universal(&root, token) {
                Ok(mv) => {
                  root.apply(&mv);
                  history.push((root.key, mv.is_capture()));
                  previous = current;
                  current = mv;
                }
                Err(msg) => {
                  ttyeprintln!("error: {}", msg);
                  break;
                }
              }
            }
          }
        }
      }

      "flip" => {
        root.turn = !root.turn;
      }

      // Search  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      "go" => {
        let mut params = SearchParams::new();
        while let Some(opt) = inp.next() {
          if opt == "infinite" {
            params.depth = Some(64);
            continue;
          }
          if let Some(arg) = inp.next() {
            match opt {
              "movetime"  => { if let Ok(n) = arg.parse::<usize>() { params.movetime  = Some(n); } }
              "wtime"     => { if let Ok(n) = arg.parse::<usize>() { params.wtime     = Some(n); } }
              "btime"     => { if let Ok(n) = arg.parse::<usize>() { params.btime     = Some(n); } }
              "winc"      => { if let Ok(n) = arg.parse::<usize>() { params.winc      = Some(n); } }
              "binc"      => { if let Ok(n) = arg.parse::<usize>() { params.binc      = Some(n); } }
              "movestogo" => { if let Ok(n) = arg.parse::<usize>() { params.movestogo = Some(n); } }
              "depth"     => { if let Ok(n) = arg.parse::<u8>()    { params.depth     = Some(n); } }
              _ => {}
            }
          }
        }
        if !isatty(STDOUT) {
          params.overhead = search_overhead;
        }
        let limits = params.calculate_limits(&root);
        if search_in_progress() {
          std::thread::sleep(std::time::Duration::from_millis(100));
          if search_in_progress() {
            ttyeprintln!("error: search in progress");
            continue;
          }
        }
        start_search(&root, &history, limits, search_threads);
      }

      "stop" => {
        if !search_in_progress() {
          std::thread::sleep(std::time::Duration::from_millis(100));
          if !search_in_progress() {
            ttyeprintln!("error: no search in progress");
            continue;
          }
        }
        loop {
          stop_search();
          std::thread::sleep(std::time::Duration::from_millis(10));
          if !search_in_progress() { break; }
        }
      }

      // Statistics  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      "reset"  => {
        if search_in_progress() {
          ttyeprintln!("error: search in progress");
          continue;
        }
        unsafe { GLOBAL_STATISTICS.reset(); }
      }

      "stat" => {
        if !isatty(STDERR) { continue; }
        if search_in_progress() { eprintln!("error: search in progress"); continue; }
        unsafe { GLOBAL_STATISTICS.print_stats(); }
      }

      "trace" => {
        if !isatty(STDERR) { continue; }
        if search_in_progress() { eprintln!("error: search in progress"); continue; }
        unsafe { GLOBAL_STATISTICS.print_trace(); }
      }

      // Tools ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      "stderr-isatty" => {
        if let Some(toggle) = inp.next() {
          match toggle.to_lowercase().as_str() {
            "true" | "t" | "yes" | "y" => unsafe { STDERR_ISATTY = Some(true);  }
            "false" | "f" | "no" | "n" => unsafe { STDERR_ISATTY = Some(false); }
            _ => {}
          }
        }
      }

      "stdout-isatty" => {
        if let Some(toggle) = inp.next() {
          match toggle.to_lowercase().as_str() {
            "true" | "t" | "yes" | "y" => unsafe { STDOUT_ISATTY = Some(true);  }
            "false" | "f" | "no" | "n" => unsafe { STDOUT_ISATTY = Some(false); }
            _ => {}
          }
        }
      }

      "clear" => {
        if isatty(STDERR) { eprint!("\x1B[H\x1B[J"); let _ = std::io::stderr().flush(); }
      }

      "show" => {
        if isatty(STDERR) { show(&root, Color::White, &hi(&previous, &current)); }
      }

      "perft" => {
        if isatty(STDERR) { run_perft(); }
      }

      "regression" => {
        if isatty(STDERR) { if let Some(path) = inp.next() { fit_dataset(path)?; } }
      }

      "resolve-debug" => {
        if !isatty(STDERR) { continue; }
        debug_resolving_search(
          &mut root, 0, 0, i16::MIN+1, i16::MAX, &mut Context::new()
        );
      }

      "resolve-leaves" => {
        if !isatty(STDERR) { continue; }
        let mut context = Context::new();
        if let Some(path) = inp.next() {
          for state in FENReader::open(path)? {
            let mut state = state?;
            state.initialize_nnue();
            resolving_search_leaves(
              &mut state, 0, 0, i16::MIN+1, i16::MAX, &mut context
            );
          }
        }
      }

      "resolve-error" => {
        if !isatty(STDERR) { continue; }

        let mut context = Context::new();
        let mut statistics = Statistics::new();
        let mut total_error = 0.0;
        let mut num_positions = 0;

        if let Some(path) = inp.next() {
          let dataset = ScoredFENReader::open(
            path,
            ScoreUnit::FractionalPawn,
            ScoreSign::WhitePositive
          )?;
          for pair in dataset {
            let (score, mut state) = pair.unwrap();
            let score = score as f32 / 100.0;
            state.initialize_nnue();
            let prediction = resolving_search(
              &mut state, 0, 0, i16::MIN+1, i16::MAX, &mut context, &mut statistics
            ) as f32 / 100.0;
            let error = compress(prediction) - compress(score);
            total_error += error * error;
            num_positions += 1;
          }

          for x in 0..20 { eprintln!("  {:2} {}", x, statistics.r_nodes_at_length[x]); }
          let nodes = statistics.r_nodes_at_height.iter().sum::<usize>();
          eprintln!("  {:8} node", nodes);
          eprintln!("  {:8.1} node/pos", nodes as f32 / num_positions as f32);
          let avg_error = total_error / num_positions as f32;
          eprintln!("  {:.6} error", avg_error);
        }
      }

      "resolve-perf" => {
        if !isatty(STDERR) { continue; }
        let mut context = Context::new();
        let mut statistics = Statistics::new();
        let mut total_duration = 0.0;
        for pos in RESOLVING_TESTS {
          eprintln!("{}", pos);
          let mut state = State::from_fen(pos).unwrap();
          state.initialize_nnue();
          let mut selector = MoveSelector::new(Selectivity::Everything, 0, NULL_MOVE);
          let metadata = state.save();
          while let Some(mv) = selector.next(&mut state, &context) {
            state.apply(&mv);
            let timer = Instant::now();
            let score = -resolving_search(
              &mut state, 0, 0, i16::MIN+1, i16::MAX,
              &mut context, &mut statistics
            );
            let duration = timer.elapsed().as_secs_f64();
            total_duration += duration;
            if duration >= 0.0001 {
              let fg = if duration >= 0.0100 { "\x1B[91m" }
                  else if duration >= 0.0010 { "\x1B[39m" }
                  else                       { "\x1B[2m"  };
              eprintln!(
                "  {:5} {:+6} {}{:9.6}\x1B[0m  \x1B[2m{}\x1B[0m",
                mv, score, fg, duration, state.to_fen()
              );
            }
            state.undo(&mv);
            state.restore(&metadata);
          }
        }
        for x in 0..24 {
          eprintln!("  {:2} {}", x, statistics.r_nodes_at_length[x]);
        }
        let nodes = statistics.r_nodes_at_height.iter().sum::<usize>();
        eprintln!("{} knodes", (nodes as f64 / 1000.0).round() as usize);
        eprintln!("{} knode/s", (nodes as f64 / 1000.0 / total_duration).round() as usize);
      }

      // NNUE  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      "eval" => {
        if  isatty(STDERR) { show_derived(&root, unsafe { &SEARCH_NETWORK }); }
        if !isatty(STDOUT) { println!("{}", (root.evaluate() * 100.0).round() as i16); }
      }

      "load" => {
        if let Some(path) = inp.next() {
          let mut path = path.to_owned();
          if !path.ends_with(".nnue") { path.push_str(".nnue"); }
          unsafe { SEARCH_NETWORK = Network::load(&path)?; }
          root.initialize_nnue();
        }
      }

      "save" => {
        if let Some(path) = inp.next() {
          let mut path = path.to_owned();
          if !path.ends_with(".rs") { path.push_str(".rs"); }
          unsafe { SEARCH_NETWORK.save_source(&path)?; }
        }
      }

      "train" => {
        if !isatty(STDERR) { continue; }

        eprint!("  Input path:    ");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let inp_path = String::from(buf.trim());

        eprint!("  Output prefix: ");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let out_prefix = String::from(buf.trim());

        eprint!("  Learning rate: \x1B[2m0.01\x1B[22m\x1B[4D");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let alpha = buf.trim().parse::<f32>().unwrap_or(0.01);

        eprint!("  Beta:          \x1B[2m0.875\x1B[22m\x1B[5D");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let beta = buf.trim().parse::<f32>().unwrap_or(0.875);

        eprint!("  Gamma:         \x1B[2m0.96875\x1B[22m\x1B[7D");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let gamma = buf.trim().parse::<f32>().unwrap_or(0.96875);

        eprint!("  Batch size:    \x1B[2m16384\x1B[22m\x1B[5D");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let batch_size = buf.trim().parse::<usize>().unwrap_or(16384);

        eprint!("  Thread count:  ");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let num_threads = buf.trim().parse::<usize>().unwrap_or(1);

        eprint!("  RNG seed:      \x1B[2m0\x1B[22m\x1B[D");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let rng_seed = buf.trim().parse::<u64>().unwrap_or(0);

        train_nnue(
          &inp_path, &out_prefix,
          alpha, beta, gamma,
          batch_size, num_threads, rng_seed
        )?;
      }

      // Program ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      "help" => {
        if isatty(STDERR) { eprintln!("{}", HELP); } else { println!("{}", HELP); }
      }

      "license" => {
        if isatty(STDERR) { eprintln!("{}", LICENSE); } else { println!("{}", LICENSE); }
      }

      "quit" | "exit" => {
        break;
      }

      _ => {
        match State::from_fen(&buf) {
          Ok(new) => {
            root = new;
            root.initialize_nnue();
            previous = NULL_MOVE;
            current  = NULL_MOVE;
            history = Vec::new();
          }
          Err(_) => match parse_pgn(&State::new(), &buf, true) {
            Ok(movelist) => {
              if movelist.is_empty() { continue; }
              root = State::new();
              root.initialize_nnue();
              previous = NULL_MOVE;
              current  = NULL_MOVE;
              history = Vec::new();
              for mv in movelist {
                root.apply(&mv);
                history.push((root.key, mv.is_capture()));
                previous = current;
                current = mv;
              }
            }
            Err(_) => { ttyeprintln!("error: unknown command"); }
          }
        }
      }

      // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    }
  }
  return Ok(());
}

const HELP : &str = "
DESCRIPTION
  Expositor is a UCI-conforming chess engine for AMD64 / Intel 64 systems. There
  are no command line options but the engine can be configured through these UCI
  options:

    setoption name Hash value <num>
      Set the size of the transposition (in MiB) to the largest power of two
      less than or equal to <num>.

    setoption name Threads value <num>
      Use <num> search threads. Performance will suffer if this is set larger
      than the number of logical cores on your machine, and depending on your
      processor, may suffer if this is set larger than the number of physical
      cores.

    setoption name Overhead value <num>
      Set the move overhead (used in time control calculations) to <num>
      milliseconds. \"Overhead\" refers to the time spent per move on I/O
      operations between the engine and your client or user interface, any
      time spent on network requests that is not corrected by the server
      (if the engine is playing online), and any other latency that uses
      time on the clock. It is important that this is not set to a value
      less than the true overhead, or the engine will have a increased
      risk of flagging.

    setoption name Persist value <bool>
      Allow the engine to reuse transposition table entries from previous
      searches. When set to false, singlythreaded searches are deterministic
      and repeatable, regardless of the state of the transposition table.
      When set to true, search results may depend upon previous searches.
      (This is achieved by tagging each table entry with a generation and
      does not incur the penalty of actually zeroing the table.) Setting
      this option to true generally increases playing strength.

  As well as some nonstandard commands:

    flip          switch side to move in the current position
    eval          print the static evaluation of the current position
    load <file>   load a set of neural network weights and begin using them

    help          prints this help message
    license       prints information about the license
    exit          alias for the UCI command \"quit\"

  These commands are also available when stderr is a terminal:

    stat        displays cumulative statistics related to move ordering
    trace       displays cumulative statistics related to main search
    reset       resets statistics

    show        displays a human readable board with the current position
    eval        displays NNUE-derived piece values and the static evaluation
    clear       clears the terminal display

  Expositor will automatically detect whether stderr and stdout are connected to
  a terminal when running on a Linux system, but assumes when running on Windows
  that neither stderr nor stdout are connected to a terminal. This can, however,
  be explicitly overridden with the following commands:

    stderr-isatty <bool>
      Inform the engine that stderr is (or is not) connected to a terminal,
      or to behave as if stderr is (or is not) connected to a terminal.

    stdout-isatty <bool>
      Inform the engine that stdout is (or is not) connected to a terminal,
      or to behave as if stdout is (or is not) connected to a terminal.

  Expositor is lenient when reading moves ??? short algebraic notation can be used
  wherever long algebraic notation is expected. The current position can also be
  set by entering FEN directly (without being prefaced by \"position fen \") or by
  entering a PGN movelist (movetext without comments or evaluation annotations).

COPYRIGHT
  Copyright 2022 Kade <expositor@fastmail.com>
  This is free software, and you are welcome to modify and redistribute it under
  certain conditions.  If users can interact with a modified version of the pro-
  gram (or a work based on the program) remotely through a computer network, you
  must provide a way for users to obtain a copy of its source; see the \"license\"
  command for more details.
";

const LICENSE : &str = "
  Expositor (chess engine)
  Copyright 2022 Kade

  This program is free software: you can redistribute it and/or modify it under
  the terms of version 3 of the GNU Affero General Public License (as published
  by the Free Software Foundation).

  This program is distributed in the hope that it will be useful, but note that
  it is distributed WITHOUT ANY WARRANTY ??? without even the implied warranty of
  merchantability or fitness for a particular purpose. See the full text of the
  license for more details.

  https://www.gnu.org/licenses
";
