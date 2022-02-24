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

pub fn uci() -> std::io::Result<()>
{
  unsafe { SEARCH_NETWORK = DEFAULT_NETWORK; }

  let mut root     = State::new();
  let mut history  = Vec::new();
  let mut previous = NULL_MOVE;
  let mut current  = NULL_MOVE;

  root.initialize_nnue();

  let mut search_overhead = 100;
  let mut search_threads = 1;

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
        println!("option name EvalFile type string");
        println!("option name Hash type spin default 64 min 1 max 32768");
        println!("option name Threads type spin default 1 min 1 max 240");
        println!("option name Move Overhead type spin default 100 min 0 max 1000");
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
          "EvalFile" => {
            if val.len() > 0 {
              let mut path = val;
              if !path.ends_with(".nnue") { path.push_str(".nnue"); }
              unsafe { SEARCH_NETWORK = Network::load(&path)?; }
              root.initialize_nnue();
            }
          }

          "Hash" => {
            if let Ok(mb) = val.parse::<usize>() {
              if mb > 32768 || 1 > mb {
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

          "Move Overhead" => {
            if let Ok(oh) = val.parse::<usize>() {
              search_overhead = oh;
            }
            else { ttyeprintln!("error: invalid or missing value"); }
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
        let limits = params.limits(&root);
        if search_in_progress() {
          ttyeprintln!("error: search in progress");
          continue;
        }
        start_search(&root, &history, limits, search_threads);
      }

      "stop" => {
        if !search_in_progress() {
          ttyeprintln!("error: no search in progress");
          continue;
        }
        stop_search();
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
        if isatty(STDERR) { fit_dataset(inp.next().unwrap())?; }
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
        for state in FENReader::open(inp.next().unwrap())? {
          let mut state = state?;
          state.initialize_nnue();
          resolving_search_leaves(
            &mut state, 0, 0, i16::MIN+1, i16::MAX, &mut context
          );
        }
      }

      "resolve-error" => {
        if !isatty(STDERR) { continue; }

        let mut context = Context::new();
        let mut statistics = Statistics::new();
        let mut total_error = 0.0;
        let mut num_positions = 0;

        let dataset = ScoredFENReader::open(
          inp.next().unwrap(),
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
        if isatty(STDERR) { show_derived(&root, unsafe { &SEARCH_NETWORK }); }
        else { println!("{}", (root.evaluate() * 100.0).round() as i16); }
      }

      "load" => {
        let mut path = inp.next().unwrap().to_owned();
        if !path.ends_with(".nnue") { path.push_str(".nnue"); }
        unsafe { SEARCH_NETWORK = Network::load(&path)?; }
        root.initialize_nnue();
      }

      "save" => {
        let mut path = inp.next().unwrap().to_owned();
        if !path.ends_with(".rs") { path.push_str(".rs"); }
        unsafe { SEARCH_NETWORK.save_source(&path)?; }
      }

      "train" => {
        if !isatty(STDERR) { continue; }

        eprint!("  Input path:    ");
        buf.clear();
        stdin.read_line(&mut buf).unwrap();
        let inp_path = String::from(buf.trim());

        eprint!("  Output prefix: ");
        buf.clear();
        stdin.read_line(&mut buf).unwrap();
        let out_prefix = String::from(buf.trim());

        eprint!("  Learning rate: \x1B[2m0.01\x1B[22m\x1B[4D");
        buf.clear();
        stdin.read_line(&mut buf).unwrap();
        let alpha = buf.trim().parse::<f32>().unwrap_or(0.01);

        eprint!("  Beta:          \x1B[2m0.875\x1B[22m\x1B[5D");
        buf.clear();
        stdin.read_line(&mut buf).unwrap();
        let beta = buf.trim().parse::<f32>().unwrap_or(0.875);

        eprint!("  Gamma:         \x1B[2m0.96875\x1B[22m\x1B[7D");
        buf.clear();
        stdin.read_line(&mut buf).unwrap();
        let gamma = buf.trim().parse::<f32>().unwrap_or(0.96875);

        eprint!("  Batch size:    \x1B[2m16384\x1B[22m\x1B[5D");
        buf.clear();
        stdin.read_line(&mut buf).unwrap();
        let batch_size = buf.trim().parse::<usize>().unwrap_or(16384);

        eprint!("  Thread count:  ");
        buf.clear();
        stdin.read_line(&mut buf).unwrap();
        let num_threads = buf.trim().parse::<usize>().unwrap();

        eprint!("  RNG seed:      \x1B[2m0\x1B[22m\x1B[D");
        buf.clear();
        stdin.read_line(&mut buf).unwrap();
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
EXPOSITOR

DESCRIPTION
  Expositor is a UCI-conforming chess engine for AMD64 / Intel 64 systems. There
  are no command line options but the engine supports some nonstandard commands:

    flip        switches side to move in the current position
    eval        prints the static evaluation of the current position
    load ...    alias for the command \"setoption name EvalFile value ...\"

    help        prints this message
    license     prints information about the license
    exit        alias for the command \"quit\"

  On Linux systems, these commands are also available when stderr is a terminal:

    stat        displays cumulative statistics related to move ordering
    trace       displays cumulative statistics related to main search
    reset       resets statistics

    show        displays a human readable board with the current position
    eval        displays NNUE-derived piece values and the static evaluation
    clear       clears the terminal display

  Expositor is lenient when reading moves – short algebraic notation can be used
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
  it is distributed WITHOUT ANY WARRANTY – without even the implied warranty of
  merchantability or fitness for a particular purpose. See the full text of the
  license for more details.

  https://www.gnu.org/licenses
";
