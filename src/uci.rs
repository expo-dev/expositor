use crate::algebraic::{Algebraic, parse_long, parse_short};
use crate::cache::initialize_cache;
use crate::color::Color::{White, Black};
use crate::context::Context;
use crate::datagen::selfplaytest;
use crate::global::{syzygy_enabled, searching, enable_prev_gen};
use crate::limits::SearchParams;
use crate::movetype::Move;
use crate::nnue::{NETWORK, Network, Image};
use crate::perft::run_perft;
use crate::resolve::debug_resolving_search;
use crate::score::{format_score, format_uci_score};
use crate::search::{start_search, stop_search};
use crate::show::{show, derived};
use crate::state::{MiniState, State};
use crate::syzygy::*;
use crate::tablebase::probe_tb_line;
use crate::training::train_nnue;
use crate::util::{
  VERSION,
  STDOUT,
  STDERR,
  STDOUT_ISATTY,
  STDERR_ISATTY,
  isatty,
  num_cores
};

use std::convert::TryFrom;

macro_rules! ttyeprintln {
  ($($x:expr),*) => {
    if isatty(STDERR) { eprintln!($($x),*); }
  }
}

const MARGIN : u64 = 100;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub const CACHE_SIZE_DEFAULT  : usize = 67_108_864; // 64 MiB ~ 4 million entries
const SEARCH_THREADS_DEFAULT  : usize = 1;
const SEARCH_OVERHEAD_DEFAULT : usize = 10;

#[cfg(    debug_assertions) ] const USE_PREV_GEN_DEFAULT : bool = false;
#[cfg(not(debug_assertions))] const USE_PREV_GEN_DEFAULT : bool = true;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub fn uci() -> std::io::Result<()>
{
  initialize_cache(CACHE_SIZE_DEFAULT);

  enable_prev_gen(USE_PREV_GEN_DEFAULT);
  let mut search_threads  = SEARCH_THREADS_DEFAULT;
  let mut search_overhead = SEARCH_OVERHEAD_DEFAULT;

  let mut supervisor : Option<std::thread::JoinHandle<()>> = None;

  let mut root    = State::new();
  let mut history = Vec::new();

  root.initialize_nnue();

  let stdin = std::io::stdin();
  let mut buf = String::new();
  loop {
    buf.clear();
    let inp_len = stdin.read_line(&mut buf)?;
    if inp_len == 0 { break; }  // EOF
    let mut inp = buf.split_ascii_whitespace();
    let cmd = match inp.next() { Some(cmd) => cmd, None => continue };
    match cmd {
      // UCI ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      "uci" => {
        println!("id name Expositor {}", VERSION);
        println!("id author Korawend");
        println!("option name Hash type spin default {} min 1 max 262144", CACHE_SIZE_DEFAULT >> 20);
        println!("option name Threads type spin default {} min 1 max 252", SEARCH_THREADS_DEFAULT);
        println!("option name Overhead type spin default {} min 0 max 1000", SEARCH_OVERHEAD_DEFAULT);
        println!("option name Persist type check default {}", USE_PREV_GEN_DEFAULT);
        println!("option name SyzygyPath type string default <empty>");
        println!("uciok");
      }

      "setoption" => {
        if searching() {
          std::thread::sleep(std::time::Duration::from_millis(MARGIN));
          if searching() {
            ttyeprintln!("error: search in progress");
            continue;
          }
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
              if mb > 262144 || 1 > mb {
                ttyeprintln!("error: invalid value");
                continue;
              }
              initialize_cache(mb << 20);
            }
            else { ttyeprintln!("error: invalid or missing value"); }
          }

          "Threads" => {
            if let Ok(th) = val.parse::<usize>() {
              if th > 252 || 1 > th {
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
              "true" | "t" | "yes" | "y" => { enable_prev_gen(true ); }
              "false" | "f" | "no" | "n" => { enable_prev_gen(false); }
              _ => { ttyeprintln!("error: invalid or missing value"); }
            }
          }

          "SyzygyPath" => {
            if val.as_str() == "<empty>" {
              if syzygy_enabled() {
                disable_syzygy();
                ttyeprintln!("note: syzygy tablebase disabled");
              }
            }
            else if initialize_syzygy(&val) {
              ttyeprintln!("note: initialized {}-man syzygy tablebase", syzygy_support());
            }
            else {
              ttyeprintln!("note: unable to initialize syzygy tablebase");
            }
          }

          _ => { ttyeprintln!("error: unrecognized option"); }
        }
      }

      "auto" => {
        let th = std::cmp::max(num_cores()/2, 1);
        search_threads = th;
        initialize_cache(CACHE_SIZE_DEFAULT * th);
      }

      "ucinewgame" => {
        history.clear();
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
              history.clear();
            }
            "fen" => {
              match State::from_fen_fields(&mut inp) {
                Ok(new) => {
                  root = new;
                  root.initialize_nnue();
                  history.clear();
                }
                Err(msg) => {
                  ttyeprintln!("error: {}", msg);
                  break;
                }
              }
            }
            "moves" => {
              // do nothing
            }
            _ => {
              if let Ok(n) = token.parse::<usize>() {
                if n >= 324 {
                  ttyeprintln!("error: starting position must be less than 324");
                  break;
                }
                root = State::new324(n);
                root.initialize_nnue();
                history.clear();
                continue;
              }
              if let Ok(mv) = parse_long(&root, token) {
                root.apply(&mv);
                history.push((root.key, mv.is_capture()));
                continue;
              }
              if let Ok(mv) = parse_short(&root, token) {
                root.apply(&mv);
                history.push((root.key, mv.is_capture()));
                continue;
              }
              ttyeprintln!("error: unable to parse \"{}\"", token);
              break;
            }
          } // match token
        } // while let Some(token) = inp.next()
        root.truncate();
      } // "position"

      "flip" => {
        root.turn = !root.turn;
      }

      "key" => {
        if isatty(STDERR) { eprintln!("{:016x}", root.key); root.verify_zobrist(); }
      }

      "fen" => {
        println!("{}", root.to_fen());
      }

      "quick" => {
        if let Some(token) = inp.next() {
          if let Ok(ary) = <&[u8; 40]>::try_from(token.as_bytes()) {
            let mini = MiniState::from_quick(ary);
            root = State::from(&mini);
            root.key = root.zobrist();
            root.initialize_nnue();
            history.clear();
            ttyeprintln!("note: {} {}", format_score(mini.score), mini.outcome());
          }
          else {
            ttyeprintln!("error: position must be 40 bytes long");
          }
        }
        else {
          println!("{}", MiniState::from(&root).to_quick());
        }
      }

      // Search  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      "go" => {
        let men = (root.sides[White] | root.sides[Black]).count_ones();
        let syz = syzygy_enabled() && syzygy_support() >= men;
        if root.rights == 0 && (men == 3 || syz) {
          let mut score = i16::MIN;
          let mut pv = Vec::new();
          if men == 3 {
            (score, pv) = probe_tb_line(&mut root);
          }
          else if let Some(pair) = probe_syzygy_line(&mut root) {
            (score, pv) = pair;
          }
          if score != i16::MIN {
            let best = if pv.is_empty() { Move::NULL } else { pv[0].clone() };
            if isatty(STDERR) {
              let rectified = match root.turn { White => score, Black => -score };
              eprint!("TB \x1B[1m{:>4}\x1B[22m", format_score(rectified));
              for mv in pv.iter() { eprint!(" {}", mv); }
              eprint!("\n");
            }
            if !isatty(STDOUT) || !isatty(STDERR) {
              print!("info depth 64 seldepth 64 nodes 1 time 0 score {}", format_uci_score(score));
              if !pv.is_empty() {
                print!(" multipv 1 pv");
                for mv in pv { print!(" {}", mv.algebraic()); }
              }
              print!("\n");
              println!("bestmove {}", best.algebraic());
            }
            continue;
          }
        }

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
        if searching() {
          std::thread::sleep(std::time::Duration::from_millis(MARGIN));
          if searching() {
            ttyeprintln!("error: search in progress");
            continue;
          }
        }
        supervisor = Some(start_search(&root, &history, limits, search_threads));
      }

      "stop" => {
        if !searching() {
          std::thread::sleep(std::time::Duration::from_millis(MARGIN));
          if !searching() {
            ttyeprintln!("error: no search in progress");
            continue;
          }
        }
        loop {
          if let Some(ref handle) = supervisor { stop_search(&handle); }
          std::thread::sleep(std::time::Duration::from_millis(10));
          if !searching() { break; }
        }
      }

      // Tools ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      "isatty" => {
        let mut maybe = inp.next();
        let mut which = 0b_11;
        if let Some(token) = maybe {
          match token.to_lowercase().as_str() {
            "stdout" => { which = 0b_01; maybe = inp.next(); }
            "stderr" => { which = 0b_10; maybe = inp.next(); }
            _ => {}
          }
          if let Some(token) = maybe {
            let set = match token.to_lowercase().as_str() {
              "true" | "t" | "yes" | "y" => true,
              "false" | "f" | "no" | "n" => false,
              _ => { ttyeprintln!("error: invalid value"); continue; }
            };
            unsafe {
              if which & 0b_01 != 0 { STDOUT_ISATTY = Some(set); }
              if which & 0b_10 != 0 { STDERR_ISATTY = Some(set); }
            }
          }
        }
      }

      "show" => {
        if isatty(STDERR) { show(&root, White); }
      }

      "perft" => {
        if isatty(STDERR) { run_perft(); }
      }

      "resolve" => {
        if !isatty(STDERR) { continue; }
        debug_resolving_search(
          &mut root, 0, 0, i16::MIN+1, i16::MAX, &mut Context::new()
        );
      }

      "selfplay" => {
        selfplaytest("test.qk", &root)?;
      }

      // NNUE  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      "eval" => {
        if  isatty(STDERR) { derived(&root, unsafe { &NETWORK }); }
        if !isatty(STDOUT) { println!("{}", (root.evaluate() * 100.0).round() as i16); }
      }

      "load" => {
        if let Some(path) = inp.next() {
          let mut path = path.to_owned();
          if !path.ends_with(".nnue") { path.push_str(".nnue"); }
          unsafe { NETWORK = Network::load(&path)?; }
          root.initialize_nnue();
        }
      }

      "save" => {
        let subcmd = match inp.next() { Some(x) => x, None => continue };
        let path   = match inp.next() { Some(x) => x, None => continue };
        match subcmd {
          "source" => {
            let mut path = path.to_owned();
            if !path.ends_with(".rs") { path.push_str(".rs"); }
            unsafe { NETWORK.save_source(&path)?; }
          }
          "image" => {
            unsafe {
              NETWORK.save_image(&format!("{path}-fst-indp"), Image::FstIndp)?;
              NETWORK.save_image(&format!("{path}-fst-unif"), Image::FstUnif)?;
            }
          }
          _ => {}
        }
      }

      "stat" => {
        // ↓↓↓ TEMPORARY ↓↓↓
        if let Some(path) = inp.next() {
          use crate::import::QuickReader;
          use crate::nnue::king_region;
          use crate::misc::vmirror;
          let mut region_count = [[0; 4]; 4];
          let mut sm_count     = [0; 64];
          let mut sw_count     = [0; 64];
          for qk in QuickReader::open(path)? {
            let mini = qk?;
            let wk =         mini.positions[White][0] as usize ;
            let bk = vmirror(mini.positions[Black][0] as usize);
            let side_to_move = match mini.turn() { White => wk, Black => bk };
            let side_waiting = match mini.turn() { White => bk, Black => wk };
            region_count[king_region(side_to_move)][king_region(side_waiting)] += 1;
            sm_count[side_to_move] += 1;
            sw_count[side_waiting] += 1;
          }
          eprintln!("Aggregated");
          for rank in (0..8).rev() {
            for file in 0..8 {
              eprint!("  {:9}", sm_count[rank*8 + file] + sw_count[rank*8 + file]);
            }
            eprintln!("");
          }
          eprintln!("Side to move");
          for rank in (0..8).rev() {
            for file in 0..8 {
              eprint!("  {:9}", sm_count[rank*8 + file]);
            }
            eprintln!("");
          }
          eprintln!("Side waiting");
          for rank in (0..8).rev() {
            for file in 0..8 {
              eprint!("  {:9}", sw_count[rank*8 + file]);
            }
            eprintln!("");
          }
          eprintln!("Side to move");
          for sm in 0..4 {
            let c = region_count[sm].iter().sum::<usize>();
            eprintln!("  {} {:9}", sm, c);
          }
          eprintln!("Side waiting");
          for sw in 0..4 {
            let c = region_count.iter().map(|x| x[sw]).sum::<usize>();
            eprintln!("  {} {:9}", sw, c);
          }
          eprintln!("Regions");
          for sm in 0..4 {
            eprint!("  {}", sm);
            for sw in 0..4 {
              eprint!("  {:9}", region_count[sm][sw]);
            }
            eprintln!("");
          }
        }
        // ↑↑↑ TEMPORARY ↑↑↑
        // else {
        //   unsafe { NETWORK.stat(); }
        // }
      }

      "train" | "refine" => {
        if !isatty(STDERR) { continue; }

        let finetune = cmd == "refine";

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

        eprint!("  Thread count:  \x1B[2m1\x1B[22m\x1B[D");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let num_threads = buf.trim().parse::<usize>().unwrap_or(1);

        let mut default_sz = num_threads;
        while default_sz < 16384 { default_sz *= 2; }

        eprint!("  Batch size:    \x1B[2m{default_sz}\x1B[22m\x1B[5D");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let batch_size = buf.trim().parse::<usize>().unwrap_or(default_sz);

        eprint!("  RNG seed:      \x1B[2m0\x1B[22m\x1B[D");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let rng_seed = buf.trim().parse::<u64>().unwrap_or(0);

        eprint!("  Epochs:        \x1B[2m34\x1B[22m\x1B[2D");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let num_epochs = buf.trim().parse::<usize>().unwrap_or(34);

        train_nnue(
          finetune,
          &inp_path, &out_prefix,
          alpha, beta, gamma,
          batch_size, num_epochs,
          num_threads, rng_seed
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

      // Shortcuts ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      _ => {
        if let Ok(new) = State::from_fen(&buf) {
          root = new;
          root.initialize_nnue();
          history.clear();
          continue;
        }
        if let Ok(ary) = <&[u8; 40]>::try_from(cmd.as_bytes()) {
          let mini = MiniState::from_quick(ary);
          root = State::from(&mini);
          root.key = root.zobrist();
          root.initialize_nnue();
          history.clear();
          continue;
        }
        let tokens = buf.split_ascii_whitespace();
        let mut working = root.clone();
        let mut movelist = Vec::new();
        for token in tokens {
          if let Ok(mv) = parse_long(&working, token) {
            working.apply(&mv);
            movelist.push(mv);
            continue;
          }
          if let Ok(mv) = parse_short(&working, token) {
            working.apply(&mv);
            movelist.push(mv);
            continue;
          }
          ttyeprintln!("error: unknown command");
          movelist.clear();
          break;
        }
        let num_moves = movelist.len();
        for mv in movelist {
          root.apply(&mv);
          history.push((root.key, mv.is_capture()));
        }
        if num_moves > 0 { root.truncate(); }
      }

      // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    }
  }
  return Ok(());
}

pub static HELP : &str = "
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

    setoption name SyzygyPath value <path>
      Inform the engine that Syzygy tablebase files are located in the
      directory <path> and enable the use of the Syzygy tablebases if the
      files can be loaded.

  As well as some nonstandard commands:

    fen           print the current position in Forsyth-Edwards notation
    eval          print the static evaluation of the current position
    flip          switch side to move in the current position
    load <file>   load a set of neural network weights and begin using them

    help          prints this help message
    license       prints information about the license
    exit          alias for the UCI command \"quit\"

  These commands are also available when stderr is a terminal:

    show        displays a human readable board with the current position
    eval        displays NNUE-derived piece values and the static evaluation
    resolve     display the tree of a quiescing search from the current position

  Expositor will automatically detect whether stderr and stdout are connected to
  a terminal when running on a Linux system, but assumes when running on Windows
  that neither stderr nor stdout are connected to a terminal. This can, however,
  be explicitly overridden with the following commands:

    isatty stderr <bool>
      Inform the engine that stderr is (or is not) connected to a terminal,
      or to behave as if stderr is (or is not) connected to a terminal.

    isatty stdout <bool>
      Inform the engine that stdout is (or is not) connected to a terminal,
      or to behave as if stdout is (or is not) connected to a terminal.

    isatty <bool>
      Shorthand to set both isatty stderr and isatty stdout.

  Expositor is lenient when reading moves – short algebraic notation can be used
  wherever long algebraic notation is expected. The current position can also be
  set by entering FEN directly (without being prefaced by `position fen `).

COPYRIGHT
  Copyright 2022 Korawend <expositor@fastmail.com>
  This is free software, and you are welcome to modify and redistribute it under
  certain conditions.  If users can interact with a modified version of the pro-
  gram (or a work based on the program) remotely through a computer network, you
  must provide a way for users to obtain a copy of its source; see the \"license\"
  command for more details.
";

static LICENSE : &str = "
  Expositor (chess engine)
  Copyright 2022 Korawend

  This program is free software: you can redistribute it and/or modify it under
  the terms of version 3 of the GNU Affero General Public License (as published
  by the Free Software Foundation).

  This program is distributed in the hope that it will be useful, but note that
  it is distributed WITHOUT ANY WARRANTY – without even the implied warranty of
  merchantability or fitness for a particular purpose. See the full text of the
  license for more details.

  https://www.gnu.org/licenses
";
