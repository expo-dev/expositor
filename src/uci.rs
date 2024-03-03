use crate::algebraic::{Algebraic, parse_long, parse_short};
use crate::cache::initialize_cache;
use crate::color::Color::*;
use crate::context::Context;
use crate::datagen::{OpeningParams, generate_training_data};
use crate::global::{syzygy_enabled, searching, enable_prev_gen/*, enable_nnue*/};
use crate::limits::SearchParams;
use crate::movetype::Move;
use crate::nnue::{NETWORK, Network};
use crate::perft::run_perft;
use crate::policy::{PolicyNetwork, train_policy};
use crate::proof::{ALLOC_SIZE, ProofNode, ProofBuffer, ProofAlloc, pns};
use crate::resolve::debug_resolving_search;
use crate::score::{INVALID_SCORE, LOWEST_SCORE, HIGHEST_SCORE, format_score, format_uci_score};
use crate::search::{MAX_DEPTH, start_search, stop_search};
use crate::show::{show, showpolicy, derived};
use crate::simplex::{SimplexConfig, simplexitor};
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

macro_rules! ttyeprint {
  ($($x:expr),*) => {
    if isatty(STDERR) { eprint!($($x),*); }
  }
}
macro_rules! ttyeprintln {
  ($($x:expr),*) => {
    if isatty(STDERR) { eprintln!($($x),*); }
  }
}

// TODO use let-else! for example,
//   let Some(path) = inp.next() else { crate::nnue::print_stats(); continue };
// TODO consistently use unwrap_or_continue and unwrap_or_break macros
// macro_rules! unwrap_or_continue {
//   ($opt:expr) => { match $opt { Some(x) => x, None => continue } }
// }

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub const CACHE_SIZE_DEFAULT  : usize = 67_108_864; // 64 MiB ~ 4 million entries
const SEARCH_THREADS_DEFAULT  : usize = 1;
const SEARCH_OVERHEAD_DEFAULT : usize = 10;
const USE_PREV_GEN_DEFAULT    : bool  = true;
const PIN_THREADS_DEFAULT     : bool  = false;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

const MARGIN : u64 = 100;

pub fn uci() -> std::io::Result<()>
{
  initialize_cache(CACHE_SIZE_DEFAULT);

  enable_prev_gen(USE_PREV_GEN_DEFAULT);
  let mut search_threads  = SEARCH_THREADS_DEFAULT;
  let mut search_overhead = SEARCH_OVERHEAD_DEFAULT;
  let mut pin_threads     = PIN_THREADS_DEFAULT;

  let mut supervisor : Option<std::thread::JoinHandle<()>> = None;

  let mut root = State::new();
  let mut history = Vec::new();

  root.initialize_nnue();

  let mut policy_network = unsafe { std::mem::transmute(*crate::default::DEFAULT_POLICY) };
  let mut simplex_mode : Option<SimplexConfig> = None;

  let mut proof_alloc : Option<ProofAlloc> = None;

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
        println!("id author Kade");
        println!("option name Hash type spin default {} min 1 max 262144", CACHE_SIZE_DEFAULT >> 20);
        println!("option name Threads type spin default {} min 1 max 252", SEARCH_THREADS_DEFAULT);
        println!("option name Overhead type spin default {} min 0 max 1000", SEARCH_OVERHEAD_DEFAULT);
        println!("option name Persist type check default {}", USE_PREV_GEN_DEFAULT);
        println!("option name Pin type check default {}", PIN_THREADS_DEFAULT);
/*      println!("option name Nnue type check default true"); */
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

          "Pin" => {
            match val.to_lowercase().as_str() {
              "true" | "t" | "yes" | "y" => { pin_threads = true;  }
              "false" | "f" | "no" | "n" => { pin_threads = false; }
              _ => { ttyeprintln!("error: invalid or missing value"); }
            }
          }
/*
          "Nnue" => {
            match val.to_lowercase().as_str() {
              "true" | "t" | "yes" | "y" => { enable_nnue(true ); }
              "false" | "f" | "no" | "n" => { enable_nnue(false); }
              _ => { ttyeprintln!("error: invalid or missing value"); }
            }
          }
*/
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

      "hash" => {
        let Some(tok) = inp.next() else { continue };
        if let Ok(mb) = tok.parse::<usize>() {
          if mb < 1 || 262144 < mb { ttyeprintln!("error: invalid value"); }
          else                     { initialize_cache(mb << 20);           }
        }
      }

      "threads" => {
        let Some(tok) = inp.next() else { continue };
        if let Ok(th) = tok.parse::<usize>() {
          if th < 1 || 252 < th { ttyeprintln!("error: invalid value"); }
          else                  { search_threads = th;                  }
        }
      }

      "pin" => {
        let Some(tok) = inp.next() else { continue };
        match tok.to_lowercase().as_str() {
          "true" | "t" | "yes" | "y" => { pin_threads = true;  }
          "false" | "f" | "no" | "n" => { pin_threads = false; }
          _ => { ttyeprintln!("error: invalid value"); }
        }
      }

      "auto" => {
        let th = std::cmp::max(1, num_cores() / 2);
        search_threads = th;
        pin_threads = true;
        initialize_cache(536_870_912 * th);
        initialize_syzygy("/syzygy");
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
        root.incheck = root.in_check(root.turn);
        root.key = root.zobrist();
      }

      "key" => {
        if isatty(STDERR) { eprintln!("{:016x}", root.key); root.verify_zobrist(); }
      }

      // "invert" => {
      //   if !isatty(STDERR) { continue; }
      //   let tok = match inp.next() { Some(tok) => tok, None => continue };
      //   let key = match tok.parse::<u64>() { Ok(key) => key, Err(_) => continue };
      //   let inversion = crate::zobrist::invert(key, &root, 0);
      //   show(&inversion, White);
      // }

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
            continue;
          }
          ttyeprintln!("error: position must be 40 bytes long");
        }
        else {
          println!("{}", MiniState::from(&root).to_quick());
        }
      }

      // Search  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      "simplex" => {
        let tok = match inp.next() { Some(x) => x, None => continue };
        simplex_mode = match tok.to_lowercase().as_str() {
          "false" | "no" | "f" | "n" => None,
          "true" | "yes" | "t" | "y" =>
            if simplex_mode.is_some() { simplex_mode } else { Some(SimplexConfig::PolicyOnly) },
          "value"  => Some(SimplexConfig::ValueOnly ),
          "hybrid" => Some(SimplexConfig::Hybrid    ),
          "policy" => Some(SimplexConfig::PolicyOnly),
          _ => { ttyeprintln!("error: invalid value"); continue; }
        };
      }

      "go" => {
        let men = (root.sides[White] | root.sides[Black]).count_ones();
        let syz = syzygy_enabled() && syzygy_support() >= men;
        if root.rights == 0 && (men == 3 || syz) {
          let mut score = INVALID_SCORE;
          let mut pv = Vec::new();
          if men == 3 {
            (score, pv) = probe_tb_line(&mut root);
          }
          else if let Some(pair) = probe_syzygy_line(&mut root) {
            (score, pv) = pair;
          }
          if score != INVALID_SCORE {
            let best = if pv.is_empty() { Move::NULL } else { pv[0].clone() };
            if isatty(STDERR) {
              let rectified = match root.turn { White => score, Black => -score };
              eprint!("TB \x1B[1m{:>4}\x1B[22m", format_score(rectified));
              for mv in pv.iter() { eprint!(" {}", mv); }
              eprint!("\n");
            }
            if !isatty(STDOUT) || !isatty(STDERR) {
              print!("info depth 1 seldepth 1 nodes 1 time 0 score {}", format_uci_score(score));
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
            params.depth = Some(MAX_DEPTH as u8);
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
        if let Some(mode) = simplex_mode {
          // ↓↓↓ TEMPORARY ↓↓↓
          // showpolicy(&root, &policy_network, false);
          // ↑↑↑ TEMPORARY ↑↑↑
          let limits = params.calculate_limits(&root, 0.0);
          simplexitor(&root, &history, limits, mode, &policy_network);
          continue;
        }
        if searching() {
          std::thread::sleep(std::time::Duration::from_millis(MARGIN));
          if searching() {
            ttyeprintln!("error: search in progress");
            continue;
          }
        }
        let limits = params.calculate_limits(&root, 10.0);
        supervisor = Some(start_search(&root, &history, limits, search_threads, pin_threads));
      }

      "stop" => {
        if simplex_mode.is_some() { continue; }
        if !searching() {
          std::thread::sleep(std::time::Duration::from_millis(MARGIN));
          if !searching() {
            ttyeprintln!("error: no search in progress");
            continue;
          }
        }
        loop {
          if let Some(ref handle) = supervisor { stop_search(handle); }
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
              "false" | "no" | "f" | "n" => false,
              "true" | "yes" | "t" | "y" => true,
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
          &mut root, 0, 0, LOWEST_SCORE, HIGHEST_SCORE, &mut Context::new()
        );
      }

      "selfplay" => {
        let subcmd = match inp.next() { Some(x) => x, None => continue };
        let gen_policy_data = match subcmd {
          "moves" => true, "positions" => false, _ => continue
        };

        ttyeprint!("  Output path:      ");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let path = String::from(buf.trim());

        ttyeprint!("  Num processes:    \x1B[s\x1B[2m1\x1B[22m\x1B[u");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let num_procs = buf.trim().parse::<u8>().unwrap_or(1);
        ttyeprintln!("\x1B[A\x1B[21G{}\x1B[K", num_procs);
        if num_procs == 0 { continue; }

        ttyeprint!("  Size of dataset:  ");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let total_todo = buf.trim().parse::<usize>().unwrap_or(0);
        if total_todo == 0 { continue; }

        ttyeprint!("  Random startpos:  \x1B[s\x1B[2mno\x1B[22m\x1B[u");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let rand_startpos = match buf.to_lowercase().as_str().trim() {
          "y" | "yes" | "t" | "true" => true, _ => false
        };
        ttyeprintln!("\x1B[A\x1B[21G{}\x1B[K", if rand_startpos { "yes" } else { "no" });

        ttyeprint!("  Tgt branch fact:  ");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let branch_target = buf.trim().parse::<f64>().unwrap_or(0.0);
        if !(branch_target > 0.0) { continue; }

        // ttyeprint!("  PRNG seed:        \x1B[s\x1B[2m0\x1B[22m\x1B[u");
        ttyeprint!("  PRNG seed:        ");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let seed = buf.trim().parse::<u64>().unwrap();
        ttyeprintln!("\x1B[A\x1B[21G{}\x1B[K", seed);

        let args = OpeningParams {
          gen_moves:     gen_policy_data,
          num_procs:     num_procs,
          total_todo:    total_todo,
          rand_startpos: rand_startpos,
          branch_target: branch_target,
        };
        let (openings, emitted) = generate_training_data(&path, &args, &root, seed)?;
        ttyeprintln!("note: explored {openings} openings");
        ttyeprintln!("note: emitted {emitted} samples");
      }

      "policy" => {
        let path = match inp.next() { Some(x) => x, None => continue };
        train_policy(path)?;
      }

      "lp" => {
        let path = match inp.next() { Some(x) => x, None => continue };
        policy_network = PolicyNetwork::load(path)?;
      }

      "ls" => {
        showpolicy(&root, &policy_network, false);
      }

      "ip" => {
        let path = match inp.next() { Some(x) => x, None => continue };
        policy_network.save_image(&format!("1-{}", path), 1)?;
        policy_network.save_image(&format!("2-{}", path), 2)?;
        policy_network.save_image(&format!("3-{}", path), 3)?;
      }

      // Proof Number Search ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      /* "alloc" => {
        if !isatty(STDERR) { continue; }
        let tok = match inp.next() { Some(x) => x, None => continue };
        let mb = match tok.parse::<usize>() { Ok(sz) => sz, Err(_) => continue };
        let mut sz = mb * 1024 * 1024 / 24;
        let mut log2 = 0;
        loop { sz >>= 1; if size == 0 { break; } log2 += 1; }
        let sz : usize = 1 << log2;
      } */

      "proof" => {
        if !isatty(STDERR) { continue; }
        if proof_alloc.is_none() {
          eprint!("Allocating... ");
          let buf = Box::into_raw(vec![ProofNode::NULL; ALLOC_SIZE].into_boxed_slice());
          proof_alloc = Some(unsafe { Box::from_raw(buf as *mut ProofBuffer) });
          eprintln!("Done.");
        }
        let mut limit = 255;
        if let Some(tok) = inp.next() {
          if let Ok(p) = tok.parse::<u8>() {
            limit = if p > 127 { 255 } else { std::cmp::max(1, p) * 2 - 1 };
          }
        }
        pns(proof_alloc.as_mut().unwrap(), &root, limit);
      }

      // NNUE  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

      "eval" => {
        if  isatty(STDERR) { derived(&root); }
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
        match subcmd {
          "default" => {
            unsafe { NETWORK.save_default()?; }
          }
          "image" => {
            let path = match inp.next() { Some(x) => x, None => continue };
            unsafe {
              NETWORK.save_image(&format!("{path}-fst-indp-all"), false, -1)?;
              NETWORK.save_image(&format!("{path}-fst-unif-all"), true,  -1)?;
              for r in 0..crate::nnue::REGIONS {
                NETWORK.save_image(&format!("{path}-fst-indp-{r}"), false, r as i8)?;
                NETWORK.save_image(&format!("{path}-fst-unif-{r}"), true,  r as i8)?;
              }
            }
          }
          _ => {}
        }
      }

      "stat" => {
        let Some(path) = inp.next() else { crate::nnue::print_stats(); continue };
        let mut context = Context::new();
        use crate::import::QuickReader;
        for qk in QuickReader::open(path)? {
          let mut state = State::from(&qk?);
          state.initialize_nnue();
          crate::datagen::approx_time_search(&mut state, &mut context, 0, 0.2);
        }
        crate::nnue::print_stats();
      }

      /*
      "stat" => {
        // ↓↓↓ TEMPORARY ↓↓↓
        if let Some(path) = inp.next() {
          use crate::import::QuickReader;
          use crate::nnue::king_region;
          use crate::misc::vmirror;
          let mut region_count = [[0; 5]; 5];
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
            eprintln!();
          }
          eprintln!("Side to move");
          for rank in (0..8).rev() {
            for file in 0..8 {
              eprint!("  {:9}", sm_count[rank*8 + file]);
            }
            eprintln!();
          }
          eprintln!("Side waiting");
          for rank in (0..8).rev() {
            for file in 0..8 {
              eprint!("  {:9}", sw_count[rank*8 + file]);
            }
            eprintln!();
          }
          eprintln!("Side to move");
          for sm in 0..5 {
            let c = region_count[sm].iter().sum::<usize>();
            eprintln!("  {} {:9}", sm, c);
          }
          eprintln!("Side waiting");
          for sw in 0..5 {
            let c = region_count.iter().map(|x| x[sw]).sum::<usize>();
            eprintln!("  {} {:9}", sw, c);
          }
          eprintln!("Regions");
          for sm in 0..5 {
            eprint!("  {}", sm);
            for sw in 0..5 {
              eprint!("  {:9}", region_count[sm][sw]);
            }
            eprintln!();
          }
        }
        // ↑↑↑ TEMPORARY ↑↑↑
        else {
          unsafe { NETWORK.stat(); }
        }
      }
      */

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

        eprint!("  Learning rate: \x1B[s\x1B[2m0.015625\x1B[22m\x1B[u");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let alpha = buf.trim().parse::<f32>().unwrap_or(0.015625);
        eprintln!("\x1B[A\x1B[18G{}\x1B[K", alpha);

        eprint!("  Beta:          \x1B[s\x1B[2m0.875\x1B[22m\x1B[u");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let beta = buf.trim().parse::<f32>().unwrap_or(0.875);
        eprintln!("\x1B[A\x1B[18G{}\x1B[K", beta);

        eprint!("  Gamma:         \x1B[s\x1B[2m0.96875\x1B[22m\x1B[u");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let gamma = buf.trim().parse::<f32>().unwrap_or(0.96875);
        eprintln!("\x1B[A\x1B[18G{}\x1B[K", gamma);

        eprint!("  1/Epsilon:     \x1B[s\x1B[2m4096\x1B[22m\x1B[u");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let recip_epsilon = buf.trim().parse::<usize>().unwrap_or(4096) as f32;
        eprintln!("\x1B[A\x1B[18G{}\x1B[K", recip_epsilon);

        eprint!("  Thread count:  \x1B[s\x1B[2m1\x1B[22m\x1B[u");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let num_threads = buf.trim().parse::<usize>().unwrap_or(1);
        eprintln!("\x1B[A\x1B[18G{}\x1B[K", num_threads);

        let mut default_sz = num_threads;
        while default_sz < 16384 { default_sz *= 2; }

        eprint!("  Batch size:    \x1B[s\x1B[2m{default_sz}\x1B[22m\x1B[u");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let batch_size = buf.trim().parse::<usize>().unwrap_or(default_sz);
        eprintln!("\x1B[A\x1B[18G{}\x1B[K", batch_size);

        eprint!("  RNG seed:      \x1B[s\x1B[2m0\x1B[22m\x1B[u");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let rng_seed = buf.trim().parse::<u64>().unwrap_or(0);
        eprintln!("\x1B[A\x1B[18G{}\x1B[K", rng_seed);

        let default_epochs = (batch_size + 256) / 512;

        eprint!("  Epochs:        \x1B[s\x1B[2m{default_epochs}\x1B[22m\x1B[u");
        buf.clear();
        stdin.read_line(&mut buf)?;
        let num_epochs = buf.trim().parse::<usize>().unwrap_or(default_epochs);
        eprintln!("\x1B[A\x1B[18G{}\x1B[K", num_epochs);

        train_nnue(
          finetune,
          &inp_path, &out_prefix,
          alpha, beta, gamma, recip_epsilon,
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
        if buf == "*\n" { continue; }
        // FEN
        if buf.contains('/') {
          match State::from_fen(&buf) {
            Ok(new) => {
              root = new;
              root.initialize_nnue();
              history.clear();
            }
            Err(msg) => {
              ttyeprintln!("error: {}", msg);
            }
          }
          continue;
        }
        // Quick
        let stripped = buf.trim();
        if !stripped.contains(' ') {
          if let Ok(ary) = <&[u8; 40]>::try_from(stripped.as_bytes()) {
            let mini = MiniState::from_quick(ary);
            root = State::from(&mini);
            root.key = root.zobrist();
            root.initialize_nnue();
            history.clear();
            continue;
          }
        }
        // Move List
        let tokens = buf.split_ascii_whitespace();
        let mut working = root.clone_empty();
        let mut movelist = Vec::new();
        let mut reset = false;
        for token in tokens {
          let token = match token.split_once('.') {
            None => token,
            Some((mv_number, mv_token)) => {
              match mv_number.parse::<u16>() {
                Ok(n) =>
                  if n == 1 {
                    working = State::new();
                    working.initialize_nnue();
                    movelist.clear();
                    reset = true;
                  }
                Err(_) => {
                  ttyeprintln!("error: unknown command");
                  movelist.clear();
                  break;
                }
              }
              mv_token
            }
          };
          if token.len() == 0 { continue; }
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
        if num_moves > 0 && reset {
          root = State::new();
          root.initialize_nnue();
          history.clear();
        }
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
      Set the size of the transposition (in MiB) to <num>.

    setoption name Threads value <num>
      Use <num> search threads. Performance will suffer if this is set larger
      than the number of logical processors on your machine, and depending on
      your processor, may suffer if this is set larger than the number of
      physical cores.

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

    setoption name Pin value <bool>
      Request that search threads set their affinity mask so that they are
      pinned to different logical processors. The engine will attempt to
      determine which logical processors share each physical core by querying
      sysfs and then assign search threads to different physical cores. Once
      every physical core has a search thread assigned to it, the engine will
      assign search threads to the remaining logical processors in an attempt
      to evenly distribute search threads over the physical cores. Once every
      logical processor has a search thread assigned to it, the engine will
      stop issuing assignments, leaving any remaining threads with their
      default affinity masks. If the engine is unable to determine the
      topology, it will behave as if the Pin option were set to false,
      and placement will be left to the operating system process scheduler.
      If multiple instances of the engine are running on the same machine,
      it is strongly recommended that you set this option to false, otherwise
      the instances will contend for the same cores despite the availability
      of other cores. This feature is only implemented for Linux systems.

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
  Copyright 2024 Kade <helo@expositor.dev>
  This is free software, and you are welcome to modify and redistribute it under
  certain conditions.  If users can interact with a modified version of the pro-
  gram (or a work based on the program) remotely through a computer network, you
  must provide a way for users to obtain a copy of its source; see the \"license\"
  command for more details.
";

static LICENSE : &str = "
  Expositor (chess engine)
  Copyright 2024 Kade

  This program is free software: you can redistribute it and/or modify it under
  the terms of version 3 of the GNU Affero General Public License (as published
  by the Free Software Foundation).

  This program is distributed in the hope that it will be useful, but note that
  it is distributed WITHOUT ANY WARRANTY – without even the implied warranty of
  merchantability or fitness for a particular purpose. See the full text of the
  license for more details.

  https://www.gnu.org/licenses
";
