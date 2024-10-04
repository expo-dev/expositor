use crate::formats::*;
use crate::movegen::Selectivity::Everything;
use crate::movetype::Move;
use crate::piece::Piece::{WhiteKing, BlackKing};
use crate::search::twofold;
use crate::state::{State, SavedMetadata};

use std::time::Instant;

/* THIS IS HIGHLY EXPERIMENTAL AND NEEDS TO BE REWRITTEN IN δ-φ FORM */
/*   or better yet, I should switch to DFPNS with a cache            */

#[derive(Clone)]
pub struct ProofNode {
  pub mv   : Move,  // move that led to this node
  pub pn   : u32,   // proof number
  pub dn   : u32,   // disproof number
  pub next : u32,   // starting index of children
  pub succ : u8,    // number of children
  pub term : bool   // this node is known to be terminal
  // 16 bits free
}

impl ProofNode {
  pub const NULL : ProofNode = ProofNode {
    mv: Move::NULL,
    pn: 0,
    dn: 0,
    next: 0,
    succ: 0,
    term: false
  };
}

//    16 777 216 × 24 bytes =    384 MB
//    33 554 432 × 24 bytes =    768 MB
//    67 108 864 × 24 bytes =  1 536 MB
//   134 217 728 × 24 bytes =  3 072 MB
//   268 435 456 × 24 bytes =  6 144 MB
//   536 870 912 × 24 bytes = 12 288 MB
// 1 073 741 824 × 24 bytes = 24 576 MB
pub const ALLOC_SIZE : usize = 134_217_728;

pub type ProofBuffer = [ProofNode; ALLOC_SIZE];
pub type ProofAlloc = Box<ProofBuffer>;

const INF : u32 = 0x_3fff_ffff; // 1 073 741 823

fn usat_sum(a : u32, b : u32) -> u32
{
  if a == INF { return INF; }
  if b == INF { return INF; }
  if a > INF - 256 { panic!("{}", a); }
  if b > INF - 256 { panic!("{}", b); }
  let s = a + b;
  assert!(s < INF, "{}, {}", a, b);
  return s;
}

fn sat_sum(a : u32, b : i32) -> u32
{
  if a == INF { assert!(b >= 0, "{}, {}", a, b); return INF; }
  if b == INF as i32 { return INF; }
  if a                > INF - 256 { panic!("{} {}", a, b); }
  if b.unsigned_abs() > INF - 256 { panic!("{} {}", a, b); }
  let s = a as i32 + b;
  if s < 0 { return 0; }
  let s = s as u32;
  assert!(s < INF, "{}, {}", a, b);
  return s;
}

fn diff(a : u8, b : u8) -> u8 { return (a as i8 - b as i8).unsigned_abs(); }

fn king_steps(a : u8, b : u8) -> u8
{
  let a = (a / 8, a % 8);
  let b = (b / 8, b % 8);
  return std::cmp::max(diff(a.0, b.0), diff(a.1, b.1));
}

fn step_dist(a : u8, b : u8) -> u32
{
  return [0, 0, 1, 2, 3, 4, 5, 6][king_steps(a, b) as usize];
}

fn shuffle(dfz : u16) -> u32
{
  if dfz < 12 { return 4; }
  if dfz < 16 { return 6; }
  return 8;
}

fn expand(
  buf    : &mut ProofAlloc,
  free   : &mut u32,
  state  : &mut State,
  hist   : &mut Vec<(u64, bool)>,
  idx    : usize,
  height : u8,
  limit  : u8
) -> bool
{
  assert!(!buf[idx].term && buf[idx].succ == 0);

  if state.dfz > 100 || twofold(hist) {
    buf[idx].pn = INF;
    buf[idx].dn = 0;
    buf[idx].term = true;
    return true;
  }

  let metadata = state.save();
  let legal_moves = state.collect_legal_moves(Everything);
  let successors = legal_moves.len();
  assert!(successors < 256);

  let next = *free;
  if next as usize + successors > ALLOC_SIZE { return false; }

  let defending = height & 1 != 0;
  if successors == 0 {
    let win = defending && state.incheck;
    buf[idx].pn = if win { 0 } else { INF };
    buf[idx].dn = if win { INF } else { 0 };
    buf[idx].term = true;
    return true;
  }

  if height == limit || state.dfz == 100 {
    buf[idx].pn = INF;
    buf[idx].dn = 0;
    buf[idx].term = true;
    return true;
  }

  buf[idx].next = next;
  buf[idx].succ = successors as u8;

  // At even heights (OR-nodes),
  //   P(n) = min P(s) for s in succ(n)
  //   D(n) = sum D(s) for s in succ(n).
  //
  // At odd heights (AND-nodes),
  //   P(n) = sum P(s) for s in succ(n)
  //   D(n) = min D(s) for s in succ(n).

  let mut sum = 0;
  let mut min = INF;

  let mut sdx = next as usize;
  for mv in legal_moves.into_iter() {
    state.apply(&mv);

    // init_p is an estimation of the smallest number of nodes in the subtree
    //   that have to be proven in order to prove that this node is winning.
    // init_d is an estimation of the smallest number of nodes in the subtree
    //   that have to be disproven in order to prove this node is not winning.
    let mut init_p = 8;
    let mut init_d = 8;
    if mv.gives_check() {
      match defending {
        false => { init_p /= 2; /* init_d *= 2; */ }
        true  => { init_p *= 2; /* init_d /= 2; */ }
      }
    }
    if mv.gives_discovered_check() { init_p /= 2; /* init_d *= 2; */ }
    let wk = state.boards[WhiteKing].trailing_zeros() as u8;
    let bk = state.boards[BlackKing].trailing_zeros() as u8;
    init_p = (init_p * step_dist(wk, bk) ) / 4;
    // init_p = (init_p * shuffle(state.dfz)) / 4;
    if defending && mv.piece.kind() == crate::piece::Kind::King { init_p /= 2; }

    if init_p == 0 { init_p = 1; }
    if init_d == 0 { init_d = 1; }

    assert!(sdx < ALLOC_SIZE, "out of space ({})", state.to_fen());
    buf[sdx] = ProofNode {
      mv: mv.clone(),
      pn: init_p,
      dn: init_d,
      next: 0,
      succ: 0,
      term: false
    };
    sdx += 1;

    match defending {
      false => {
        min = std::cmp::min(min, init_p);
        sum = usat_sum(sum, init_d);
      }
      true => {
        sum = usat_sum(sum, init_p);
        min = std::cmp::min(min, init_d);
      }
    }

    state.undo(&mv);
    state.restore(&metadata);
  }

  match defending {
    false => { buf[idx].pn = min; buf[idx].dn = sum; }  // even height (OR-node)
    true  => { buf[idx].pn = sum; buf[idx].dn = min; }  // odd height (AND-node)
  }

  *free = sdx as u32;
  return true;
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum StepResult {
  Success,
  Terminal,
  OutOfMemory
}

fn step(
  buf   : &mut ProofAlloc,
  free  : &mut u32,
  state : &mut State,
  hist  : &mut Vec<(u64, bool)>,
  limit : u8
) -> StepResult
{
  // Step 1. Select the most proving node
  //   At even heights (OR-nodes), this is the child with the smallest proof number.
  //   At odd height (AND-nodes), this is the child with the smallest disproof number.

  let mut idx : usize = 0;
  let mut height : u8 = 0;
  let mut moves    : Vec<Move>          = Vec::new();
  let mut metadata : Vec<SavedMetadata> = Vec::new();
  let mut indices  : Vec<u32>           = Vec::new();

  loop {
    if buf[idx].term { return StepResult::Terminal; }

    let succ = buf[idx].succ as usize;
    if succ == 0 { break; }

    indices.push(idx as u32);

    let next = buf[idx].next as usize;
    let defending = height & 1 != 0;

    let mut min_num : u32 = INF;
    let mut min_ofs : usize = 0;
    for ofs in 0..succ {
      let num = match defending {
        false => buf[next+ofs].pn,  // even height (OR-node)
        true  => buf[next+ofs].dn   // odd height (AND-node)
      };
      if num < min_num {
        min_num = num;
        min_ofs = ofs;
      }
    }

    idx = next + min_ofs;
    metadata.push(state.save());
    let mv = &buf[idx].mv;
    state.apply(mv);
    hist.push((state.key, mv.is_zeroing()));
    moves.push(*mv);
    height += 1;
  }

  // Steps 2 and 3. Expand the the most proving node
  //   and evaluate the new leaves

  let prev_pn = buf[idx].pn;
  let prev_dn = buf[idx].dn;

  if !expand(buf, free, state, hist, idx, height, limit) {
    return StepResult::OutOfMemory;
  }

  if prev_pn > INF - 256 { panic!("{} {}", buf[idx].pn, prev_pn); }
  if prev_dn > INF - 256 { panic!("{} {}", buf[idx].dn, prev_dn); }

  let mut delta_pn =
    if buf[idx].pn == INF { INF as i32 } else { buf[idx].pn as i32 - prev_pn as i32 };
  let mut delta_dn =
    if buf[idx].dn == INF { INF as i32 } else { buf[idx].dn as i32 - prev_dn as i32 };

  // Step 4. Backpropagation

  //   At even heights (OR-nodes),
  //     P(n) = min P(s) for s in succ(n)
  //     D(n) = sum D(s) for s in succ(n).
  //
  //   At odd heights (AND-nodes),
  //     P(n) = sum P(s) for s in succ(n)
  //     D(n) = min D(s) for s in succ(n).

  while height > 0 {
    let mv = moves.pop().unwrap();
    let md = metadata.pop().unwrap();
    hist.pop();
    state.undo(&mv);
    state.restore(&md);

    height -= 1;
    let idx = indices.pop().unwrap() as usize;
    let defending = height & 1 != 0;
    let prev_pn = buf[idx].pn;
    let prev_dn = buf[idx].dn;
    match defending {
      false => { buf[idx].dn = sat_sum(buf[idx].dn, delta_dn); }  // even height (OR-node)
      true  => { buf[idx].pn = sat_sum(buf[idx].pn, delta_pn); }  // odd height (AND-node)
    }

    let next = buf[idx].next as usize;
    let succ = buf[idx].succ as usize;
    let mut min_num : u32 = INF;
    for ofs in 0..succ {
      let num = match defending {
        false => buf[next+ofs].pn,  // even height (OR-node)
        true  => buf[next+ofs].dn   // odd height (AND-node)
      };
      min_num = std::cmp::min(min_num, num);
    }
    match defending {
      false => { buf[idx].pn = min_num; }
      true  => { buf[idx].dn = min_num; }
    }

    if prev_pn > INF - 256 { panic!("{} {}", buf[idx].pn, prev_pn); }
    if prev_dn > INF - 256 { panic!("{} {}", buf[idx].dn, prev_dn); }

    delta_pn = if buf[idx].pn == INF { INF as i32 } else { buf[idx].pn as i32 - prev_pn as i32 };
    delta_dn = if buf[idx].dn == INF { INF as i32 } else { buf[idx].dn as i32 - prev_dn as i32 };
  }
  return StepResult::Success;
}

/* fn set_horizon(
  buf    : &mut ProofAlloc,
  path   : &mut Vec<usize>,
  idx    : usize,
  height : u8,
  limit  : u8,
)
{
  if height == limit {
    let idx = idx as usize;

    buf[idx].pn = INF;
    buf[idx].dn = 0;
    buf[idx].term = true;

    let mut height = height;
    while height > 0 {
      height -= 1;
      let odd = height & 1 != 0;
      let idx = path[height as usize];
      let next = buf[idx].next as usize;
      let succ = buf[idx].succ as usize;
      let mut min : u32 = INF;
      let mut sum : u32 = 0;
      for ofs in 0..succ {
        min = std::cmp::min(min, if odd { buf[next+ofs].dn } else { buf[next+ofs].pn });
        sum =      usat_sum(sum, if odd { buf[next+ofs].pn } else { buf[next+ofs].dn });
      }
      match odd {
        false => { buf[idx].pn = min; buf[idx].dn = sum; }
        true  => { buf[idx].dn = min; buf[idx].pn = sum; }
      }
    }
    return;
  }

  if buf[idx].pn > 0 { return; }
  if buf[idx].term   { return; }

  let succ = buf[idx].succ as usize;
  let next = buf[idx].next as usize;
  if succ == 0 { return; }
  path.push(idx);
  for ofs in 0..succ {
    set_horizon(buf, path, next+ofs, height+1, limit);
  }
  path.pop();
} */

fn find_pv(
  buf    : &ProofAlloc,
  idx    : usize,
  height : u16
) -> (u16, Vec<u8>)
{
  if buf[idx].term { return (height, Vec::new()); }

  let succ = buf[idx].succ as usize;
  let next = buf[idx].next as usize;
  if succ == 0 { panic!(); }

  let mut ret : Option<(u16, Vec<u8>)> = None;

  let defending = height & 1 != 0;
  for ofs in 0..succ {
    if buf[next+ofs].pn > 0 { continue; }
    if defending {
      let (len, mut pv) = find_pv(buf, next+ofs, height+1);
      let replace = match ret.as_ref() {
        None => true, Some(prev) => len > prev.0
      };
      if replace {
        pv.insert(0, ofs as u8);
        ret = Some((len, pv));
      }
    }
    else {
      if buf[next+ofs].term { return (height+1, vec![ofs as u8]); }
      if ret.is_some() { panic!(); }
      let (len, mut pv) = find_pv(buf, next+ofs, height+1);
      pv.insert(0, ofs as u8);
      ret = Some((len, pv));
    }
  }
  return match ret {
    Some(ret) => ret,
    None => panic!()
  };
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum PNSResult {
  Win,
  NonWin,
  Unknown
}

/* pub fn pns(alloc : &mut ProofAlloc, state : &State)
{
  eprint!("Searching... ");
  let mut state = state.clone_empty();
  let mut hist = vec![(state.key, true)];
  // let buf = Box::into_raw(vec![ProofNode::NULL; ALLOC_SIZE].into_boxed_slice());
  // let mut alloc = unsafe { Box::from_raw(buf as *mut ProofBuffer) };
  let mut free : u32 = 1;
  alloc[0] = ProofNode {
    mv: Move::NULL,
    pn: 1,
    dn: 1,
    next: 0,
    succ: 0,
    term: false
  };
  let clock = Instant::now();
  let pns_result = loop {
    let step_result = step(alloc, &mut free, &mut state, &mut hist, 255);
    match step_result {
      StepResult::Success     => {}
      StepResult::Terminal    => { panic!(); }
      StepResult::OutOfMemory => { break PNSResult::Unknown; }
    }
    if alloc[0].pn == 0 { break PNSResult::Win; }
    if alloc[0].dn == 0 { break PNSResult::NonWin; }
  };
  let elapsed = clock.elapsed().as_secs_f64();

  match pns_result {
    PNSResult::Win     => { eprintln!("Proven."); }
    PNSResult::NonWin  => { eprintln!("Disproven."); }
    PNSResult::Unknown => { eprintln!("Out of memory."); }
  }

  eprintln!(
    "  {}node \x1B[2m= {}b\x1B[22m",
    variable_format_node(free as usize),
    variable_format_node(free as usize * 24)
  );
  eprintln!(
    "  {} {}",
    variable_format_time(elapsed), if elapsed < 60.0 { "seconds" } else { "elapsed" }
  );

  if pns_result != PNSResult::Win { return; }

  let (len, proof) = find_pv(alloc, 0, 0);
  eprint!("\x1B[1m#+{}\x1B[22m", (len + 1) / 2);
  let mut scratch = state.clone_empty();
  let mut idx = 0;
  for h in 0.. {
    if h == 7 || h == proof.len() { eprint!(" ..."); break; }
    idx = alloc[idx].next as usize + proof[h] as usize;
    let mv = &alloc[idx].mv;
    eprint!(" {}", mv.in_context(&scratch));
    scratch.apply(mv);
    if alloc[idx].term { break; }
  }
  eprint!("\n");

  eprint!("Minimizing...     ");
  let mut total_nodes = 0;
  let clock = Instant::now();
  let mut success = false;
  for limit in 1..=255 {
    eprint!("\x1B[4D{limit:3} ");
    free = 1;
    alloc[0] = ProofNode {
      mv: Move::NULL,
      pn: 1,
      dn: 1,
      next: 0,
      succ: 0,
      term: false
    };
    let pns_result = loop {
      let step_result = step(alloc, &mut free, &mut state, &mut hist, limit);
      if alloc[0].pn == 0 { break PNSResult::Win; }
      if alloc[0].dn == 0 { break PNSResult::NonWin; }
      match step_result {
        StepResult::Success     => {}
        StepResult::Terminal    => { panic!(); }
        StepResult::OutOfMemory => { break PNSResult::Unknown; }
      }
    };
    total_nodes += free as usize;
    match pns_result {
      PNSResult::Win     => { eprintln!("\x1B[4DDone."); success = true; break; }
      PNSResult::NonWin  => { }
      PNSResult::Unknown => { eprintln!("\x1B[4DOut of memory."); break;}
    }
  }
  let elapsed = clock.elapsed().as_secs_f64();

  eprintln!("  {}node", variable_format_node(total_nodes));
  eprintln!(
    "  {} {}",
    variable_format_time(elapsed), if elapsed < 60.0 { "seconds" } else { "elapsed" }
  );

  if !success { return; }

  let (len, pv) = find_pv(alloc, 0, 0);
  if pv.len() > 0 {
    eprint!("\x1B[1m#+{}\x1B[22m", (len + 1) / 2);
    let mut idx = 0;
    for ofs in pv {
      idx = alloc[idx].next as usize + ofs as usize;
      let mv = &alloc[idx].mv;
      eprint!(" {}", mv.in_context(&state));
      state.apply(mv);
    }
    eprint!("\n");
  }
} */

pub fn pns(alloc : &mut ProofAlloc, state : &State, limit : u8)
{
  eprint!("Searching... ");
  let mut state = state.clone_empty();
  let mut hist = vec![(state.key, true)];
  let mut free : u32 = 1;
  alloc[0] = ProofNode {
    mv: Move::NULL,
    pn: 1,
    dn: 1,
    next: 0,
    succ: 0,
    term: false
  };
  let clock = Instant::now();
  let pns_result = loop {
    let step_result = step(alloc, &mut free, &mut state, &mut hist, limit);
    match step_result {
      StepResult::Success     => {}
      StepResult::Terminal    => { panic!(); }
      StepResult::OutOfMemory => { break PNSResult::Unknown; }
    }
    if alloc[0].pn == 0 { break PNSResult::Win; }
    if alloc[0].dn == 0 { break PNSResult::NonWin; }
  };
  let elapsed = clock.elapsed().as_secs_f64();

  match pns_result {
    PNSResult::Win     => { eprintln!("Proven."); }
    PNSResult::NonWin  => { eprintln!("Disproven."); }
    PNSResult::Unknown => { eprintln!("Out of memory."); }
  }

  eprintln!(
    "  {}node \x1B[2m= {}b\x1B[22m",
    variable_format_node(free as usize),
    variable_format_node(free as usize * 24)
  );
  eprintln!(
    "  {} {}",
    variable_format_time(elapsed),
    if elapsed < 60.0 { "seconds" } else { "elapsed" }
  );

  if pns_result != PNSResult::Win { return; }

  let (len, proof) = find_pv(alloc, 0, 0);
  eprint!("\x1B[1m#+{}\x1B[22m", (len + 1) / 2);
  let mut scratch = state.clone_empty();
  let mut idx = 0;
  for h in 0.. {
    if h == 7 || h == proof.len() { eprint!(" ..."); break; }
    idx = alloc[idx].next as usize + proof[h] as usize;
    let mv = &alloc[idx].mv;
    eprint!(" {}", mv.disambiguate(&scratch));
    scratch.apply(mv);
    if alloc[idx].term { break; }
  }
  eprint!("\n");
}
