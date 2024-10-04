use crate::color::Color::{self, *};
// use crate::misc::vmirror;
// use crate::movegen::Selectivity::Everything;
// use crate::nnue::NETWORK;
use crate::piece::Kind::King;
use crate::piece::Piece;
// use crate::policy::{PolicyBuffer, POLICY};
use crate::state::State;

pub fn show(state : &State, perspective : Color)
{
  let icon = ['K', 'Q', 'R', 'B', 'N', '●'];
  eprint!("\x1B[0m");
  for rank in (0..8).rev() {
    for file in 0..8 {
      let square = rank*8 + file;
      let square = if perspective == Black { 63 - square } else { square };
      let parity = (rank + file) % 2 == 0;

      let mut piece = Some(Piece::Null);  // None indicates an error
      for x in 0..16 {
        if state.boards[x] & (1 << square) != 0 {
          if piece.unwrap() != Piece::Null {
            piece = None;
            break;
          }
          piece = Some(Piece::from(x as u8));
        }
      }
      if let Some(p) = piece {
        if state.squares[square] != p { piece = None; }
      }

      let bg = if parity { (181, 135, 99) } else { (212, 190, 154) };
      eprint!("\x1B[48;2;{};{};{}m", bg.0, bg.1, bg.2);

      if let Some(p) = piece {
        if p.is_null() {
          eprint!("   ");
        }
        else {
          match p.color() {
            White => { eprint!("\x1B[97m"); }
            Black => { eprint!("\x1B[30m"); }
          }
          eprint!(" {} ", icon[p.kind()]);
        }
      }
      else {
        eprint!("\x1B[97;101m");
        eprint!(" ? ");
      }
    }
    eprintln!("\x1B[0m");
  }
  eprintln!("\x1B[2m{} to move  {:5} dfz\x1B[22m", state.turn, state.dfz);
}

pub fn derived(state : &State, which : usize)
{
  use crate::nnext::{NNETWORK, FNETWORK};
  let mut state = state.clone_empty();
  let base = match which {
    0 => unsafe { NNETWORK.evaluate(&state) },
    _ => unsafe { FNETWORK.evaluate(&state) },
  };
  let base = match state.turn { White => base, Black => -base };

  eprint!("\x1B[0m");
  for _ in 0..24 { eprintln!(); }
  eprint!("\x1B[24A");
  for rank in (0..8).rev() {
    for file in 0..8 {
      let square = rank*8 + file;
      let piece = state.squares[square];
      let parity = (rank + file) % 2 == 0;
      let bg = if parity { (181, 135, 99) } else { (212, 190, 154) };
      eprint!("\x1B[48;2;{};{};{}m", bg.0, bg.1, bg.2);
      eprint!("       \x1B[7D\x1B[B");
      if piece.is_null() {
        eprint!("       \x1B[7D\x1B[B");
        eprint!("       \x1B[7D");
      }
      else {
        match piece.color() {
          White => eprint!("\x1B[97m"),
          Black => eprint!("\x1B[30m"),
        }
        eprint!("   {}   \x1B[7D\x1B[B", piece.kind().upper());
        if piece.kind() == King {
          eprint!("       \x1B[7D");
        }
        else {
          state.boards[piece] ^= 1 << square;
          state.sides[piece.color()] ^= 1 << square;
          let hyp = match which {
            0 => unsafe { NNETWORK.evaluate(&state) },
            _ => unsafe { FNETWORK.evaluate(&state) },
          };
          let diff = match state.turn {
            White => base - hyp,
            Black => base + hyp
          };
          state.boards[piece] ^= 1 << square;
          state.sides[piece.color()] ^= 1 << square;
          if diff.abs() >= 10.0 {
            eprint!(" {:+5.1} \x1B[7D", diff);
          }
          else {
            eprint!(" {:+5.2} \x1B[7D", diff);
          }
        }
      }
      eprint!("\x1B[2A\x1B[7C");
    }
    eprint!("\x1B[0m\r\x1B[3B");
  }
  eprintln!("NNUE evaluation {:+7.3}\x1B[44G{} to move", base, state.turn);
  // eprint!("\x1B[2m");
  // for h in 0..HEADS {
  //   eprint!("hd {} evaluation {:+7.3}", h, unsafe { NETWORK.evaluate(&state, h) });
  //   if h == head_idx { eprintln!(" <-"); } else { eprintln!(); }
  // }
  // eprint!("\x1B[0m");
}

/*
pub fn showpolicy(state : &State, quiet_only : bool)
{
  let mut buf = PolicyBuffer::zero();
  unsafe { POLICY.initialize(state, &mut buf); }

  let mut scores = [f32::INFINITY; 384];
  let mut best = [(f32::INFINITY, crate::movetype::Move::NULL); 20];
  let legal_moves = state.collect_legal_moves(Everything);
  for m in legal_moves.into_iter() {
    if quiet_only && m.is_capture() { continue; }
    let src = m.src as usize;
    let src = match state.turn { White => src, Black => vmirror(src) };
    let dst = m.dst as usize;
    let dst = match state.turn { White => dst, Black => vmirror(dst) };
    let s = unsafe { POLICY.evaluate(&buf, m.piece.kind(), src, dst) };
    let idx = (m.piece.kind() as usize)*64 + m.dst as usize;
    let z = scores[idx];
    if z.is_infinite() || s < z { scores[idx] = s; }
    for x in 0..20 {
      if s < best[x].0 {
        for y in (x+1..20).rev() { best[y] = best[y-1]; }
        best[x] = (s, m);
        break;
      }
    }
  }

  eprint!("\x1B[0m");
  for row in 0..2 {
    for rank in (0..8).rev() {
      for column in 0..3 {
        let kind = row*3 + column;
        for file in 0..8 {
          let square = rank*8 + file;
          let idx = kind*64 + square;
          let parity = (rank + file) % 2 == 0;
          // let bg = if parity { (121,  93, 70) } else { (151, 122,  98) };
          // let wc = if parity { (136, 107, 84) } else { (167, 136, 113) };
          // let bc = if parity { (106,  79, 57) } else { (136, 107,  84) };
          let bg = if parity { (22, 22, 22) } else { (34, 34, 34) };
          let wc = if parity { (34, 34, 34) } else { (46, 46, 46) };
          let bc = if parity { (11, 11, 11) } else { (22, 22, 22) };
          eprint!("\x1B[48;2;{};{};{}m", bg.0, bg.1, bg.2);
          let s = scores[idx];
          let p = state.squares[square];
          if s.is_infinite() {
            if p.is_null() {
              eprint!("   ");
            }
            else {
              let icon = ['K', 'Q', 'R', 'B', 'N', 'P'];
              let fg = match p.color() { White => wc, Black => bc };
              eprint!("\x1B[38;2;{};{};{}m", fg.0, fg.1, fg.2);
              eprint!(" {} ", icon[p.kind()]);
            }
          }
          else {
            // let a = (1.0 + s).recip();
            // let b = 1.0 - a;
            // let fg = match state.turn { White => (255, 255, 255), Black => (0, 0, 0) };
            // let red = (a * fg.0 as f32 + b * bg.0 as f32).round() as u8;
            // let grn = (a * fg.1 as f32 + b * bg.1 as f32).round() as u8;
            // let blu = (a * fg.2 as f32 + b * bg.2 as f32).round() as u8;
            let wow =  (  0, 255, 128);
            let okay = (  0,  96, 255);
            let bad  = (192,   0,   0);
            let a; let f; let g;
            let x = s - 0.5;
            if x >= 0.0 {
              a = (1.0 + x).recip();
              f = okay; g = bad;
            }
            else {
              a = (1.0 - x*4.0).recip();
              f = okay; g = wow;
            }
            let b = 1.0 - a;
            let red = (a * f.0 as f32 + b * g.0 as f32).round() as u8;
            let grn = (a * f.1 as f32 + b * g.1 as f32).round() as u8;
            let blu = (a * f.2 as f32 + b * g.2 as f32).round() as u8;
            eprint!("\x1B[38;2;{red};{grn};{blu}m");
            if p.is_null() {
              eprint!(" ● ");
            }
            else {
              let icon = ['K', 'Q', 'R', 'B', 'N', 'P'];
              eprint!(" {} ", icon[p.kind()]);
            }
          }
        }
        if column != 2 { eprint!("\x1B[49m  "); }
      }
      eprintln!("\x1B[0m");
    }
    // if row != 1 { eprint!("\n"); }
    eprint!("\n");
  }
  for r in 0..4 {
    for c in 0..5 {
      let (s, m) = &best[c*4 + r];
      if c == 0 { eprint!("\x1B[2m"); } else { eprint!("   "); }
      if *s > 100.0 {
        eprint!("  ..... ...  ");
      }
      else {
        eprint!("{:+7.2} {:5}", s, m.disambiguate(state));
      }
    }
    eprint!("\x1B[22m\n");
  }
}
*/
