use crate::color::*;
use crate::movetype::*;
use crate::nnue::*;
use crate::piece::*;
use crate::state::*;

pub struct ShowParams {
  pub previous : (i8, i8),
  pub current : (i8, i8),
  pub good   : (i8, i8),
  pub bad   : (i8, i8),
}

pub const NOHILT : ShowParams =
  ShowParams {
    previous: (-1, -1),
    current: (-1, -1),
    good:   (-1, -1),
    bad:   (-1, -1),
  };

pub fn highlight(prev : &Move, curr : &Move) -> ShowParams
{
  return ShowParams {
    previous: if prev.is_null() { (-1, -1) } else { (prev.src, prev.dst) },
    current: if curr.is_null() { (-1, -1) } else { (curr.src, curr.dst) },
    good:   (-1, -1),
    bad:   (-1, -1),
  };
}

pub fn hi(prev : &Move, curr : &Move) -> ShowParams { return highlight(prev, curr); }

pub fn show(state : &State, perspective : Color, params : &ShowParams)
{
  eprint!("\x1B[0m");
  for rank in (0..8).rev() {
    for file in 0..8 {
      let square = rank*8 + file;
      let square = if perspective == Color::Black { 63 - square } else { square };
      let parity = (rank + file) % 2 == 0;

      let mut p = -1;
      for x in 0..16 {
        if (state.boards[x as usize] >> square) & 1 != 0 {
          p = if p < 0 { x as i8 } else { 16 };
        }
      }
      if p < 16 {
        if state.squares[square] != Piece::from_i8(p) { p = 16; }
      }

      let square = square as i8;

      let previous = square == params.previous.0 || square == params.previous.1;
      let current  = square == params.current.0  || square == params.current.1;
      let good     = square == params.good.0     || square == params.good.1;
      let bad      = square == params.bad.0      || square == params.bad.1;

      let mut bg = if parity { (181, 135, 99) } else { (212, 190, 154) };

      if previous        { bg = if parity { (159, 145,  71) } else { (209, 193, 116) }; }
      if bad             { bg = if parity { (206,  80,  69) } else { (233, 105,  95) }; }
      if good            { bg = if parity { ( 19, 167,  77) } else { ( 65, 191, 103) }; }
      if current         { bg = if parity { (123, 102, 229) } else { (146, 126, 252) }; }
      if current && bad  { bg = if parity { (195,  77, 156) } else { (219, 103, 179) }; }
      if current && good { bg = if parity { (  8, 136, 126) } else { ( 60, 160, 149) }; }

      eprint!("\x1B[48;2;{};{};{}m", bg.0, bg.1, bg.2);

      if p < 0 {
        eprint!("   ");
      }
      else {
        if p >= 16 {
          eprint!("\x1B[97;101m");
          eprint!(" ? ");
        }
        else {
          match Color::from_u8(p as u8 >> 3) {
            Color::White => eprint!("\x1B[97m"),
            Color::Black => eprint!("\x1B[30m"),
          }
          eprint!(" {} ", DISPLAY[p as usize]);
        }
      }
    }
    eprintln!("\x1B[0m");
  }
  eprintln!("\x1B[2m{} to move  {:5} dfz\x1B[22m", state.turn, state.dfz);
}

pub fn show_derived(state : &State, network : &Network)
{
  let mut state = state.clone();
  let base = network.evaluate(&state);
  let base = match state.turn { Color::White => base, Color::Black => -base };

  eprint!("\x1B[0m");
  for _ in 0..24 { eprint!("\n"); }
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
          Color::White => eprint!("\x1B[97m"),
          Color::Black => eprint!("\x1B[30m"),
        }
        eprint!("   {}   \x1B[7D\x1B[B", UPPER[piece as usize]);
        if piece.kind() == KING {
          eprint!("       \x1B[7D");
        }
        else {
          state.boards[piece as usize] ^= 1u64 << square;
          let hyp = network.evaluate(&state);
          let ofs = match state.turn { Color::White => base - hyp, Color::Black => base + hyp };
          state.boards[piece as usize] ^= 1u64 << square;
          if ofs.abs() >= 10.0 {
            eprint!(" {:+5.1} \x1B[7D", ofs);
          }
          else {
            eprint!(" {:+5.2} \x1B[7D", ofs);
          }
        }
      }
      eprint!("\x1B[2A\x1B[7C");
    }
    eprint!("\x1B[0m\r\x1B[3B");
  }
  eprintln!("NNUE evaluation {:+.3}\x1B[44G{} to move", base, state.turn);
}
