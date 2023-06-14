use crate::color::Color::{self, *};
use crate::nnue::{HEADS, Network};
use crate::piece::Kind::King;
use crate::piece::Piece;
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

pub fn derived(state : &State, network : &Network)
{
  let mut state = state.clone();
  let head_idx = state.head_index();
  let base = network.evaluate(&state, head_idx);
  let base = match state.turn { White => base, Black => -base };

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
          White => eprint!("\x1B[97m"),
          Black => eprint!("\x1B[30m"),
        }
        eprint!("   {}   \x1B[7D\x1B[B", piece.kind().upper());
        if piece.kind() == King {
          eprint!("       \x1B[7D");
        }
        else {
          state.boards[piece] ^= 1 << square;
          let hyp = network.evaluate(&state, head_idx);
          let diff = match state.turn {
            White => base - hyp,
            Black => base + hyp
          };
          state.boards[piece] ^= 1 << square;
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
  eprint!("\x1B[2m");
  for h in 0..HEADS {
    eprint!("hd {} evaluation {:+7.3}", h, network.evaluate(&state, h));
    if h == head_idx { eprintln!(" <-"); } else { eprintln!(""); }
  }
  eprint!("\x1B[0m");
}
