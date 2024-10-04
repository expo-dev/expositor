use crate::misc::Algebraic;
use crate::color::Color::*;
use crate::piece::Piece::*;
use crate::state::State;

use std::str::SplitAsciiWhitespace;

impl State {
  pub fn from_fen(record : &str) -> Result<State, &'static str>
  {
    #[cfg(debug_assertions)] if !record.is_ascii() { return Err("not ASCII"); }
    let mut fields = record.split_ascii_whitespace();
    let result = Self::from_fen_fields(&mut fields);
    if !fields.next().is_none() { return Err("trailing fields"); }
    return result;
  }

  // NOTE that this doesn't check whether there is extraneous output at the end.
  pub fn from_fen_fields(record : &mut SplitAsciiWhitespace) -> Result<State, &'static str>
  {
    let mut state = State {
      sides:   [0; 2],
      boards:  [0; 16],
      squares: [Null; 64],
      rights:  0,
      enpass:  -1,
      incheck: false,
      turn:    White,
      dfz:     0,
      ply:     0,
      key:     0,
      s1:      Vec::new(),
    };
    let layout = match record.next() {
      Some(field) => field,
      None => return Err("nothing to parse")
    };
    let mut rank : i8 = 7;
    let mut file : i8 = 0;
    for c in layout.as_bytes() {
      if *c == b'/' {
        rank -= 1;
        if rank < 0 { return Err("more than eight ranks"); }
        if file != 8 { return Err("incomplete rank"); }
        file = 0;
        continue;
      }
      if file > 7 { return Err("more than eight files"); }
      if b'8' >= *c && *c > b'0' {
        file += c.wrapping_sub(b'0') as i8;
        continue;
      }
      let piece = match *c {
        b'K' => WhiteKing   ,
        b'Q' => WhiteQueen  ,
        b'R' => WhiteRook   ,
        b'B' => WhiteBishop ,
        b'N' => WhiteKnight ,
        b'P' => WhitePawn   ,
        b'k' => BlackKing   ,
        b'q' => BlackQueen  ,
        b'r' => BlackRook   ,
        b'b' => BlackBishop ,
        b'n' => BlackKnight ,
        b'p' => BlackPawn   ,
        _ => return Err("spurious character in layout")
      };
      let square = (rank*8 + file) as usize;
      state.sides[piece.color()] |= 1u64 << square;
      state.boards[piece] |= 1u64 << square;
      state.squares[square] = piece;
      file += 1;
    }
    if rank != 0 { return Err("fewer than eight ranks"); }

    let side_to_move = match record.next() {
      Some(field) => field,
      None => return Err("missing side to move")
    };
    state.turn = match side_to_move {
      "w" => White,
      "b" => Black,
       _  => return Err("invalid side to move")
    };

    if let Some(rights) = record.next() {
      if rights != "-" {
        for c in rights.as_bytes() {
          state.rights |= match *c {
            b'K' => 1,
            b'Q' => 2,
            b'k' => 4,
            b'q' => 8,
            _  => return Err("spurious character in rights")
          };
        }
      }
    }
    else {
      state.rights = 0b_1111;
    }

    if state.boards[WhiteKing] & (1 << 4) == 0 { state.rights &= 0b_1100; }
    if state.boards[WhiteRook] & (1 << 7) == 0 { state.rights &= 0b_1110; }
    if state.boards[WhiteRook] & (1 << 0) == 0 { state.rights &= 0b_1101; }

    if state.boards[BlackKing] & (1 << 60) == 0 { state.rights &= 0b_0011; }
    if state.boards[BlackRook] & (1 << 63) == 0 { state.rights &= 0b_1011; }
    if state.boards[BlackRook] & (1 << 56) == 0 { state.rights &= 0b_0111; }

    if let Some(enpass) = record.next() {
      if enpass != "-" {
        if enpass.len() != 2 { return Err("invalid en passant target"); }
        let enpass = enpass.as_bytes();
        if enpass[0] < b'a' || b'h' < enpass[0]
          || enpass[1] < b'1' || b'8' < enpass[1] {
          return Err("invalid en passant target");
        }
        let file = enpass[0] - b'a';
        let rank = enpass[1] - b'1';
        state.enpass = (rank*8 + file) as i8;
      }
    }

    if let Some(dfz) = record.next() {
      match dfz.parse::<u16>() {
        Ok(n) => { state.dfz = n; }
        Err(_) => { return Err("invalid number of ply since zeroing"); }
      }
    }

    if let Some(ply) = record.next() {
      match ply.parse::<u16>() {
        Ok(n) => {
          state.ply = (std::cmp::max(n, 1) - 1) * 2 + state.turn as u16;
        }
        Err(_) => { return Err("invalid move number"); }
      }
    }
    else {
      state.ply = state.turn as u16;
    }

    state.incheck = state.in_check(state.turn);
    state.key = state.zobrist();
    return Ok(state);
  }

  pub fn to_fen(&self) -> String
  {
    let mut record = String::new();

    for rank in (0..8).rev() {
      let mut run = 0;
      for file in 0..8 {
        let idx = rank*8 + file;
        if self.squares[idx].is_null() {
          run += 1;
        }
        else {
          if run > 0 {
            record.push_str(&run.to_string());
            run = 0;
          }
          record.push(self.squares[idx].abbrev());
        }
      }
      if run > 0 { record.push_str(&run.to_string()); }
      record.push(if rank > 0 {'/'} else {' '});
    }

    record.push(match self.turn {
      White => 'w',
      Black => 'b',
    });

    record.push(' ');

    let white_kingside  = self.rights & 1 != 0;
    let white_queenside = self.rights & 2 != 0;
    let black_kingside  = self.rights & 4 != 0;
    let black_queenside = self.rights & 8 != 0;
    if !white_kingside && !white_queenside && !black_kingside && !black_queenside {
      record.push('-');
    }
    else {
      if white_kingside  { record.push('K'); }
      if white_queenside { record.push('Q'); }
      if black_kingside  { record.push('k'); }
      if black_queenside { record.push('q'); }
    }

    record.push(' ');
    if self.enpass < 0 {
      record.push('-');
    }
    else {
      record.push_str(&self.enpass.id());
    }
    record.push(' ');
    record.push_str(&self.dfz.to_string());
    record.push(' ');
    record.push_str(&(1 + self.ply/2).to_string());
    return record;
  }
}
