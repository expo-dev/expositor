use crate::algebraic::*;
use crate::color::*;
use crate::piece::*;
use crate::state::*;

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
      squares: [Piece::NullPiece; 64],
      rights:  0,
      enpass:  -1,
      incheck: false,
      turn:    Color::White,
      dfz:     0,
      ply:     0,
      key:     0,
      s1:      Vec::new(),
    };
    let layout = match record.next() {
      Some(field) => field,
      None => return Err("missing board layout")
    };
    let mut rank : i8 = 7;
    let mut file : i8 = 0;
    for c in layout.as_bytes() {
      if *c as char == '/' {
        rank -= 1;
        if rank < 0 { return Err("more than eight ranks"); }
        if file != 8 { return Err("incomplete rank"); }
        file = 0;
        continue;
      }
      if file > 7 { return Err("more than eight files"); }
      if ('8' as u8) >= *c && *c > ('0' as u8) {
        file += (c - ('0' as u8)) as i8;
        continue;
      }
      let piece = match *c as char {
        'K' => Piece::WhiteKing   ,
        'Q' => Piece::WhiteQueen  ,
        'R' => Piece::WhiteRook   ,
        'B' => Piece::WhiteBishop ,
        'N' => Piece::WhiteKnight ,
        'P' => Piece::WhitePawn   ,
        'k' => Piece::BlackKing   ,
        'q' => Piece::BlackQueen  ,
        'r' => Piece::BlackRook   ,
        'b' => Piece::BlackBishop ,
        'n' => Piece::BlackKnight ,
        'p' => Piece::BlackPawn   ,
         _  => return Err("extraneous character in board layout")
      };
      let square = rank*8 + file;
      state.sides[piece.color() as usize] |= 1u64 << square;
      state.boards[piece as usize] |= 1u64 << square;
      state.squares[square as usize] = piece;
      file += 1;
    }
    if rank != 0 { return Err("fewer than eight ranks"); }

    let side_to_move = match record.next() {
      Some(field) => field,
      None => return Err("missing side to move")
    };
    state.turn = match side_to_move {
      "w" => Color::White,
      "b" => Color::Black,
       _  => return Err("side to move is not 'w' or 'b'")
    };

    let rights = match record.next() {
      Some(field) => field,
      None => return Err("missing castling rights")
    };
    if rights != "-" {
      for c in rights.as_bytes() {
        state.rights |= match *c as char {
          'K' => 1,
          'Q' => 2,
          'k' => 4,
          'q' => 8,
           _  => return Err("spurious character in castling rights")
        };
      }
    }

    if state.boards[WHITE+KING] & 0x0000000000000010 == 0 { state.rights &= 0xC; }
    if state.boards[WHITE+ROOK] & 0x0000000000000080 == 0 { state.rights &= 0xE; }
    if state.boards[WHITE+ROOK] & 0x0000000000000001 == 0 { state.rights &= 0xD; }

    if state.boards[BLACK+KING] & 0x1000000000000010 == 0 { state.rights &= 0x3; }
    if state.boards[BLACK+ROOK] & 0x8000000000000080 == 0 { state.rights &= 0xB; }
    if state.boards[BLACK+ROOK] & 0x0100000000000001 == 0 { state.rights &= 0x7; }

    let enpass = match record.next() {
      Some(field) => field,
      None => return Err("missing en passant target")
    };
    if enpass != "-" {
      if enpass.len() != 2 { return Err("invalid en passant target"); }
      let enpass = enpass.as_bytes();
      if enpass[0] > ('h' as u8) || enpass[0] < ('a' as u8)
        || enpass[1] > ('8' as u8) || enpass[1] < ('1' as u8) {
        return Err("invalid en passant target");
      }
      let file = enpass[0] - ('a' as u8);
      let rank = enpass[1] - ('1' as u8);
      state.enpass = (rank*8 + file) as i8;
    }

    let depth_from_zeroing = match record.next() {
      Some(field) => field,
      None => return Err("missing number of ply since zeroing")
    };
    state.dfz = match depth_from_zeroing.parse::<u16>() {
      Ok(n) => n, Err(_) => return Err("invalid number of ply since zeroing")
    };

    let move_number = match record.next() {
      Some(field) => field,
      None => return Err("missing move number")
    };
    state.ply = match move_number.parse::<u16>() {
      Ok(n) => if n == 0 { 0 } else { (n-1)*2 + state.turn as u16 },
      Err(_) => return Err("invalid move number")
    };

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
          record.push(ABBREVIATION[self.squares[idx] as usize]);
        }
      }
      if run > 0 { record.push_str(&run.to_string()); }
      record.push(if rank > 0 {'/'} else {' '});
    }

    record.push(match self.turn {
      Color::White => 'w',
      Color::Black => 'b',
    });

    record.push(' ');

    let white_kingside  = self.rights & 0x01 != 0;
    let white_queenside = self.rights & 0x02 != 0;
    let black_kingside  = self.rights & 0x04 != 0;
    let black_queenside = self.rights & 0x08 != 0;
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
      record.push_str(&self.enpass.algebraic());
    }
    record.push(' ');
    record.push_str(&self.dfz.to_string());
    record.push(' ');
    record.push_str(&(1 + self.ply/2).to_string());
    return record;
  }
}
