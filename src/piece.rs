use crate::color::*;
use std::fmt::{self, Write};

pub const KING   : usize = 0;
pub const QUEEN  : usize = 1;
pub const ROOK   : usize = 2;
pub const BISHOP : usize = 3;
pub const KNIGHT : usize = 4;
pub const PAWN   : usize = 5;

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece {
  WhiteKing   =  0,
  WhiteQueen  =  1,
  WhiteRook   =  2,
  WhiteBishop =  3,
  WhiteKnight =  4,
  WhitePawn   =  5,
  BlackKing   =  8,
  BlackQueen  =  9,
  BlackRook   = 10,
  BlackBishop = 11,
  BlackKnight = 12,
  BlackPawn   = 13,
  NullPiece   = 255
}

pub const ZERO_PIECE : Piece = Piece::WhiteKing;

impl Piece {
  pub fn from_u8(x : u8) -> Self
  {
    debug_assert!(  x < 16,  "cannot convert u8 to Piece");
    debug_assert!(x & 7 < 6, "cannot convert u8 to Piece");
    return unsafe { std::mem::transmute(x) };
  }

  pub fn from_usize(x : usize) -> Self
  {
    debug_assert!(  x < 16,  "cannot convert usize to Piece");
    debug_assert!(x & 7 < 6, "cannot convert usize to Piece");
    return unsafe { std::mem::transmute(x as u8) };
  }

  pub fn from_i8(x : i8) -> Self
  {
    if x < 0 { return Self::NullPiece; }
    debug_assert!(  x < 16,  "cannot convert i8 to Piece");
    debug_assert!(x & 7 < 6, "cannot convert i8 to Piece");
    return unsafe { std::mem::transmute(x) };
  }

  pub fn new(color : Color, piece : u8) -> Self
  {
    return Self::from_u8(color as u8 * 8 + piece);
  }

  pub fn color(self) -> Color
  {
    return Color::from_u8(self as u8 >> 3);
  }

  pub fn kind(self) -> usize
  {
    return self as usize & 7;
  }

  pub fn is_ranging(self) -> bool
  {
    let kind = self as u8 & 7;
    return kind < 4 && kind != 0;
  }

  pub fn is_null(self) -> bool
  {
    return (self as i8) < 0;
  }
}

pub const NAME : [&str; 16] = ["King", "Queen", "Rook", "Bishop", "Knight", "Pawn", "?", "?",
                               "King", "Queen", "Rook", "Bishop", "Knight", "Pawn", "?", "?"];

pub const FULLNAME : [&str; 16] =
  ["White King", "White Queen", "White Rook", "White Bishop", "White Knight", "White Pawn", "?", "?",
   "Black King", "Black Queen", "Black Rook", "Black Bishop", "Black Knight", "Black Pawn", "?", "?"];

pub const ABBREVIATION : [char; 16] = ['K', 'Q', 'R', 'B', 'N', 'P', '?', '?',
                                       'k', 'q', 'r', 'b', 'n', 'p', '?', '?'];

pub const UPPER : [char; 16] = ['K', 'Q', 'R', 'B', 'N', 'P', '?', '?',
                                'K', 'Q', 'R', 'B', 'N', 'P', '?', '?'];

pub const LOWER : [char; 16] = ['k', 'q', 'r', 'b', 'n', 'p', '?', '?',
                                'k', 'q', 'r', 'b', 'n', 'p', '?', '?'];

pub const DISPLAY : [char; 16] = ['K', 'Q', 'R', 'B', 'N', '●', '?', '?',
                                  'K', 'Q', 'R', 'B', 'N', '●', '?', '?'];

impl fmt::Display for Piece {
  fn fmt(&self, f : &mut fmt::Formatter<'_>) -> fmt::Result
  {
    f.write_char(if (*self as usize) < 16 { ABBREVIATION[*self as usize] } else { 'X' })
  }
}
