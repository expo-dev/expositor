use crate::color::Color;

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
  King   = 0,
  Queen  = 1,
  Rook   = 2,
  Bishop = 3,
  Knight = 4,
  Pawn   = 5,
}

use Kind::*;
pub const KQRBNP : [Kind; 6] = [King, Queen, Rook, Bishop, Knight, Pawn];
pub const KQRBN  : [Kind; 5] = [King, Queen, Rook, Bishop, Knight      ];
pub const QRBNP  : [Kind; 5] = [      Queen, Rook, Bishop, Knight, Pawn];
pub const QRBN   : [Kind; 4] = [      Queen, Rook, Bishop, Knight      ];
pub const QRB    : [Kind; 3] = [      Queen, Rook, Bishop              ];

pub const PNBRQK : [Kind; 6] = [Pawn, Knight, Bishop, Rook, Queen, King];
pub const  NBRQK : [Kind; 5] = [      Knight, Bishop, Rook, Queen, King];
pub const  PNBRQ : [Kind; 5] = [Pawn, Knight, Bishop, Rook, Queen      ];
pub const   NBRQ : [Kind; 4] = [      Knight, Bishop, Rook, Queen      ];
pub const    BRQ : [Kind; 3] = [              Bishop, Rook, Queen      ];

impl std::ops::Add<Kind> for Color {
  type Output = Piece;

  #[inline]
  fn add(self, rhs : Kind) -> Self::Output
  {
    return unsafe { std::mem::transmute(self as u8 * 8 + rhs as u8) };
  }
}

impl<T> std::ops::Index<Kind> for [T] {
  type Output = T;

  #[inline]
  fn index(&self, idx : Kind) -> &Self::Output
  {
    return &self[idx as usize];
  }
}

impl<T> std::ops::IndexMut<Kind> for [T] {
  #[inline]
  fn index_mut(&mut self, idx : Kind) -> &mut Self::Output
  {
    return &mut self[idx as usize];
  }
}

impl Kind {
  pub fn upper(self) -> char
  {
    return [b'K', b'Q', b'R', b'B', b'N', b'P'][self as usize] as char;
  }

  pub fn lower(self) -> char
  {
    return [b'k', b'q', b'r', b'b', b'n', b'p'][self as usize] as char;
  }
}

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Piece {
  WhiteKing   =  0,
  WhiteQueen  =  1,
  WhiteRook   =  2,
  WhiteBishop =  3,
  WhiteKnight =  4,
  WhitePawn   =  5,
  //             6
  //             7
  BlackKing   =  8,
  BlackQueen  =  9,
  BlackRook   = 10,
  BlackBishop = 11,
  BlackKnight = 12,
  BlackPawn   = 13,
  //            14
  //            15
  Null = 255
}

// impl std::ops::Not for Piece {
//   type Output = Self;
//
//   #[inline]
//   fn not(self) -> Self::Output
//   {
//     return unsafe { std::mem::transmute(self as u8 ^ 8) };
//   }
// }

impl<T> std::ops::Index<Piece> for [T] {
  type Output = T;

  #[inline]
  fn index(&self, idx : Piece) -> &Self::Output
  {
    return &self[idx as usize];
  }
}

impl<T> std::ops::IndexMut<Piece> for [T] {
  #[inline]
  fn index_mut(&mut self, idx : Piece) -> &mut Self::Output
  {
    return &mut self[idx as usize];
  }
}

impl Piece {
  pub const ZERO : Piece = Piece::WhiteKing;

  #[inline]
  pub fn from(p : u8) -> Self
  {
    debug_assert!((p < 16 && p % 8 < 6) || p == 255);
    return unsafe { std::mem::transmute(p) };
  }

  #[inline]
  pub fn color(self) -> Color
  {
    debug_assert!(!self.is_null());
    return unsafe { std::mem::transmute(self as u8 >> 3) };
  }

  #[inline]
  pub fn kind(self) -> Kind
  {
    debug_assert!(!self.is_null());
    return unsafe { std::mem::transmute(self as u8 & 7) };
  }

  #[inline]
  pub fn is_null(self) -> bool
  {
    return (self as i8) < 0;
  }

  #[inline]
  pub fn is_ranging(self) -> bool
  {
    let kind = self as u8 & 7;
    return kind < 4 && kind != 0;
  }

  #[inline]
  pub fn abbrev(self) -> char
  {
    let abbrevs =
      [b'K', b'Q', b'R', b'B', b'N', b'P', 0, 0,
       b'k', b'q', b'r', b'b', b'n', b'p', 0, 0];
    return abbrevs[self as usize] as char;
  }
}
