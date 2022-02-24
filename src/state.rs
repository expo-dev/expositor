use crate::basis::*;
use crate::color::*;
use crate::piece::*;
use crate::nnue::*;

// I expect the compiler to lay out
//   this struct roughly as follows:
//
//   boards    128   128
//   squares    64   192
//   sides      16   208
//   inputs     24   232
//   key         8   240
//   dfz         2   242
//   ply         2   244
//   rights      1   245
//   enpass      1   246
//   incheck     1   247
//   turn        1   248
//  (padding     8   256)

#[derive(Clone)]
pub struct State {
  pub sides   : [u64; 2],     // side   -> composite board
  pub boards  : [u64; 16],    // piece  -> board
  pub squares : [Piece; 64],  // square -> piece

  pub rights  : u8,     // ....qkQK
  pub enpass  : i8,     // square or -1
  pub incheck : bool,   // side to move
  pub turn    : Color,  // side to move
  pub dfz     : u16,    // depth from zeroing
  pub ply     : u16,    // zero-indexed
  pub key     : u64,    // zobrist key

  pub s1 : Vec<[Simd32; V1]>
}

pub struct SavedMetadata {
  pub sides    : [u64; 2],
  pub rights   : u8,
  pub enpass   : i8,
  pub incheck  : bool,
  pub dfz      : u16,
  pub key      : u64,
}

impl State {
  pub const fn new() -> Self
  {
    let state = Self {
      sides: [
        0x000000000000FFFF,
        0xFFFF000000000000,
      ],
      boards: [
        0x0000000000000010,
        0x0000000000000008,
        0x0000000000000081,
        0x0000000000000024,
        0x0000000000000042,
        0x000000000000FF00,
        0,
        0,
        0x1000000000000000,
        0x0800000000000000,
        0x8100000000000000,
        0x2400000000000000,
        0x4200000000000000,
        0x00FF000000000000,
        0,
        0,
      ],
      squares: [
        Piece::WhiteRook, Piece::WhiteKnight, Piece::WhiteBishop, Piece::WhiteQueen,
        Piece::WhiteKing, Piece::WhiteBishop, Piece::WhiteKnight, Piece::WhiteRook ,

        Piece::WhitePawn, Piece::WhitePawn  , Piece::WhitePawn  , Piece::WhitePawn ,
        Piece::WhitePawn, Piece::WhitePawn  , Piece::WhitePawn  , Piece::WhitePawn ,

        Piece::NullPiece, Piece::NullPiece  , Piece::NullPiece  , Piece::NullPiece ,
        Piece::NullPiece, Piece::NullPiece  , Piece::NullPiece  , Piece::NullPiece ,

        Piece::NullPiece, Piece::NullPiece  , Piece::NullPiece  , Piece::NullPiece ,
        Piece::NullPiece, Piece::NullPiece  , Piece::NullPiece  , Piece::NullPiece ,

        Piece::NullPiece, Piece::NullPiece  , Piece::NullPiece  , Piece::NullPiece ,
        Piece::NullPiece, Piece::NullPiece  , Piece::NullPiece  , Piece::NullPiece ,

        Piece::NullPiece, Piece::NullPiece  , Piece::NullPiece  , Piece::NullPiece ,
        Piece::NullPiece, Piece::NullPiece  , Piece::NullPiece  , Piece::NullPiece ,

        Piece::BlackPawn, Piece::BlackPawn  , Piece::BlackPawn  , Piece::BlackPawn ,
        Piece::BlackPawn, Piece::BlackPawn  , Piece::BlackPawn  , Piece::BlackPawn ,

        Piece::BlackRook, Piece::BlackKnight, Piece::BlackBishop, Piece::BlackQueen,
        Piece::BlackKing, Piece::BlackBishop, Piece::BlackKnight, Piece::BlackRook ,
      ],
      rights:   0x0F,
      enpass:   -1,
      incheck:  false,
      turn:     Color::White,
      dfz:      0,
      ply:      0,
      key:      START_KEY,
      s1:       Vec::new(),
    };
    return state;
  }

  pub fn save(&self) -> SavedMetadata
  {
    return SavedMetadata {
      sides:   self.sides,
      rights:  self.rights,
      enpass:  self.enpass,
      incheck: self.incheck,
      dfz:     self.dfz,
      key:     self.key,
    };
  }

  pub fn restore(&mut self, saved : &SavedMetadata)
  {
    self.sides   = saved.sides;
    self.rights  = saved.rights;
    self.enpass  = saved.enpass;
    self.incheck = saved.incheck;
    self.dfz     = saved.dfz;
    self.key     = saved.key;
  }

  // For use when the squares field may not be set properly
  pub fn at_square(&self, square : usize, hint : Color) -> Piece
  {
    let mask = 1u64 << square;
    let ofs = hint as usize * 8;
    for x in 0..6 {
      if self.boards[ofs+x] & mask != 0 { return Piece::from_usize(ofs+x); }
    }
    return Piece::NullPiece;
  }
}
