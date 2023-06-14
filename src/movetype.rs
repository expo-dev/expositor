use crate::algebraic::Algebraic;
use crate::color::Color::*;
use crate::misc::{
  piece_destinations,
  pawn_attacks,
  WHITE_SHORT_CASTLE_BTWN,
  WHITE_LONG_CASTLE_BTWN,
  BLACK_SHORT_CASTLE_BTWN,
  BLACK_LONG_CASTLE_BTWN,
  WHITE_SHORT_CASTLE_CHCK,
  WHITE_LONG_CASTLE_CHCK,
  BLACK_SHORT_CASTLE_CHCK,
  BLACK_LONG_CASTLE_CHCK
};
use crate::piece::Kind::{self, *};
use crate::piece::Piece;
use crate::state::State;

use std::fmt::Write;

const CAPTURE_MASK : u8 = 0b_0000_0001; // .... ...1
const  ENPASS_MASK : u8 = 0b_0000_0010; // .... ..1.
const PROMOTE_MASK : u8 = 0b_0000_0100; // .... .1..
const  CASTLE_MASK : u8 = 0b_0000_1000; // .... 1...
const    PAWN_MASK : u8 = 0b_0001_0000; // ...1 ....
const GAINFUL_MASK : u8 = 0b_0000_0101; // .... .1.1 captures + promotions
const ZEROING_MASK : u8 = 0b_0001_0001; // ...1 ...1 captures + pawn moves
const UNUSUAL_MASK : u8 = 0b_0000_0110; // .... .11. promotions + en passant

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum MoveType {
  Manoeuvre          = 0b_0000_0000,  // .... ....
  Capture            = 0b_0000_0001,  // .... ...1
  Castle             = 0b_0000_1000,  // .... 1...
  PawnManoeuvre      = 0b_0001_0000,  // ...1 ....
  PawnCapture        = 0b_0001_0001,  // ...1 ...1
  CaptureEnPassant   = 0b_0001_0011,  // ...1 ..11
  Promotion          = 0b_0001_0100,  // ...1 .1..
  PromotionByCapture = 0b_0001_0101,  // ...1 .1.1
}

impl MoveType {
  #[inline] pub fn is_manoeuvre(self) -> bool { return (self as u8) &   !PAWN_MASK == 0; }
  #[inline] pub fn is_capture(self)   -> bool { return (self as u8) & CAPTURE_MASK != 0; }
  #[inline] pub fn is_enpass(self)    -> bool { return (self as u8) &  ENPASS_MASK != 0; }
  #[inline] pub fn is_promotion(self) -> bool { return (self as u8) & PROMOTE_MASK != 0; }
  #[inline] pub fn is_castle(self)    -> bool { return (self as u8) &  CASTLE_MASK != 0; }
  #[inline] pub fn is_pawn(self)      -> bool { return (self as u8) &    PAWN_MASK != 0; }
  #[inline] pub fn is_gainful(self)   -> bool { return (self as u8) & GAINFUL_MASK != 0; }
  #[inline] pub fn is_zeroing(self)   -> bool { return (self as u8) & ZEROING_MASK != 0; }
  #[inline] pub fn is_unusual(self)   -> bool { return (self as u8) & UNUSUAL_MASK != 0; }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Move {
  pub src        : i8,
  pub dst        : i8,
  pub piece      : Piece,
  pub promotion  : Piece,

  pub captured   : Piece,
  pub movetype   : MoveType,
  pub givescheck : u8,
  pub score      : i8,
}

impl Move {
  pub const NULL : Move = Move {
    movetype:   MoveType::Manoeuvre,
    givescheck: 0,
    src:        0,
    dst:        0,
    piece:      Piece::ZERO,
    captured:   Piece::ZERO,
    promotion:  Piece::ZERO,
    score:      0
  };
}

// NOTE that this function compares all fields of a Move except givescheck and
//   score. The order of the comparisons is essentially arbitrary, but they are
//   roughly in the order of likeliness to fail.
impl PartialEq for Move {
  fn eq(&self, other : &Self) -> bool
  {
    return self.src       == other.src
        && self.dst       == other.dst
        && self.piece     == other.piece
        && self.promotion == other.promotion
        && self.captured  == other.captured
        && self.movetype  == other.movetype;
  }
}

impl Eq for Move {}

// NOTE that this function only compares the first four fields of a Move: src,
//   dst, piece, and promotion. This requires that the Move struct uses C layout
//   rules and that those four fields are first in the struct declaration.
// NOTE that this is mega jank.
pub fn fast_eq(a : &Move, b : &Move) -> bool
{
  let ptr_a : *const Move = a;
  let ptr_b : *const Move = b;
  return unsafe { *(ptr_a as *const u32) == *(ptr_b as *const u32) };
}

impl Move {
  #[inline] pub fn is_manoeuvre(&self) -> bool { return self.movetype.is_manoeuvre(); }
  #[inline] pub fn is_capture(&self)   -> bool { return self.movetype.is_capture();   }
  #[inline] pub fn is_enpass(&self)    -> bool { return self.movetype.is_enpass();    }
  #[inline] pub fn is_promotion(&self) -> bool { return self.movetype.is_promotion(); }
  #[inline] pub fn is_castle(&self)    -> bool { return self.movetype.is_castle();    }
  #[inline] pub fn is_pawn(&self)      -> bool { return self.movetype.is_pawn();      }
  #[inline] pub fn is_gainful(&self)   -> bool { return self.movetype.is_gainful();   }
  #[inline] pub fn is_zeroing(&self)   -> bool { return self.movetype.is_zeroing();   }
  #[inline] pub fn is_unusual(&self)   -> bool { return self.movetype.is_unusual();   }
  #[inline] pub fn is_null(&self)      -> bool { return self.src == self.dst;         }

  #[inline] pub fn gives_check(&self)            -> bool { return self.givescheck     != 0; }
  #[inline] pub fn gives_direct_check(&self)     -> bool { return self.givescheck & 1 != 0; }
  #[inline] pub fn gives_discovered_check(&self) -> bool { return self.givescheck & 2 != 0; }

  const LT64 : u16 = 0b_0000_0000_0011_1111;
  const MSB  : u16 = 0b_1000_0000_0000_0000;

  pub fn compress(&self) -> u16
  {
    // Format is pkkk dddd ddss ssss
    //
    //      p   non-promotion if p = 0, promotion if p = 1
    //    kkk   ignored if p = 0, promotion kind if p = 1
    // dddddd   destination square
    // ssssss   source square
    //
    let src = self.src as u16;
    let dst = self.dst as u16;
    debug_assert!(src & !Self::LT64 == 0, "unable to compress source {}", src);
    debug_assert!(dst & !Self::LT64 == 0, "unable to compress destination {}", dst);
    let prom = if self.is_promotion() { Self::MSB } else { 0 };
    let kind = self.promotion.kind() as u16;
    return prom | (kind << 12) | (dst << 6) | src;
  }

  // NOTE that this function only checks the pseudolegality of the move in the
  //   given context (i.e. the king may be left in check)!
  // NOTE that this function does not set givescheck.
  // NOTE that this function does not set the score.
  pub fn decompress(state : &State, cm : u16) -> Self
  {
    use MoveType::*;

    let src = (   cm     & Self::LT64) as i8;
    let dst = ((cm >> 6) & Self::LT64) as i8;

    // Is this a null move?
    if src == dst { return Self::NULL; }

    // Is there a piece of the right color on the source?
    let piece = state.squares[src as usize];
    if piece.is_null() || piece.color() != state.turn { return Self::NULL; }

    // Is that piece attempting to capture another piece of its own color?
    let mut captured = state.squares[dst as usize];
    if !captured.is_null() && captured.color() == state.turn { return Self::NULL; }

    let promotion : Piece;
    let movetype : MoveType;

    let composite = state.sides[White] | state.sides[Black];

    // Non-promotions
    if cm & Self::MSB == 0 {
      if piece.kind() == Pawn {
        if dst < 8 || dst >= 56 { return Self::NULL; }
        if captured.is_null() {
          let front = src + match state.turn { White => 8, Black => -8 };
          if dst == front {
            movetype = PawnManoeuvre;
          }
          else if dst == state.enpass {
            let not_diagonal = pawn_attacks(state.turn, 1 << src) & (1 << dst) == 0;
            if not_diagonal { return Self::NULL; }
            captured = (!state.turn) + Pawn;
            movetype = CaptureEnPassant;
          }
          else {
            let start = match state.turn { White => src < 16, Black => src > 47 };
            if !start { return Self::NULL; }
            if (1 << front) & composite != 0 { return Self::NULL; }
            let advance = front + match state.turn { White => 8, Black => -8 };
            if dst != advance { return Self::NULL; }
            movetype = PawnManoeuvre;
          }
        }
        else {
          let not_diagonal = pawn_attacks(state.turn, 1 << src) & (1 << dst) == 0;
          if not_diagonal { return Self::NULL; }
          movetype = PawnCapture;
        }
      }
      else if piece.kind() == King && (dst - src).abs() == 2 {
        let direction = if dst < src { 1 } else { 0 };

        // Does the corresponding right exist?
        //   (This also ensures the king is on the correct square and that the rook is present)
        let rights_mask = match state.turn { White => [1, 2], Black => [4, 8] };
        if state.rights & rights_mask[direction] == 0 { return Self::NULL; }

        // Are the intermediate squares clear?
        let intermediate = match state.turn {
          White => [WHITE_SHORT_CASTLE_BTWN, WHITE_LONG_CASTLE_BTWN],
          Black => [BLACK_SHORT_CASTLE_BTWN, BLACK_LONG_CASTLE_BTWN],
        };
        if composite & intermediate[direction] != 0 { return Self::NULL; }

        // Are the travel squares safe?
        let travel = match state.turn {
          White => [WHITE_SHORT_CASTLE_CHCK, WHITE_LONG_CASTLE_CHCK],
          Black => [BLACK_SHORT_CASTLE_CHCK, BLACK_LONG_CASTLE_CHCK],
        };
        let danger = state.attacks_by(!state.turn, composite);
        if danger & travel[direction] != 0 { return Self::NULL; }

        movetype = Castle;
      }
      else {
        // Is the destination reachable?
        let possible = piece_destinations(piece.kind(), src as usize, composite);
        if (1 << dst) & possible == 0 { return Self::NULL; }
        movetype = if captured.is_null() { Manoeuvre } else { Capture };
      }
      promotion = Piece::ZERO;
    }

    // Promotions
    else {
      if piece.kind() != Pawn { return Self::NULL; }
      let endzone = match state.turn { White => dst >= 56, Black => dst < 8 };
      if !endzone { return Self::NULL; }
      if captured.is_null() {
        let step = dst - src == match state.turn { White => 8, Black => -8 };
        if !step { return Self::NULL; }
        movetype = Promotion;
      }
      else {
        let not_diagonal = pawn_attacks(state.turn, 1 << src) & (1 << dst) == 0;
        if not_diagonal { return Self::NULL; }
        movetype = PromotionByCapture;
      }
      let kind : Kind = unsafe { std::mem::transmute(((cm >> 12) & 0b_0111) as u8) };
      promotion = state.turn + kind;
    }

    if captured.is_null() { captured = Piece::ZERO };
    return Self {
      src:        src,
      dst:        dst,
      piece:      piece,
      promotion:  promotion,
      captured:   captured,
      movetype:   movetype,
      givescheck: 0,
      score:      0,
    };
  }
}

impl std::fmt::Display for Move {
  // NOTE that this does not disambiguate moves as required of short algebraic
  //   notation and does not annotate checkmates, as these are contextual.
  fn fmt(&self, f : &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    let mut written = 0;
    if self.is_null() {
      f.write_str("0000")?;
      written += 4;
    }
    else {
      let kind = self.piece.kind();
      if kind == King && self.dst == self.src + 2 {
        f.write_str("0-0")?;
        written += 3;
      }
      else if kind == King && self.dst == self.src - 2 {
        f.write_str("0-0-0")?;
        written += 5;
      }
      else {
        if kind == Pawn {
          if self.is_capture() {
            let files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
            f.write_char(files[self.src as usize % 8])?;
            written += 1;
          }
        }
        else {
          f.write_char(kind.upper())?;
          written += 1;
        }
        if self.is_capture() {
          f.write_char('x')?;
          written += 1;
        }
        f.write_str(&self.dst.algebraic())?;
        written += 2;
        if self.is_promotion() {
          f.write_char('=')?;
          f.write_char(self.promotion.kind().upper())?;
          written += 2;
        }
      }
      if self.gives_check() {
        f.write_char('+')?;
        written += 1;
      }
    }
    if let Some(width) = f.width() {
      if written < width { for _ in 0..(width-written) { f.write_char(' ')?; } }
    }
    return Ok(());
  }
}
