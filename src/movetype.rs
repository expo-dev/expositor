use crate::algebraic::*;
use crate::color::*;
use crate::misc::*;
use crate::piece::*;
use crate::state::*;

use std::fmt::{self, Write};

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

const CAPTURE_MASK : u8 = 0x01; // .... ...1
const ENPASS_MASK  : u8 = 0x02; // .... ..1.
const PROMOTE_MASK : u8 = 0x04; // .... .1..
const CASTLE_MASK  : u8 = 0x08; // .... 1...
const PAWN_MASK    : u8 = 0x10; // ...1 ....
const GAINFUL_MASK : u8 = 0x05; // .... .1.1  captures + promotions
const ZEROING_MASK : u8 = 0x11; // ...1 ...1  captures + pawn moves
const UNUSUAL_MASK : u8 = 0x06; // .... .11.  promotions + en passant

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum MoveType {
  Manoeuvre          = 0x00,    // .... ....
  Capture            = 0x01,    // .... ...1
  Castle             = 0x08,    // .... 1...
  PawnManoeuvre      = 0x10,    // ...1 ....
  PawnCapture        = 0x11,    // ...1 ...1
  CaptureEnPassant   = 0x13,    // ...1 ..11
  Promotion          = 0x14,    // ...1 .1..
  PromotionByCapture = 0x15,    // ...1 .1.1
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

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

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

pub const NULL_MOVE : Move = Move {
  movetype:   MoveType::Manoeuvre,
  givescheck: 0,
  src:        0,
  dst:        0,
  piece:      ZERO_PIECE,
  captured:   ZERO_PIECE,
  promotion:  ZERO_PIECE,
  score:      0
};

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
pub fn quick_eq(a : &Move, b : &Move) -> bool
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
    debug_assert!(src & 0xFFC0 == 0, "unable to compress source {}", src);
    debug_assert!(dst & 0xFFC0 == 0, "unable to compress destination {}", dst);
    let prom = if self.is_promotion() { 0x8000 } else { 0x0000 };
    let kind = self.promotion.kind() as u16;
    return prom | (kind << 12) | (dst << 6) | src;
  }

  // NOTE that this function only checks the pseudolegality of the move in the
  //   given context (i.e. the king may be left in check)!
  // NOTE that this function does not set givescheck.
  // NOTE that this function does not set the score.
  pub fn decompress(state : &State, cm : u16) -> Self
  {
    let src = (   cm     & 0x003F) as i8;
    let dst = ((cm >> 6) & 0x003F) as i8;

    // Is this a null move?
    if src == dst { return NULL_MOVE; }

    // Is there a piece of the right color on the source?
    let piece = state.squares[src as usize];
    if piece.is_null() || piece.color() != state.turn { return NULL_MOVE; }

    // Is that piece attempting to capture another piece of its own color?
    let mut captured = state.squares[dst as usize];
    if !captured.is_null() && captured.color() == state.turn { return NULL_MOVE; }

    let promotion : Piece;
    let movetype : MoveType;

    // Non-promotions
    if cm >> 15 == 0 {
      if piece.kind() == PAWN {
        if dst < 8 || dst >= 56 { return NULL_MOVE; }
        if captured.is_null() {
          let front = src + match state.turn { Color::White => 8, Color::Black => -8 };
          if dst == front {
            movetype = MoveType::PawnManoeuvre;
          }
          else if dst == state.enpass {
            let not_diagonal = pawn_attacks(state.turn, 1u64 << src) & (1u64 << dst) == 0;
            if not_diagonal { return NULL_MOVE; }
            captured = Piece::new(!state.turn, PAWN as u8);
            movetype = MoveType::CaptureEnPassant;
          }
          else {
            let start = match state.turn { Color::White => src < 16, Color::Black => src > 47 };
            if !start { return NULL_MOVE; }
            let composite = state.sides[W] | state.sides[B];
            if (1u64 << front) & composite != 0 { return NULL_MOVE; }
            let advance = front + match state.turn { Color::White => 8, Color::Black => -8 };
            if dst != advance { return NULL_MOVE; }
            movetype = MoveType::PawnManoeuvre;
          }
        }
        else {
          let not_diagonal = pawn_attacks(state.turn, 1u64 << src) & (1u64 << dst) == 0;
          if not_diagonal { return NULL_MOVE; }
          movetype = MoveType::PawnCapture;
        }
      }
      else if piece.kind() == KING && (dst - src).abs() == 2 {
        let direction = if dst < src { 1 } else { 0 };
        // Does the corresponding right exist?
        //   (This also ensures the king is on the correct square and that the rook is present)
        let rights_mask = match state.turn {
          Color::White => [0x01, 0x02], Color::Black => [0x04, 0x08]
        };
        if state.rights & rights_mask[direction] == 0 { return NULL_MOVE; }
        // Are the intermediate squares clear?
        let composite = state.sides[W] | state.sides[B];
        let intermediate = match state.turn {
          Color::White => [0x0000000000000060, 0x000000000000000E],
          Color::Black => [0x6000000000000000, 0x0E00000000000000],
        };
        if composite & intermediate[direction] != 0 { return NULL_MOVE; }
        // Are the travel squares safe?
        let travel = match state.turn {
          Color::White => [0x0000000000000070, 0x000000000000001C],
          Color::Black => [0x7000000000000000, 0x1C00000000000000],
        };
        let danger = state.attacked_by(!state.turn, composite);
        if danger & travel[direction] != 0 { return NULL_MOVE; }
        movetype = MoveType::Castle;
      }
      else {
        // Is the destination reachable?
        let composite = state.sides[W] | state.sides[B];
        let possible = piece_destinations(piece.kind(), src as usize, composite);
        if (1u64 << dst) & possible == 0 { return NULL_MOVE; }
        movetype = if captured.is_null() { MoveType::Manoeuvre } else { MoveType::Capture };
      }
      promotion = ZERO_PIECE;
    }

    // Promotions
    else {
      if piece.kind() != PAWN { return NULL_MOVE; }
      let endzone = match state.turn { Color::White => dst >= 56, Color::Black => dst < 8 };
      if !endzone { return NULL_MOVE; }
      if captured.is_null() {
        let step = dst - src == match state.turn { Color::White => 8, Color::Black => -8 };
        if !step { return NULL_MOVE; }
        movetype = MoveType::Promotion;
      }
      else {
        let not_diagonal = pawn_attacks(state.turn, 1u64 << src) & (1u64 << dst) == 0;
        if not_diagonal { return NULL_MOVE; }
        movetype = MoveType::PromotionByCapture;
      }
      promotion = Piece::new(state.turn, ((cm >> 12) & 0x0007) as u8);
    }

    if captured.is_null() { captured = ZERO_PIECE };
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

impl fmt::Display for Move {
  // NOTE that this does not disambiguate moves as required of short algebraic
  //   notation and does not annotate checkmates, as these are contextual.
  fn fmt(&self, f : &mut fmt::Formatter<'_>) -> fmt::Result
  {
    let mut written = 0;
    if self.is_null() {
      f.write_str("0000")?;
      written += 4;
    }
    else {
      let kind = self.piece.kind();
      if kind == KING && self.dst == self.src + 2 {
        f.write_str("0-0")?;
        written += 3;
      }
      else if kind == KING && self.dst == self.src - 2 {
        f.write_str("0-0-0")?;
        written += 5;
      }
      else {
        if kind == PAWN {
          if self.is_capture() {
            f.write_char(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'][self.src as usize % 8])?;
            written += 1;
          }
        }
        else {
          f.write_char(UPPER[self.piece as usize])?;
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
          f.write_char(UPPER[self.promotion as usize])?;
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
