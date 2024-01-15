use crate::color::Color::{self, *};
use crate::dest::*;
use crate::global::nnue_enabled;
use crate::piece::Kind::{self, *};
use crate::piece::Piece;
use crate::piece::KQRBN;
use crate::span::*;
use crate::state::State;

// ↓↓↓ TEMPORARY ↓↓↓
impl State {
  pub fn evaluate_hce(&self) -> f32
  {
    return (self.sides [ self.turn     ].count_ones() as f32) * 3.00
         - (self.boards[ self.turn+Pawn].count_ones() as f32) * 1.00
         - (self.sides [!self.turn     ].count_ones() as f32) * 3.00
         + (self.boards[!self.turn+Pawn].count_ones() as f32) * 1.00
         + ((self.key & 255) as i8) as f32 / 512.0
         + 0.125;
  }
}
// ↑↑↑ TEMPORARY ↑↑↑

impl State {
  pub fn game_eval(&self) -> i16
  {
    if self.dfz >= 100 { return 0; }
    if let Some(score) = self.endgame() { return score; }
    let raw = if nnue_enabled() { self.evaluate() } else { self.evaluate_hce() };
    // ↓↓↓ DEBUG ↓↓↓
    // let check = unsafe { crate::nnue::NETWORK.evaluate(&self, self.head_index()) };
    // if (check - raw).abs() > 0.001 { panic!("{} ~ {}", check, raw); }
    // ↑↑↑ DEBUG ↑↑↑
    // if self.dfz > 20 { return (raw * ((120 - self.dfz) as f32)).round() as i16; }
    return (raw * 100.0).round() as i16;
  }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Op {
  pub square : i8,
  pub piece  : Piece,
}

pub const NOP : Op = Op { square: -1, piece: Piece::Null };

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum NodeKind {
  Unk = 0b_00,
  All = 0b_01,  // upper bound (the score is this or less)
  Cut = 0b_10,  // lower bound (the score is at least this)
  PV  = 0b_11,  // exact score
}

impl NodeKind {
  pub fn as_str(&self) -> &str
  {
    return match self {
      NodeKind::Unk => "unknown",
      NodeKind::All => "upper bound",
      NodeKind::Cut => "lower bound",
      NodeKind::PV  => "exact score",
    };
  }

  pub fn abbrev(&self) -> &str
  {
    return match self {
      NodeKind::Unk => "Unk",
      NodeKind::All => "All",
      NodeKind::Cut => "Cut",
      NodeKind::PV  => "P-V",
    };
  }
}

pub const SQRT2 : f64 = 1.414_213_562_373_095_049;
pub const SQRT3 : f64 = 1.732_050_807_568_877_293;
pub const SQRT6 : f64 = 2.449_489_742_783_178_098;

pub const ONE_THIRD : f64 = 0.333_333_333_333_333_333;
pub const TWO_THIRD : f64 = 0.666_666_666_666_666_667;

#[inline] pub const fn hmirror(x : usize) -> usize { return x ^  7; }
#[inline] pub const fn vmirror(x : usize) -> usize { return x ^ 56; }
#[inline] pub const fn  rotate(x : usize) -> usize { return x ^ 63; }

pub const FILE_A : u64 = 0x_01_01_01_01_01_01_01_01;
pub const FILE_H : u64 = 0x_80_80_80_80_80_80_80_80;

pub const RANK_1 : u64 = 0x_00_00_00_00_00_00_00_ff;
pub const RANK_8 : u64 = 0x_ff_00_00_00_00_00_00_00;

pub const LIGHT_SQUARES : u64 = 0x_55_aa_55_aa_55_aa_55_aa;
pub const  DARK_SQUARES : u64 = 0x_aa_55_aa_55_aa_55_aa_55;

pub const WHITE_HOME : u64 = 0x_00_00_00_00_00_00_ff_ff;
pub const BLACK_HOME : u64 = 0x_ff_ff_00_00_00_00_00_00;

pub const WHITE_SHORT_CASTLE_ROOK : u64 = 0b_1010_0000;
pub const WHITE_SHORT_CASTLE_KING : u64 = 0b_0101_0000;
pub const WHITE_SHORT_CASTLE_BTWN : u64 = 0b_0110_0000;
pub const WHITE_SHORT_CASTLE_CHCK : u64 = 0b_0111_0000;
pub const  WHITE_LONG_CASTLE_ROOK : u64 = 0b_0000_1001;
pub const  WHITE_LONG_CASTLE_KING : u64 = 0b_0001_0100;
pub const  WHITE_LONG_CASTLE_BTWN : u64 = 0b_0000_1110;
pub const  WHITE_LONG_CASTLE_CHCK : u64 = 0b_0001_1100;

pub const BLACK_SHORT_CASTLE_ROOK : u64 = WHITE_SHORT_CASTLE_ROOK << 56;
pub const BLACK_SHORT_CASTLE_KING : u64 = WHITE_SHORT_CASTLE_KING << 56;
pub const BLACK_SHORT_CASTLE_BTWN : u64 = WHITE_SHORT_CASTLE_BTWN << 56;
pub const BLACK_SHORT_CASTLE_CHCK : u64 = WHITE_SHORT_CASTLE_CHCK << 56;
pub const  BLACK_LONG_CASTLE_ROOK : u64 =  WHITE_LONG_CASTLE_ROOK << 56;
pub const  BLACK_LONG_CASTLE_KING : u64 =  WHITE_LONG_CASTLE_KING << 56;
pub const  BLACK_LONG_CASTLE_BTWN : u64 =  WHITE_LONG_CASTLE_BTWN << 56;
pub const  BLACK_LONG_CASTLE_CHCK : u64 =  WHITE_LONG_CASTLE_CHCK << 56;

#[inline]
pub fn sweep_n(u : u64) -> u64
{
  let v = u | (u <<  8);
  let w = v | (v << 16);
  return  w | (w << 32);
}

#[inline]
pub fn sweep_s(u : u64) -> u64
{
  let v = u | (u >>  8);
  let w = v | (v >> 16);
  return  w | (w >> 32);
}

#[inline] pub fn shift_nw(board : u64) -> u64 { return (board & !FILE_A) << 7; }
#[inline] pub fn shift_ne(board : u64) -> u64 { return (board & !FILE_H) << 9; }
#[inline] pub fn shift_sw(board : u64) -> u64 { return (board & !FILE_A) >> 9; }
#[inline] pub fn shift_se(board : u64) -> u64 { return (board & !FILE_H) >> 7; }

#[inline]
pub fn pawn_attacks(color : Color, sources : u64) -> u64
{
  return match color {
    White => shift_nw(sources) | shift_ne(sources),
    Black => shift_sw(sources) | shift_se(sources),
  };
}

#[inline]
pub fn piece_destinations(kind : Kind, src : usize, composite : u64) -> u64
{
  return match kind {
    King   =>   king_destinations(           src),
    Queen  =>  queen_destinations(composite, src),
    Rook   =>   rook_destinations(composite, src),
    Bishop => bishop_destinations(composite, src),
    Knight => knight_destinations(           src),
    _ => unreachable!()
  };
}

#[inline]
pub fn crux_span(a : usize, b : usize) -> u64 // cruciform span
{
  return unsafe { *CRUX_SPAN.get_unchecked(a).get_unchecked(b) };
}

#[inline]
pub fn salt_span(a : usize, b : usize) -> u64 // saltire span
{
  return unsafe { *SALT_SPAN.get_unchecked(a).get_unchecked(b) };
}

#[inline]
pub fn line_span(a : usize, b : usize) -> u64
{
  return unsafe { *LINE_SPAN.get_unchecked(a).get_unchecked(b) };
}

#[inline]
pub fn line_thru(a : usize, b : usize) -> u64
{
  return unsafe { *LINE_THRU.get_unchecked(a).get_unchecked(b) };
}

impl State {
  pub fn in_check(&self, color : Color) -> bool
  {
    let king_board = self.boards[color + King];
    let king_square = king_board.trailing_zeros() as usize;
    let composite = self.sides[White] | self.sides[Black];
    let opp_boards = match color {
      White => &self.boards[8..15],
      Black => &self.boards[0..7]
    };
    let qr = opp_boards[Rook  ] | opp_boards[Queen];
    let qb = opp_boards[Bishop] | opp_boards[Queen];
    if      qr &   rook_destinations(composite, king_square) != 0 { return true; }
    if      qb & bishop_destinations(composite, king_square) != 0 { return true; }
    if opp_boards[Knight] & knight_destinations(king_square) != 0 { return true; }
    if opp_boards[Pawn]   & pawn_attacks(color, king_board)  != 0 { return true; }
    if opp_boards[King]   & king_destinations(king_square)   != 0 { return true; }
    return false;
  }

  pub fn attacks_by(&self, color : Color, composite : u64) -> u64
  {
    let mut attacked : u64 = 0;
    for piece in KQRBN {
      let mut sources = self.boards[color+piece];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let destinations = piece_destinations(piece, src, composite);
        attacked |= destinations;
        sources &= sources - 1;
      }
    }
    return attacked | pawn_attacks(color, self.boards[color+Pawn]);
  }

  pub fn attackers(&self, square : usize, color : Color) -> u64
  {
    let composite = self.sides[White] | self.sides[Black];
    let boards = match color {
      White => &self.boards[0..7],
      Black => &self.boards[8..15]
    };
    return ((boards[Rook  ] | boards[Queen]) &   rook_destinations(composite, square))
         | ((boards[Bishop] | boards[Queen]) & bishop_destinations(composite, square))
         | ( boards[Knight]                  & knight_destinations(           square))
         | ( boards[Pawn  ]                  &      pawn_attacks(!color, 1 << square));
  }
}
