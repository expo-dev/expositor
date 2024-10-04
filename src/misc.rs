use crate::color::Color::{self, *};
use crate::dest::*;
use crate::piece::Kind::{self, *};
use crate::piece::Piece;
use crate::piece::KQRBN;
use crate::span::*;
use crate::score::PovScore;
use crate::state::State;

impl State {
  pub fn game_eval(&self) -> PovScore
  {
    if self.dfz >= 100 { return PovScore::ZERO; }
    if let Some(score) = self.endgame() { return score; }
    // let raw = if nnue_enabled() { self.evaluate() } else { self._hce() };
    let raw = self.evaluate();
    // let raw = unsafe { crate::nnext::NNETWORK.evaluate(self) };
    // let tgt = unsafe { crate::nnext::FNETWORK.evaluate(self) };
    // if (raw - tgt).abs() >= 0.02 { eprintln!("{raw} {tgt} {}", self.to_fen()); }
    return PovScore::new((raw * 100.0).round() as i16);
  }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Op {
  pub square : i8,
  pub piece  : Piece,
}

pub const NOP : Op = Op { square: -1, piece: Piece::Null };

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum NodeKind {
  Unk = 0b_00,
  All = 0b_01,  // upper bound (the score is this or less)
  Cut = 0b_10,  // lower bound (the score is at least this)
  PV  = 0b_11,  // exact score
}

impl NodeKind {
  pub fn pv_or_cut(&self) -> bool
  {
    return (*self as u8) & (NodeKind::Cut as u8) != 0;
  }

  pub fn pv_or_all(&self) -> bool
  {
    return (*self as u8) & (NodeKind::All as u8) != 0;
  }

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

pub const SHORT : usize = 0;
pub const LONG  : usize = 1;

// CASTLE_CONST[color][side]

pub const CASTLE_ROOK : [[u64; 2]; 2] = [
  [0b_1010_0000      , 0b_0000_1001      ],
  [0b_1010_0000 << 56, 0b_0000_1001 << 56],
];
pub const CASTLE_KING : [[u64; 2]; 2] = [
  [0b_0101_0000      , 0b_0001_0100      ],
  [0b_0101_0000 << 56, 0b_0001_0100 << 56],
];
pub const CASTLE_BTWN : [[u64; 2]; 2] = [
  [0b_0110_0000      , 0b_0000_1110      ],
  [0b_0110_0000 << 56, 0b_0000_1110 << 56],
];
pub const CASTLE_CHCK : [[u64; 2]; 2] = [
  [0b_0111_0000      , 0b_0001_1100      ],
  [0b_0111_0000 << 56, 0b_0001_1100 << 56],
];

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

pub trait Algebraic {
  fn id     (&self) -> &'static str;
  fn file_id(&self) -> char;
  fn rank_id(&self) -> char;
}

impl Algebraic for u8 {
  fn id(&self) -> &'static str
  {
    return match *self {
      0..64 => [
        "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
        "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
        "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
        "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
        "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
        "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
        "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
        "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
      ][*self as usize],
      64..255 => "OB",
      255 => "NULL"
    };
  }

  fn file_id(&self) -> char
  {
    return ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'][(self % 8) as usize];
  }

  fn rank_id(&self) -> char
  {
    if *self >= 64 { return '?'; }
    return ['1', '2', '3', '4', '5', '6', '7', '8'][(self / 8) as usize];
  }
}

impl Algebraic for i8 {
  fn id     (&self) -> &'static str { return (*self as u8).id(); }
  fn file_id(&self) -> char         { return (*self as u8).file_id(); }
  fn rank_id(&self) -> char         { return (*self as u8).rank_id(); }
}
