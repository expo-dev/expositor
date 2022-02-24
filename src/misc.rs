use crate::color::*;
use crate::dest::*;
use crate::piece::*;
use crate::span::*;
use crate::state::*;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

impl State {
  pub fn evaluate_in_game(&self) -> i16
  {
    if let Some(score) = self.endgame() { return score; }
    let raw = self.evaluate();
    if self.dfz > 10 {
      if self.dfz >= 50 { return 0; }
      return (raw * 2.5 * (50-self.dfz) as f32).round() as i16;
    }
    else {
      return (raw * 100.0).round() as i16;
    }
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Op {
  pub square : i8,
  pub piece  : Piece,
}

pub const NOP : Op = Op { square: -1, piece: Piece::NullPiece };

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum NodeKind {
  Unk = 0b00,
  All = 0b01, // upper bound (the score is this or less)
  Cut = 0b10, // lower bound (the score is at least this)
  PV  = 0b11, // exact score
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[inline] pub fn hmirror(x : usize) -> usize { return x ^  7; }
#[inline] pub fn vmirror(x : usize) -> usize { return x ^ 56; }
#[inline] pub fn  rotate(x : usize) -> usize { return x ^ 63; }

pub const FILE_A : u64 = 0x0101010101010101;
pub const FILE_H : u64 = 0x8080808080808080;

pub const LIGHT_SQUARES : u64 = 0x55aa55aa55aa55aa;
pub const  DARK_SQUARES : u64 = 0xaa55aa55aa55aa55;

#[inline] pub fn shift_nw(board : u64) -> u64 { return (board & !FILE_A) << 7; }
#[inline] pub fn shift_ne(board : u64) -> u64 { return (board & !FILE_H) << 9; }
#[inline] pub fn shift_sw(board : u64) -> u64 { return (board & !FILE_A) >> 9; }
#[inline] pub fn shift_se(board : u64) -> u64 { return (board & !FILE_H) >> 7; }

#[inline]
pub fn pawn_attacks(color : Color, sources : u64) -> u64
{
  return match color {
    Color::White => shift_nw(sources) | shift_ne(sources),
    Color::Black => shift_sw(sources) | shift_se(sources),
  };
}

#[inline]
pub fn piece_destinations(piece : usize, src : usize, composite : u64) -> u64
{
  return match piece {
    KING   =>   king_destinations(           src),
    QUEEN  =>  queen_destinations(composite, src),
    ROOK   =>   rook_destinations(composite, src),
    BISHOP => bishop_destinations(composite, src),
    KNIGHT => knight_destinations(           src),
    _      => 0
  };
}

#[inline]
pub fn crux_span(a : usize, b : usize) -> u64 // cruciform span
{
  return if a >= b { CRUX_SPAN[b][a] } else { CRUX_SPAN[a][b] };
  // TODO write this as follows instead?
  //   return CRUX_SPAN[a][b];
  // TODO write this as follows instead?
  //   let rank_a = a / 8; let file_a = a % 8;
  //   let rank_b = b / 8; let file_b = b % 8;
  //   if rank_a == rank_b { return (RANK_SPAN[file_a][file_b] as u64) << (a & !7); }
  //   if file_a == file_b { return  FILE_SPAN[rank_a][rank_b]         <<  file_a;  }
  //   return 0;
}

#[inline]
pub fn salt_span(a : usize, b : usize) -> u64 // saltire span
{
  return if a >= b { SALT_SPAN[b][a] } else { SALT_SPAN[a][b] };
  // TODO write this as follows instead?
  //   return SALT_SPAN[a][b];
}

#[inline]
pub fn any_span(a : usize, b : usize) -> u64
{
  return if a >= b { ANY_SPAN[b][a] } else { ANY_SPAN[a][b] };
  // TODO write this as follows instead?
  //   return ANY_SPAN[a][b];
  // TODO write this as follows instead?
  //   let rank_a = a / 8; let file_a = a % 8;
  //   let rank_b = b / 8; let file_b = b % 8;
  //   if rank_a == rank_b { return (RANK_SPAN[file_a][file_b] as u64) << (a & !7); }
  //   if file_a == file_b { return  FILE_SPAN[rank_a][rank_b]         <<  file_a;  }
  //   return salt_span(a, b);
}

#[inline]
pub fn line_through(a : usize, b : usize) -> u64
{
  return if a >= b { ANY_LINE[b][a] } else { ANY_LINE[a][b] };
  // TODO write this as follows instead?
  //   return ANY_LINE[a][b];
  // TODO write this like crux_span? (calculating vertical and horizontal
  //   lines rather than performing a lookup in those cases)
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

impl State {
  pub fn in_check(&self, color : Color) -> bool
  {
    let kdx = color as usize * 8;
    let king_board = self.boards[kdx];
    let king_square = king_board.trailing_zeros() as usize;

    let composite = self.sides[W] | self.sides[B];
    let opp_boards = match color {
      Color::White => &self.boards[8..15],
      Color::Black => &self.boards[0..7]
    };

    let qr = opp_boards[ROOK] | opp_boards[QUEEN];
    if qr & rook_destinations(composite, king_square) != 0 { return true; }

    let qb = opp_boards[BISHOP] | opp_boards[QUEEN];
    if qb & bishop_destinations(composite, king_square) != 0 { return true; }

    if opp_boards[KNIGHT] & knight_destinations(king_square) != 0 { return true; }

    let pawn_attacks = match color {
      Color::White => shift_nw(king_board) | shift_ne(king_board),
      Color::Black => shift_sw(king_board) | shift_se(king_board),
    };
    if pawn_attacks & opp_boards[PAWN] != 0 { return true; }

    if king_destinations(king_square) & opp_boards[KING] != 0 { return true; }

    return false;
  }

  pub fn attacked_by(&self, color : Color, composite : u64) -> u64
  {
    let player = color as usize * 8;
    let mut attacked : u64 = 0;
    for piece in 0..5 {
      let mut sources = self.boards[player+piece];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let destinations = piece_destinations(piece, src, composite);
        attacked |= destinations;
        sources &= sources - 1;
      }
    }
    return attacked | pawn_attacks(color, self.boards[player+PAWN]);
  }

  pub fn attackers_of(&self, square : usize, color : Color) -> u64
  {
    let composite = self.sides[W] | self.sides[B];
    let boards = match color {
      Color::White => &self.boards[0..7],
      Color::Black => &self.boards[8..15]
    };
    return ((boards[ROOK]   | boards[QUEEN]) &   rook_destinations(composite, square))
         | ((boards[BISHOP] | boards[QUEEN]) & bishop_destinations(composite, square))
         | ( boards[KNIGHT]                  & knight_destinations(           square))
         | ( boards[PAWN]                    &   pawn_attacks(!color, 1u64 << square));
  }
}
