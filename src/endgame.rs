use crate::color::Color::{self, *};
use crate::misc::{LIGHT_SQUARES, DARK_SQUARES};
use crate::piece::Kind::*;
use crate::piece::Piece::{self, *};
use crate::score::{
  INEVITABLE_MATE,
  INEVITABLE_LOSS,
  LIKELY_MATE,
  LIKELY_LOSS
};
use crate::state::State;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn diff(a : u8, b : u8) -> u8 { return (a as i8 - b as i8).unsigned_abs(); }

// Adjusted manhattan distance [min 2, max 64]
fn manh_dist(a : u8, b : u8) -> i16
{
  let a = (a / 8, a % 8);
  let b = (b / 8, b % 8);
  let x = diff(a.0, b.0) + diff(a.1, b.1);
  return [0, 2, 5, 8, 11, 15, 19, 23, 28, 33, 38, 44, 50, 57, 64][x as usize];
}

fn king_steps(a : u8, b : u8) -> u8
{
  let a = (a / 8, a % 8);
  let b = (b / 8, b % 8);
  return std::cmp::max(diff(a.0, b.0), diff(a.1, b.1));
}

// Adjusted step distance (number of moves for a king) [min 4, max 64]
fn step_dist(a : u8, b : u8) -> i16
{
  return [0, 4, 9, 16, 25, 36, 49, 64][king_steps(a, b) as usize];
}

// Adjusted distance to an edge w/ symmetry breaking [min 0, max 256]
fn edge_dist(x : u8) -> i16
{
  return [
      0,  21,  42,  63,  64,  45,  26,   7,
     23,  84, 105, 126, 127, 108,  89,  30,
     46, 107, 168, 189, 190, 171, 112,  53,
     69, 130, 191, 252, 253, 194, 135,  76,
     72, 133, 194, 255, 256, 197, 138,  79,
     55, 116, 177, 198, 199, 180, 121,  62,
     38,  99, 120, 141, 142, 123, 104,  45,
     21,  42,  63,  84,  85,  66,  47,  28,
  ][x as usize];
}

// Adjusted distance to a dark corner w/ symmetry breaking [min 0, max 500]
fn diag_dist(x : u8) -> i16
{
  return [
      0,  51, 102, 153, 238, 323, 408, 493,
     52,  69, 120, 171, 222, 307, 392, 409,
    104, 121, 138, 189, 240, 291, 308, 325,
    156, 173, 190, 207, 258, 241, 224, 241,
    242, 225, 242, 259, 208, 191, 174, 157,
    328, 311, 294, 243, 192, 141, 124, 107,
    414, 397, 312, 227, 176, 125,  74,  57,
    500, 415, 330, 245, 160, 109,  58,   7,
  ][x as usize];
}

impl State
{
  fn location_of(&self, piece : Piece) -> u8
  {
    debug_assert!(
      self.boards[piece].count_ones() == 1,
      "piece {} is ambiguous in {:016x}", piece.abbrev(), self.boards[piece]
    );
    return self.boards[piece].trailing_zeros() as u8;
  }

  fn location_of_pair(&self, piece : usize) -> (u8, u8)
  {
    debug_assert!(
      self.boards[piece].count_ones() == 2,
      "expected pair of piece {} in {:016x}", piece, self.boards[piece]
    );
    let mut board = self.boards[piece];
    let sq1 = board.trailing_zeros() as u8;
    board &= board - 1;
    let sq2 = board.trailing_zeros() as u8;
    return (sq1, sq2);
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// The generic heuristic covers these cases:
//   K[QR][QRBN] v K
//   KBB v K (opposite color bishops)
//   K3+ v K
fn generic_v_k(def_king : u8, atk_king : u8) -> i16
{
  let king_edge_dist = edge_dist(def_king);
  let king_king_dist = step_dist(def_king, atk_king);
  // +7_11 to +9_99
  return 10_17 - king_edge_dist - (king_king_dist << 1);
}

// K[QR] v K
fn kh_v_k(def_king : u8, atk_king : u8, atk_piece : u8) -> i16
{
  // we need the attacking king to help out
  let king_edge_dist  = edge_dist(def_king);
  let king_king_dist  = step_dist(def_king, atk_king);
  let king_piece_dist = manh_dist(def_king, atk_piece);
  // +6_98 to +9_99
  return 10_18 - king_edge_dist - (king_king_dist << 1) - (king_piece_dist >> 1);
}

// KBN v K
fn kbn_v_k(def_king : u8, atk_king : u8, light_bishop : bool) -> i16
{
  let king_diag_dist = diag_dist(if light_bishop { def_king ^ 7 } else { def_king });
  let king_king_dist = step_dist(def_king, atk_king);
  // +3_89 to +9_99
  return 10_17 - king_diag_dist - (king_king_dist << 1);
}

// KP v K
// Unlike the other functions, this may return
//   scores that are not positively signed:
//     -  unknown outcome
//     0  draw
//     +  inevitable mate
fn kp_v_k(
  mut def_king : u8, mut atk_king : u8, mut atk_pawn : u8, attacker : Color, turn : Color
) -> i16
{
  if attacker == Black {
    def_king ^= 56;
    atk_king ^= 56;
    atk_pawn ^= 56;
  }

  // What I'm calling "the field" is what's normally called "the square";
  //   the boundary just outside the field is "the fence".

  let field_len = 7 - atk_pawn/8;
  let def_king_in_field =
    (def_king/8 >= atk_pawn/8) && (diff(def_king%8, atk_pawn%8) <= field_len);

  // NOTE that this isn't quite right if the king is blocking its own pawn
  //   (it's still a win, but maybe needs to be scored differently)
  if turn == attacker && !def_king_in_field {
    // +1_79 to +4_99
    return 5_63 - ((field_len as i16) << 6);
  }

  let def_king_in_fence =
    (def_king/8 >= atk_pawn/8 - 1) && (diff(def_king%8, atk_pawn%8) <= field_len+1);

  if turn != attacker && !def_king_in_fence {
    // +1_79 to +4_99
    return 5_63 - ((field_len as i16) << 6);
  }

  let atk_king_dst = king_steps(atk_king, atk_pawn);
  let def_king_dst = king_steps(def_king, atk_pawn);

  if atk_king_dst > def_king_dst + if turn == attacker { 1 } else { 0 } { return 0_00; }

  return -1;
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

impl State {
  pub fn endgame(&self) -> Option<i16>
  {
    let num_w_pawns = self.boards[WhitePawn].count_ones();
    let num_b_pawns = self.boards[BlackPawn].count_ones();
    let   all_pawns = num_w_pawns + num_b_pawns;

    if all_pawns > 1 { return None; }

    let w_king_only = self.boards[WhiteKing] == self.sides[White];
    let b_king_only = self.boards[BlackKing] == self.sides[Black];

    if !(w_king_only || b_king_only) { return None; }
    if w_king_only && b_king_only { return Some(0); }

    let atk = if w_king_only { Black } else { White };
    let def = !atk;

    let atk_king = self.location_of(atk+King);
    let def_king = self.location_of(def+King);

    let num_queens  = self.boards[atk+Queen ].count_ones();
    let num_rooks   = self.boards[atk+Rook  ].count_ones();
    let num_bishops = self.boards[atk+Bishop].count_ones();
    let num_knights = self.boards[atk+Knight].count_ones();

    let major = num_queens + num_rooks;
    let minor = num_bishops + num_knights;
    let total = major + minor;

    let side_to_move_atk = self.turn == atk;

    if all_pawns == 0 {
      // These lines aren't necessary because resolving search
      //   will always allow the capture of a hanging piece:
      //
      //   if !side_to_move_atk
      //     && king_destinations(def_king as usize) & self.sides[atk] != 0
      //   { return None; }
      //
      if total == 1 {
        // KQ v K and KR v K
        if major != 0 {
          let atk_piece = self.location_of(if num_rooks == 1 { atk+Rook } else { atk+Queen });
          let score = kh_v_k(def_king, atk_king, atk_piece);
          let score =
            if side_to_move_atk { INEVITABLE_MATE + score } else { INEVITABLE_LOSS - score };
          return Some(score);
        }
        // KB v K and KN v K
        return Some(0);
      }
      if total == 2 {
        // K[QR][QRBN] v K
        if major != 0 {
          let score = generic_v_k(def_king, atk_king);
          let score =
            if side_to_move_atk { INEVITABLE_MATE + score } else { INEVITABLE_LOSS - score };
          return Some(score);
        }
        let light_bishop = (self.boards[atk+Bishop] & LIGHT_SQUARES) != 0;
        let  dark_bishop = (self.boards[atk+Bishop] &  DARK_SQUARES) != 0;
        // KBB v K
        if num_bishops == 2 {
          if !(light_bishop && dark_bishop) { return Some(0); }
          let score = generic_v_k(def_king, atk_king);
          let score =
            if side_to_move_atk { LIKELY_MATE + score } else { LIKELY_LOSS - score };
          return Some(score);
        }
        // KNN v K
        if num_knights == 2 {
          return Some(0);
        }
        // KBN v K
        if major == 0 {
          let score = kbn_v_k(def_king, atk_king, light_bishop);
          let score =
            if side_to_move_atk { LIKELY_MATE + score } else { LIKELY_LOSS - score };
          return Some(score);
        }
      }
      // K3+ v K
      let score = generic_v_k(def_king, atk_king);
      let score =
        if side_to_move_atk { INEVITABLE_MATE + score } else { INEVITABLE_LOSS - score };
      return Some(score);
    }
    if all_pawns == 1 {
      if total > 0 { return None; }
      // KP v K
      let atk_pawn = self.location_of(atk+Pawn);
      let score = kp_v_k(def_king, atk_king, atk_pawn, atk, self.turn);
      if score  < 0 { return None; }
      if score == 0 { return Some(0); }
      if score >  0 {
        let score =
          if side_to_move_atk { INEVITABLE_MATE + score } else { INEVITABLE_LOSS - score };
        return Some(score);
      }
    }
    return None;
  }
}
