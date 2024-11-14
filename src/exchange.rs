use crate::dest::*;
use crate::color::Color::*;
use crate::misc::Op;
use crate::misc::{
  shift_nw,
  shift_ne,
  shift_sw,
  shift_se
};
use crate::piece::Kind::*;
use crate::piece::Piece::Null;
use crate::state::State;

// A word about exchange analysis  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
//   Exchange analysis is useful for two things:
//
//   • finding captures that appear at first blush to be losing but are actually winning (since
//     the opponent will expose themselves to recapture), and
//
//   • determining the margin of a winning capture (since the opponent may or may not recapture)
//
//   Captures that are winning from the start (capturing a piece with a less valuable one) can
//   never become losing, so exchange analysis is unnecessary for determining whether such
//   captures are winning. It is only needed for determining the margin.

const XCHG_VALUE : [i16; 6] = [200, 100, 50, 30, 30, 10];

impl State {
  // NOTE that the attacking piece must belong to the side to move. The attacker and target must
  //   accurately reflect the state of the board (i.e. there must be a piece of the claimed kind
  //   on the claimed square). The target piece may be a null piece, in which case the target
  //   square must be unoccupied.
  //
  // NOTE that this does not consider pawn promotions.
  //
  // NOTE that this does not consider whether or not any pieces are pinned to a king, and so it
  //   is possible for exchange values to be wildly incorrect!
  //
  // NOTE that this returns a score in decipawns, not centipawns!
  //
  pub fn analyze_exchange(&self, attacker : Op, target : Op) -> i16
  {
    // Suppose the board looks like
    //
    //   .  .  .  .  .  .  .  .  .
    //   .  .  .  .  .  .  .  .  .
    //   .  .  .  .  .  .D1.  .  .
    //   .  .  .  .  .  .  .  .  .
    //   .  .  .  .  (D0)  .  .D2.
    //   .  .  .  .A1.  .  .  .  .
    //   .  .  .  .  .  .  .  .  .
    //   .  .  .  .  .A2.  .  .  .
    //   .  .  .  .  .A3.  .  .  .
    //
    //   Then gain looks like
    //
    //             0   1      2          3       <- index
    //     gain = [0, $D0, $A1-$D0, $D1-$A1+$D0, ...]
    //                atk    def        atk      <- just made a capture
    //                def    atk        def      <- side to move
    //
    //   where $P = absolute material value of piece P

    let target_square = target.square as usize;

    let mut gain : [i16; 16] = [0; 16];

    let mut captive_piece = target.piece;
    let mut captor_piece = attacker.piece;
    let mut captor_location = 1 << attacker.square;

    let mut boards = self.boards;
    let mut occludes = self.sides[White] | self.sides[Black];

    let target_location = 1 << target_square;
    let pawn_map = [
      shift_sw(target_location) | shift_se(target_location),
      shift_nw(target_location) | shift_ne(target_location),
    ];
    let knight_map = knight_destinations(target_square);
    let king_map = king_destinations(target_square);

    occludes ^= captor_location;
    let mut diagonals = bishop_destinations(occludes, target_square);
    let mut straights = rook_destinations(occludes, target_square);

    let mut x = 0;
    let mut captor_color = self.turn;
    'swap_off: loop {
      x += 1;
      // TODO debug assert x < 16
      // TODO non-debug break if x >= 16
      gain[x] = if captive_piece == Null
                { 0 } else { XCHG_VALUE[captive_piece.kind()] } - gain[x-1];
      //
      // NOTE if all we care about is determining whether the exchange is a win,
      //   loss, or equivalent exchange for the initiator, we can add an early
      //   break at this point in the routine:
      //
      //   if gain[x] < 0 && -gain[x-1] < 0 { break; }
      //
      boards[captor_piece] ^= captor_location;
      captive_piece = captor_piece;
      captor_color = !captor_color;
      // Now we set captor_piece and captor_location (least valuable attacker)
      loop {
        // Pawns
        let pawns = pawn_map[captor_color] & boards[captor_color+Pawn];
        if pawns != 0 {
          captor_piece = captor_color+Pawn;
          captor_location = 1 << pawns.trailing_zeros();
          occludes ^= captor_location;
          diagonals = bishop_destinations(occludes, target_square);
          break;
        }
        // Knights
        let knights = knight_map & boards[captor_color+Knight];
        if knights != 0 {
          captor_piece = captor_color+Knight;
          captor_location = 1 << knights.trailing_zeros();
          break;
        }
        // Bishops, Rooks, and Queens
        let bishops = diagonals & boards[captor_color+Bishop];
        if bishops != 0 {
          captor_piece = captor_color+Bishop;
          captor_location = 1 << bishops.trailing_zeros();
          occludes ^= captor_location;
          diagonals = bishop_destinations(occludes, target_square);
          break;
        }
        let rooks = straights & boards[captor_color+Rook];
        if rooks != 0 {
          captor_piece = captor_color+Rook;
          captor_location = 1 << rooks.trailing_zeros();
          occludes ^= captor_location;
          straights = rook_destinations(occludes, target_square);
          break;
        }
        let d_queens = diagonals & boards[captor_color+Queen];
        if d_queens != 0 {
          captor_piece = captor_color+Queen;
          captor_location = 1 << d_queens.trailing_zeros();
          occludes ^= captor_location;
          diagonals = bishop_destinations(occludes, target_square);
          break;
        }
        let s_queens = straights & boards[captor_color+Queen];
        if s_queens != 0 {
          captor_piece = captor_color+Queen;
          captor_location = 1 << s_queens.trailing_zeros();
          occludes ^= captor_location;
          straights = rook_destinations(occludes, target_square);
          break;
        }
        // Kings
        let kings = king_map & boards[captor_color+King];
        if kings != 0 {
          captor_piece = captor_color+King;
          captor_location = 1 << kings.trailing_zeros();
          occludes ^= captor_location;
          diagonals = bishop_destinations(occludes, target_square);
          straights = rook_destinations(occludes, target_square);
          break;
        }
        break 'swap_off;
      }
    }
    while x != 0 {
      gain[x-1] = -std::cmp::max(-gain[x-1], gain[x]);
      x -= 1;
    }
    return gain[1];
  }
}
