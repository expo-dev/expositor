use crate::basis::*;
use crate::color::Color::*;
use crate::color::WB;
use crate::state::State;
use crate::piece::KQRBNP;

impl State {
  pub fn zobrist(&self) -> u64
  {
    let mut key = match self.turn { White => 0, Black => TURN_BASIS };
    for color in WB {
      for kind in KQRBNP {
        let piece = color + kind;
        let mut board = self.boards[piece];
        while board != 0 {
          let square = board.trailing_zeros() as usize;
          let entry = PIECE_BASIS[(piece as usize)*64 + square];
          key ^= entry;
          board &= board - 1;
        }
      }
    }
    if self.rights & 1 != 0 { key ^= CASTLE_BASIS[0] };
    if self.rights & 2 != 0 { key ^= CASTLE_BASIS[1] };
    if self.rights & 4 != 0 { key ^= CASTLE_BASIS[2] };
    if self.rights & 8 != 0 { key ^= CASTLE_BASIS[3] };
    if self.enpass >= 0 {
      key ^= ENPASS_BASIS[(self.enpass % 8) as usize];
    }
    return key;
  }

  pub fn verify_zobrist(&self) -> bool
  {
    let correct = self.zobrist();
    if self.key == correct { return true; }
    eprintln!("error: key desynchronization");
    eprintln!("  self.key       = {:016x}", self.key);
    eprintln!("  self.zobrist() = {:016x}", correct);
    return false;
  }
}

pub fn rezobrist(prev_rights : u8, prev_enpass : i8,
                 next_rights : u8, next_enpass : i8) -> u64
{
  let mut key_diff : u64 = 0;

  if next_rights != prev_rights {
    let rights_diff = prev_rights ^ next_rights;
    if rights_diff & 1 != 0 { key_diff ^= CASTLE_BASIS[0] };
    if rights_diff & 2 != 0 { key_diff ^= CASTLE_BASIS[1] };
    if rights_diff & 4 != 0 { key_diff ^= CASTLE_BASIS[2] };
    if rights_diff & 8 != 0 { key_diff ^= CASTLE_BASIS[3] };
  }
  if next_enpass != prev_enpass {
    if prev_enpass >= 0 {
      key_diff ^= ENPASS_BASIS[(prev_enpass % 8) as usize];
    }
    if next_enpass >= 0 {
      key_diff ^= ENPASS_BASIS[(next_enpass % 8) as usize];
    }
  }
  return key_diff;
}

/* pub fn invert(key : u64) -> State
{
  // NOTE that this function is here just for fun!
  // There are about 4.6 × 10⁴⁴ legal chess positions (see John Tromp's work
  //   estimating this number at github.com/tromp/ChessPositionRanking), which
  //   is fabulously more than the 2⁶⁴ ≈ 1.8 × 10¹⁹ possible Zobrist keys. We'd
  //   expect, then, about 2.5 × 10²⁵ positions to share each key. (It's likely
  //   that nearly all of these are ridiculous, i.e. only a very small fraction
  //   would come up in search trees.) However, the fact that there are many
  //   possible positions with the given key does nothing to help us – if we're
  //   randomly creating positions and keys are in fact distributed evenly, the
  //   expected number of guesses required to find a position with the given key
  //   is still 2⁶³. So we don't try that approach at all!
  // TODO Gaussian elimination
  return State::new();
} */

/*
There are 768 columns, from 'a = WhiteKing@a1 to 'z = BlackPawn@h8.
There are 64 rows (one for each bit of the Zobrist key).
Each column is a basis element.


a  b  c  d  · · ·  z

1  1  0  1  · · ·  1  |  1          a + b     + d  ··· + z = 1
0  1  1  0  · · ·  0  |  1              b + c      ···     = 1
·  ·  ·  ·  ·      1  |  0
·  ·  ·  ·    ·    0  |  1
·  ·  ·  ·      ·  0  |  0
1  0  0  1  · · ·  1  |  0          a         + d  ··· + z = 0

*/
