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
