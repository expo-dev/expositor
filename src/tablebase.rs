use crate::color::*;
use crate::dest::*;
use crate::misc::*;
use crate::movegen::*;
use crate::movetype::*;
use crate::piece::*;
use crate::score::*;
use crate::state::*;

use std::mem::MaybeUninit;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
// NOTE that this 4-man tablebase does NOT account for captures en passant!
//   To prevent the engine from making a disastrous double push, as in e.g.
//     8/8/8/8/2p4k/8/1P4K1/8 w - - 0 1,
//   it is critical that the tablebase is never probed in KPvKP endgames where
//   one of the pawns is on its home row or a capture en passant is possible!

// The 4-man tablebase is indexed in two steps. The first step picks out the
//   endgame subtable:
//
//   table[s][x][y] : 2 × 5 × 5
//
//   s  0 KXYvK, 1 KXvKY
//   x  piece X (0–4 for queen–pawn)
//   y  piece Y (0–4 for queen–pawn)
//
//   with the constraint that y >= x, i.e. Y is the same piece as X or weaker.
//   There are then 2 × (5+4+3+2+1) = 30 subtables.
//
// The second step picks out the position:
//
//   subtable[K][X][Y][k][s] : 64 × 64 × 64 × 64 × 2
//
//   K  attacking king square (0–63)
//   X  piece X square        (0–63)
//   Y  piece Y square        (0–63)
//   k  defending king square (0–63)
//   s  side to move (0 when side with more valuable material, 1 otherwise)
//
// We reflect horizontally to put the attacking king on the left side of the
//   board, which lowers this to
//
//   subtable[K][X][Y][k][s] : 32 × 64 × 64 × 64 × 2
//
//   K  attacking king index (0–31 for square 0–3, 8–11, ...)
//
// There are then 32 × 64 × 64 × 64 × 2 = 16 777 216 entries per subtable.
//
// For pawnless endgames, we could also reflect vertically to put the attacking
//   king in the bottom left, which would halve the size of the table, but we
//   don't bother for the sake of uniformity and simplicity.
//
// Altogether, the 4-man tables take up 30 × 16 MB = 480 MB of resident pages.

/* ↓↓↓ INCOMPLETE ↓↓↓ //
pub type Subtable  = [[[[[u8; 2]; 64]; 64]; 64]; 32];
pub type Tablebase = [[[Subtable; 5]; 5]; 2];
// ↑↑↑ INCOMPLETE ↑↑↑ */

// Thanks to https://github.com/rust-lang/rust/issues/55795, each of the
//   following declarations cause compilation to take a veritable eternity:
//
//   use std::mem::{zeroed, MaybeUninit};
//
//   pub static mut Tb : Tablebase = [[[[[[[[0;2];64];64];64];32];5];5];2];
//   pub static mut Tb : Tablebase = unsafe { zeroed() };
//   pub static mut Tb : Tablebase = unsafe { MaybeUninit::zeroed().assume_init() };
//   pub static mut Tb : Tablebase = unsafe { MaybeUninit::uninit().assume_init() };
//   pub static mut Tb : Tablebase = unsafe { transmute::<MaybeUninit<Tablebase>,_>(MaybeUninit::uninit()) };
//
// Rust is very frustrating sometimes.
//
// So instead we declare it
//
//   pub static mut Tb : MaybeUninit<Tablebase> = MaybeUninit::uninit();
//
// and do one of the following instead:
//
//   pub fn modify() {
//     let tb = unsafe { std::mem::transmute::<_,&mut Tablebase>(&mut Tb) };
//     tb[0][0][0][0][0][0][0][0] = 1;
//   }
//   pub fn modify() {
//     let tb = unsafe { Tb.assume_init_mut() };
//     tb[0][0][0][0][0][0][0][0] = 1;
//   }
//
// Note that
//
//   let mut tb = unsafe { std::mem::transmute::<_,Tablebase>(Tb) };
//   tb[0][0][0][0][0][0][0][0] = 1;
//
// will not work; no write is emitted by the compiler. (I believe this is
//   because the semantics of the code are "copy Tb to the stack, write 1,
//   and drop the new table", which the compiler optimizes to a nop.)
//
// To read the table, we use one of the following:
//
//   let tb = unsafe { std::mem::transmute::<_,&Tablebase>(&Tb) };
//   let tb = unsafe { Tb.assume_init_ref() };
//
// and then you can grab subtables like so:
//
//   let tb = unsafe { Tb.assume_init_mut() };
//   let subtable : &mut Subtable = &mut tb[0][0][0];
//
// Note that
//
//   let tb = unsafe { std::mem::transmute::<_,Tablebase>(Tb) };
//   let tb = unsafe { Tb.assume_init() };
//
// will not work; the program will attempt to copy the entire table onto the
//   stack, causing a stack overflow.

/* ↓↓↓ INCOMPLETE ↓↓↓ //
pub static mut Tb : MaybeUninit<Tablebase> = MaybeUninit::uninit();
// ↑↑↑ INCOMPLETE ↑↑↑ */

// The 3-man tablebase is similar, just smaller:
//
//   table[x] : 5
//
//   x  piece X (0–4 for queen–pawn)
//
// And each subtable:
//
//   subtable[K][X][k][side] : 32 × 64 × 64 × 2
//
//   K  attacking king index (0–31 for square 0–3, 8–11, ...)
//   X  piece X square        (0–63)
//   k  defending king square (0–63)

pub type MiniSubtable  = [[[[u8; 2]; 64]; 64]; 32];
pub type MiniTablebase = [MiniSubtable; 5];
pub static mut MiniTb : MaybeUninit<MiniTablebase> = MaybeUninit::uninit();

// Each tablebase entry is one byte layed out like so:
//
//   xx......   00 illegal, 01 draw, 10 side to move lost, 11 side to move won
//   ..dddddd   depth to mate
//
// Within the 4 man tablebase, there are no mates that take longer than 100 ply.
// [http://kirill-kryukov.com/chess/longest-checkmates/longest-checkmates.shtml]

fn legal(entry : u8) -> bool { return (entry >> 6) != 0; }
fn drawn(entry : u8) -> bool { return (entry >> 6) == 1; }
fn  lost(entry : u8) -> bool { return (entry >> 6) == 2; }
fn   won(entry : u8) -> bool { return (entry >> 6) == 3; }
fn   dtm(entry : u8) -> u8   { return entry & 0x3F; }

pub fn idx2sq(x : usize) -> usize {
  // Maps 0–31 to the left half of the board
  return (x/4)*8 + (x%4);
}

pub fn sq2idx(x : usize) -> usize {
  // Maps left half of the board to 0–31
  return (x/8)*4 + (x%4);
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn build_3man_subtable(x : usize) {
  let tb = unsafe { MiniTb.assume_init_mut() };
  // This is unsavory, but we really do need references (one of them mutable)
  //   to two different elements of the table.
  let subtable : &mut MiniSubtable = unsafe { &mut *(tb.get_unchecked_mut(x) as *mut _)};

  // First we go through and mark whether each position is legal. For a position
  //   to be legal, pawns can't be on the last rank, pieces can't be on top of
  //   each other, and the side to move cannot already be giving check.

  for akx in 0..32 {      // attacking king index
    for aps in 0..64 {    // attacking piece square
      for dks in 0..64 {  // defending king square
        let aks = idx2sq(akx);
        if aks == aps || aks == dks || aps == dks { continue; }
        let composite : u64 = (1 << aks) | (1 << aps) | (1 << dks);
        // Side with piece to move
        loop {
          if x == 4 {
            if aps >= 56 { break; }
            if pawn_attacks(Color::White, 1 << aps) & (1 << dks) != 0 { break; }
          }
          else {
            if piece_destinations(x+1, aps, composite) & (1 << dks) != 0 { break; }
          }
          if king_destinations(aks) & (1 << dks) != 0 { break; }
          subtable[akx][aps][dks][0] = 0x40; // drawn by default
          break;
        }
        // Side without piece to move
        if king_destinations(dks) & (1 << aks) == 0 { subtable[akx][aps][dks][1] = 0x40; }
      }
    }
  }

  // We can stop here for KBvK and KNvK endgames, since in those endgames all
  //   positions are either illegal or drawn.
  if x == 2 || x == 3 { return; }

  // Then we determine which positions are mate.
  for akx in 0..32 {
    for aps in 0..64 {
      for dks in 0..64 {
        // We don't consider whether the side with the piece is being mated,
        //   since the lone king cannot mate, so we skip straight to considering
        //   the side without the piece is being mated.
        if legal(subtable[akx][aps][dks][1]) {
          let aks = idx2sq(akx);
          let composite : u64 = (1 << aks) | (1 << aps) | (1 << dks);
         'mated: loop {
            // Stalemates are just draws, so we don't bother testing for them;
            //   we only consider checkmates (the king must be in check).
            if x == 4 {
              if pawn_attacks(Color::White, 1 << aps) & (1 << dks) == 0 { break; }
            }
            else {
              if piece_destinations(x+1, aps, composite) & (1 << dks) == 0 { break; }
            }

            // If you can safely take the piece, it's not mate. (We can view
            //   this as a transition into the KvK endgame, for which all
            //   positions are either illegal or drawn.)
            let mut dests = king_destinations(dks);
            if (dests & (1 << aps) != 0)
              && (king_destinations(aks) & (1 << aps) == 0) { break; }

            // Can the king evade check?
            while dests != 0 {
              let dst = dests.trailing_zeros() as usize;
              if legal(subtable[akx][aps][dst][0]) { break 'mated; }
              dests &= dests - 1;
            }
            subtable[akx][aps][dks][1] = 0x80;
            break;
          }
        }
      }
    }
  }

  // Now we perform retrograde analysis.
  loop {
    let mut modified = false;

    // On odd ply, we find positions where the side with the piece can force
    //   mate. The side with the piece gets to pick a mate if it exists (and
    //   scores this position as the shortest mate + 1).
    for akx in 0..32 {
      for aps in 0..64 {
        for dks in 0..64 {
          let current = subtable[akx][aps][dks][0];
          if !legal(current) { continue; }
          let mut best : u8 = 0x40;

          // King moves
          let aks = idx2sq(akx);
          let mut dests = king_destinations(aks);
          while dests != 0 {
            let dst = dests.trailing_zeros() as usize;
            let fdst; let faps; let fdks;
            if dst % 8 >= 4 {
              fdst = hmirror(dst); faps = hmirror(aps); fdks = hmirror(dks);
            }
            else {
              fdst = dst; faps = aps; fdks = dks;
            }
            let entry = subtable[sq2idx(fdst)][faps][fdks][1];
            if lost(entry) { best = std::cmp::min(best, dtm(entry)); }
            dests &= dests - 1;
          }

          // Pawn moves
          let composite : u64 = (1 << aks) | (1 << aps) | (1 << dks);
          if x == 4 {
            let step = aps + 8;
            if (1 << step) & composite == 0 {
              if aps / 8 == 6 {
                for p in 0..2 {
                  let promotiontable : &mut MiniSubtable =
                    unsafe { &mut *(tb.get_unchecked_mut(p) as *mut _) };
                  let entry = promotiontable[akx][step][dks][1];
                  if lost(entry) { best = std::cmp::min(best, dtm(entry)); }
                }
              }
              else {
                let entry = subtable[akx][step][dks][1];
                if lost(entry) { best = std::cmp::min(best, dtm(entry)); }

                let doublestep = step + 8;
                if aps / 8 == 1 && ((1 << doublestep) & composite == 0) {
                  let entry = subtable[akx][doublestep][dks][1];
                  if lost(entry) { best = std::cmp::min(best, dtm(entry)); }
                }
              }
            }
          }
          // Piece moves
          else {
            let mut dests = piece_destinations(x+1, aps, composite);
            while dests != 0 {
              let dst = dests.trailing_zeros() as usize;
              let entry = subtable[akx][dst][dks][1];
              if lost(entry) { best = std::cmp::min(best, dtm(entry)); }
              dests &= dests - 1;
            }
          }

          // Update the entry
          if best < 0x40 {
            let update = 0xC0 + best + 1;
            if update != current {
              subtable[akx][aps][dks][0] = update;
              modified = true;
            }
          }
        } // dks
      } // aps
    } // akx

    // On even ply we find positions where the side without the piece cannot
    //   prevent mate. The side without the piece gets to pick a draw if one
    //   exists.
    for akx in 0..32 {
      for aps in 0..64 {
        for dks in 0..64 {
          let current = subtable[akx][aps][dks][1];
          if !legal(current) { continue; }
          let mut best : u8 = 0;
          let mut dests = king_destinations(dks);

          let aks = idx2sq(akx);
          if (dests & (1 << aps) != 0)
            && (king_destinations(aks) & (1 << aps) == 0) { continue; }

          while dests != 0 {
            let dst = dests.trailing_zeros() as usize;
            let entry = subtable[akx][aps][dst][0];
            if drawn(entry) {
              best = 0; break;
            }
            else if won(entry) {
              best = std::cmp::max(best, dtm(entry));
            }
            dests &= dests - 1;
          }
          if best > 0 {
            let update = 0x80 + best + 1;
            if update != current {
              subtable[akx][aps][dks][1] = update;
              modified = true;
            }
          }
        } // dks
      } // aps
    } // akx

    if !modified { break; }
  }
}

pub fn build_3man() {
  // It's important that we build the pawn subtable last
  for x in 0..5 { build_3man_subtable(x); }
}

pub fn probe_3man(state : &State, height : i16) -> i16 {
  // NOTE that this should only be called once we've
  //   checked there are in fact three men on the board.

  let black_winning = state.sides[W].count_ones() == 1;

  let white_king = state.boards[WHITE+KING].trailing_zeros() as usize;
  let black_king = state.boards[BLACK+KING].trailing_zeros() as usize;
  let mut atk_king = if black_winning { black_king } else { white_king };
  let mut def_king = if black_winning { white_king } else { black_king };

  let composite = state.sides[W] | state.sides[B];
  let piece_board = composite ^ (state.boards[WHITE+KING] | state.boards[BLACK+KING]);
  let mut atk_piece = piece_board.trailing_zeros() as usize;

  let piece_type = state.squares[atk_piece].kind();
  let atk_to_move = (state.turn as usize) ^ (black_winning as usize);

  if atk_king % 8 >= 4 {
    atk_king  = hmirror(atk_king);
    atk_piece = hmirror(atk_piece);
    def_king  = hmirror(def_king);
  }

  // Within the 3-man tablebase, pawns always move upward,
  //   so we flip vertically when black has the pawn.
  if black_winning {
    atk_king  = vmirror(atk_king);
    atk_piece = vmirror(atk_piece);
    def_king  = vmirror(def_king);
  }

  atk_king = sq2idx(atk_king);

  let tb = unsafe { MiniTb.assume_init_mut() };
  let subtable : &mut MiniSubtable = &mut tb[piece_type-1];
  let entry = subtable[atk_king][atk_piece][def_king][atk_to_move];
  let dtm = (entry & 0x3F) as i16;
  return match entry >> 6 {
    1 => 0,
    2 => PROVEN_LOSS + height + dtm,
    3 => PROVEN_MATE - height - dtm,
    _ => unreachable!()
  };
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
/* ↓↓↓ INCOMPLETE ↓↓↓ //

#[allow(non_snake_case)]
fn build_KXYvK_subtable(x : usize, y : usize) {
  eprintln!("K{}{}vK", ['Q', 'R', 'B', 'N', 'P'][x], ['Q', 'R', 'B', 'N', 'P'][y]);

  let mini_tb = unsafe { MiniTb.assume_init_mut() };
  let tb = unsafe { Tb.assume_init_mut() };
  let subtable : &mut Subtable = unsafe { &mut *(&mut tb[0][x][y] as *mut _)};

  // First we go through and mark whether each position is legal. For a position
  //   to be legal, pawns can't be on the last rank, pieces can't be on top of
  //   each other, and the side to move cannot already be giving check.

  for akx in 0..32 {        // attacking king index
    for aps1 in 0..64 {     // attacking piece square
      for aps2 in 0..64 {   // attacking piece square
        for dks in 0..64 {  // defending king square
          let aks = idx2sq(akx);
          if aks  == aps1 || aks  == aps2 || aks == dks { continue; }
          if aps1 == aps2 || aps1 == dks  { continue; }
          if aps2 == dks  { continue; }
          let composite : u64 = (1 << aks) | (1 << aps1) | (1 << aps2) | (1 << dks);
          // Side with pieces to move
          loop {
            if x == 4 {
              if aps1 >= 56 { break; }
              if pawn_attacks(Color::White, 1 << aps1) & (1 << dks) != 0 { break; }
            }
            else {
              if piece_destinations(x+1, aps1, composite) & (1 << dks) != 0 { break; }
            }
            if y == 4 {
              if aps2 >= 56 { break; }
              if pawn_attacks(Color::White, 1 << aps2) & (1 << dks) != 0 { break; }
            }
            else {
              if piece_destinations(y+1, aps2, composite) & (1 << dks) != 0 { break; }
            }
            if king_destinations(aks) & (1 << dks) != 0 { break; }
            subtable[akx][aps1][aps2][dks][0] = 0x40; // drawn by default
            break;
          }
          // Side without pieces to move
          if king_destinations(dks) & (1 << aks) == 0 {
            subtable[akx][aps1][aps2][dks][1] = 0x40;
          }
        }
      }
    }
  }

  // Then we determine which positions are mate.
  for akx in 0..32 {
    for aps1 in 0..64 {
      for aps2 in 0..64 {
        for dks in 0..64 {
          // Is the side without pieces mated?
          if legal(subtable[akx][aps1][aps2][dks][1]) {
            let aks = idx2sq(akx);
            let composite : u64 = (1 << aks) | (1 << aps1) | (1 << aps2) | (1 << dks);
           'mated: loop {
              // The king must be in check.
              let atk1 = if x == 4 { pawn_attacks(Color::White, 1 << aps1)    }
                         else      { piece_destinations(x+1, aps1, composite) };
              if atk1 & (1 << dks) == 0 {
                let atk2 = if y == 4 { pawn_attacks(Color::White, 1 << aps2)    }
                           else      { piece_destinations(y+1, aps2, composite) };
                if atk2 & (1 << dks) == 0 { break; }
              }

              // If you can safely take the checking piece with the king, or
              //   evade the check, it's not mate.
              let mut dests = king_destinations(dks);
              while dests != 0 {
                let dst = dests.trailing_zeros() as usize;
                if dst == aps1 {
                  let mini_subtable : &mut MiniSubtable =
                    unsafe { &mut *(mini_tb.get_unchecked_mut(y) as *mut _) };
                  if legal(mini_subtable[akx][aps2][dst][0]) { break 'mated; }
                }
                else if dst == aps2 {
                  let mini_subtable : &mut MiniSubtable =
                    unsafe { &mut *(mini_tb.get_unchecked_mut(x) as *mut _) };
                  if legal(mini_subtable[akx][aps1][dst][0]) { break 'mated; }
                }
                else {
                  if legal(subtable[akx][aps1][aps2][dst][0]) { break 'mated; }
                }
                dests &= dests - 1;
              }
              subtable[akx][aps1][aps2][dks][1] = 0x80;
              break;
            }
          }
        }
      }
    }
  }

  // Now we perform retrograde analysis.
  loop {
    let mut modified = false;

    // On odd ply, we find positions where the side with pieces can force mate.
    //   The side with the piece gets to pick a mate if it exists (and scores
    //   this position as the shortest mate + 1).
    for akx in 0..32 {
      for aps1 in 0..64 {
        for aps2 in 0..64 {
          for dks in 0..64 {
            let current = subtable[akx][aps1][aps2][dks][0];
            if !legal(current) { continue; }
            let mut best : u8 = 0x40;

            // King moves
            let aks = idx2sq(akx);
            let mut dests = king_destinations(aks);
            while dests != 0 {
              let dst = dests.trailing_zeros() as usize;
              let fdst; let faps1; let faps2; let fdks;
              if dst % 8 >= 4 {
                fdst = hmirror(dst); faps1 = hmirror(aps1);
                faps2 = hmirror(aps2); fdks = hmirror(dks);
              }
              else {
                fdst = dst; faps1 = aps1; faps2 = aps2; fdks = dks;
              }
              let entry = subtable[sq2idx(fdst)][faps1][faps2][fdks][1];
              if lost(entry) { best = std::cmp::min(best, dtm(entry)); }
              dests &= dests - 1;
            }

            // Pawn 1 moves
            let composite : u64 = (1 << aks) | (1 << aps1) | (1 << aps2) | (1 << dks);
            if x == 4 {
              let step = aps1 + 8;
              if (1 << step) & composite == 0 {
                if aps1 / 8 == 6 {
                  for p in 0..4 {
                    let entry;
                    if y < p {
                      let promotiontable : &mut Subtable =
                        unsafe { &mut *(&mut tb[0][y][p] as *mut _) };
                      entry = promotiontable[akx][aps2][step][dks][1];
                    }
                    else {
                      let promotiontable : &mut Subtable =
                        unsafe { &mut *(&mut tb[0][p][y] as *mut _) };
                      entry = promotiontable[akx][step][aps2][dks][1];
                    }
                    if lost(entry) { best = std::cmp::min(best, dtm(entry)); }
                  }
                }
                else {
                  let entry = subtable[akx][step][aps2][dks][1];
                  if lost(entry) { best = std::cmp::min(best, dtm(entry)); }

                  let doublestep = step + 8;
                  if aps1 / 8 == 1 && ((1 << doublestep) & composite == 0) {
                    let entry = subtable[akx][doublestep][aps2][dks][1];
                    if lost(entry) { best = std::cmp::min(best, dtm(entry)); }
                  }
                }
              }
            }
            // Piece 1 moves
            else {
              let mut dests = piece_destinations(x+1, aps1, composite);
              while dests != 0 {
                let dst = dests.trailing_zeros() as usize;
                let entry = subtable[akx][dst][aps2][dks][1];
                if lost(entry) { best = std::cmp::min(best, dtm(entry)); }
                dests &= dests - 1;
              }
            }

            // Pawn 2 moves
            if y == 4 {
              let step = aps2 + 8;
              if (1 << step) & composite == 0 {
                if aps2 / 8 == 6 {
                  for p in 0..4 {
                    let entry;
                    if p < x {
                      let promotiontable : &mut Subtable =
                        unsafe { &mut *(&mut tb[0][p][x] as *mut _) };
                      entry = promotiontable[akx][step][aps1][dks][1];
                    }
                    else {
                      let promotiontable : &mut Subtable =
                        unsafe { &mut *(&mut tb[0][x][p] as *mut _) };
                      entry = promotiontable[akx][aps1][step][dks][1];
                    }
                    if lost(entry) { best = std::cmp::min(best, dtm(entry)); }
                  }
                }
                else {
                  let entry = subtable[akx][aps1][step][dks][1];
                  if lost(entry) { best = std::cmp::min(best, dtm(entry)); }

                  let doublestep = step + 8;
                  if aps2 / 8 == 1 && ((1 << doublestep) & composite == 0) {
                    let entry = subtable[akx][aps1][doublestep][dks][1];
                    if lost(entry) { best = std::cmp::min(best, dtm(entry)); }
                  }
                }
              }
            }
            // Piece 2 moves
            else {
              let mut dests = piece_destinations(y+1, aps2, composite);
              while dests != 0 {
                let dst = dests.trailing_zeros() as usize;
                let entry = subtable[akx][aps1][dst][dks][1];
                if lost(entry) { best = std::cmp::min(best, dtm(entry)); }
                dests &= dests - 1;
              }
            }

            // Update the entry
            if best < 0x40 {
              let update = 0xC0 + best + 1;
              if update != current {
                subtable[akx][aps1][aps2][dks][0] = update;
                modified = true;
              }
            }
          } // dks
        } // aps2
      } // aps1
    } // akx

    // On even ply we find positions where the side without pieces cannot
    //   prevent mate. The side without the piece gets to pick a draw if one
    //   exists.
    for akx in 0..32 {
      for aps1 in 0..64 {
        for aps2 in 0..64 {
          for dks in 0..64 {
            let current = subtable[akx][aps1][aps2][dks][1];
            if !legal(current) { continue; }
            let mut best : u8 = 0;
            let mut dests = king_destinations(dks);

            while dests != 0 {
              let dst = dests.trailing_zeros() as usize;

              let entry;
              if dst == aps1 {
                let mini_subtable : &mut MiniSubtable =
                  unsafe { &mut *(mini_tb.get_unchecked_mut(y) as *mut _) };
                entry = mini_subtable[akx][aps2][dst][0];
              }
              else if dst == aps2 {
                let mini_subtable : &mut MiniSubtable =
                  unsafe { &mut *(mini_tb.get_unchecked_mut(x) as *mut _) };
                entry = mini_subtable[akx][aps1][dst][0];
              }
              else {
                entry = subtable[akx][aps1][aps2][dst][0];
              }

              if drawn(entry) {
                best = 0; break;
              }
              else if won(entry) {
                best = std::cmp::max(best, dtm(entry));
              }
              dests &= dests - 1;
            }
            if best > 0 {
              let update = 0x80 + best + 1;
              if update != current {
                subtable[akx][aps1][aps2][dks][1] = update;
                modified = true;
              }
            }
          } // dks
        } // aps2
      } // aps1
    } // akx

    if !modified { break; }
  }
}

#[allow(non_snake_case)]
fn build_KXvKY_subtable(x : usize, y : usize) {
  eprintln!("K{}vK{}", ['Q', 'R', 'B', 'N', 'P'][x], ['Q', 'R', 'B', 'N', 'P'][y]);

  let mini_tb = unsafe { MiniTb.assume_init_mut() };
  let tb = unsafe { Tb.assume_init_mut() };
  let subtable : &mut Subtable = unsafe { &mut *(&mut tb[1][x][y] as *mut _)};

  // First we go through and mark whether each position is legal. For a position
  //   to be legal, pawns can't be on the last rank, pieces can't be on top of
  //   each other, and the side to move cannot already be giving check.

  for akx in 0..32 {        // attacking king index
    for aps in 0..64 {      // attacking piece square
      for dps in 0..64 {    // defending piece square
        for dks in 0..64 {  // defending king square
          let aks = idx2sq(akx);
          if aks == aps || aks == dps || aks == dks { continue; }
          if aps == dps || aps == dks { continue; }
          if dps == dks { continue; }
          let composite : u64 = (1 << aks) | (1 << aps) | (1 << dps) | (1 << dks);
          // Stronger side to move
          loop {
            if x == 4 {
              if aps >= 56 { break; }
              if pawn_attacks(Color::White, 1 << aps) & (1 << dks) != 0 { break; }
            }
            else {
              if piece_destinations(x+1, aps, composite) & (1 << dks) != 0 { break; }
            }
            if king_destinations(aks) & (1 << dks) != 0 { break; }
            subtable[akx][aps][dps][dks][0] = 0x40; // drawn by default
            break;
          }
          // Weaker side to move
          loop {
            if y == 4 {
              if dps < 8 { break; }
              if pawn_attacks(Color::Black, 1 << dps) & (1 << aks) != 0 { break; }
            }
            else {
              if piece_destinations(y+1, dps, composite) & (1 << aks) != 0 { break; }
            }
            if king_destinations(dks) & (1 << aks) != 0 { break; }
            subtable[akx][aps][dps][dks][1] = 0x40;
            break;
          }
        }
      }
    }
  }

  // Then we determine which positions are mate.
  for akx in 0..32 {
    for aps in 0..64 {
      for dps in 0..64 {
        for dks in 0..64 {
          let aks = idx2sq(akx);
          let composite : u64 = (1 << aks) | (1 << aps) | (1 << dps) | (1 << dks);

          // Is the stronger side mated?
          if legal(subtable[akx][aps][dps][dks][0]) {
           'mated: loop {
              // The king must be in check.
              if y == 4 {
                if pawn_attacks(Color::Black, 1 << dps) & (1 << aks) == 0 { break; }
              }
              else {
                if piece_destinations(y+1, dps, composite) & (1 << aks) == 0 { break; }
              }

              // If you can safely take the piece with the king, it's not mate.
              let mut dests = king_destinations(aks);
              if (dests & (1 << dps) != 0)
                && (king_destinations(dks) & (1 << dps) == 0) { break; }

              // Can the king evade check?
              while dests != 0 {
                let dst = dests.trailing_zeros() as usize;
                let fdst; let faps; let fdps; let fdks;
                if dst % 8 >= 4 {
                  fdst = hmirror(dst); faps = hmirror(aps);
                  fdps = hmirror(dps); fdks = hmirror(dks);
                }
                else {
                  fdst = dst; faps = aps; fdps = dps; fdks = dks;
                }
                if legal(subtable[sq2idx(fdst)][faps][fdps][fdks][1]) { break 'mated; }
                dests &= dests - 1;
              }

              // If you can trade pieces or simply win the piece, it's not mate.
              if x == 4 {
                if pawn_attacks(Color::White, 1 << aps) & (1 << dps) != 0 { break; }
              }
              else {
                if piece_destinations(x+1, aps, composite) & (1 << dps) != 0 { break; }
              }

              // Can the check be blocked? There's no need to check for pins,
              //   since there's no other piece that can be attacking.
              if y < 3 {
                let span = any_span(aks, dps);
                if x == 4 {
                  if ((1 << aps) << 8) & span != 0 { break; }
                }
                else {
                  if piece_destinations(x+1, aps, composite) & span != 0 { break; }
                }
              }

              subtable[akx][aps][dps][dks][0] = 0x80;
              break;
            }
          }

          // Is the side with the weaker piece mated?
          if legal(subtable[akx][aps][dps][dks][1]) {
           'mated: loop {
              // The king must be in check.
              if x == 4 {
                if pawn_attacks(Color::White, 1 << aps) & (1 << dks) == 0 { break; }
              }
              else {
                if piece_destinations(x+1, aps, composite) & (1 << dks) == 0 { break; }
              }

              // If you can safely take the piece with the king, it's not mate.
              let mut dests = king_destinations(dks);
              if (dests & (1 << aps) != 0)
                && (king_destinations(aks) & (1 << aps) == 0) { break; }

              // Can the king evade check?
              while dests != 0 {
                let dst = dests.trailing_zeros() as usize;
                if legal(subtable[akx][aps][dps][dst][0]) { break 'mated; }
                dests &= dests - 1;
              }

              // If you can trade pieces or simply win the piece, it's not mate.
              if y == 4 {
                if pawn_attacks(Color::Black, 1 << dps) & (1 << aps) != 0 { break; }
              }
              else {
                if piece_destinations(y+1, dps, composite) & (1 << aps) != 0 { break; }
              }

              // Can the check be blocked? There's no need to check for pins,
              //   since there's no other piece that can be attacking.
              if x < 3 {
                let span = any_span(dks, aps);
                if y == 4 {
                  if ((1 << dps) >> 8) & span != 0 { break; }
                }
                else {
                  if piece_destinations(y+1, dps, composite) & span != 0 { break; }
                }
              }

              subtable[akx][aps][dps][dks][1] = 0x80;
              break;
            }
          }
        }
      }
    }
  }

  // Now we perform retrograde analysis.
  for movenumber in 0..50 {
    let mut modified = false;

    // On odd ply, we find the best move available to the stronger side.
    for akx in 0..32 {
      for aps in 0..64 {
        for dps in 0..64 {
          for dks in 0..64 {
            let current = subtable[akx][aps][dps][dks][0];
            if !legal(current) { continue; }
            // +127 won in 0
            //   +1 won in 126
            //    0 drawn
            //   -1 lost in 126
            // -127 lost in 0
            // -128 stalemate
            let mut best : i8 = -128; // stalemate

            // King moves
            let aks = idx2sq(akx);
            let mut dests = king_destinations(aks);
            while dests != 0 {
              let dst = dests.trailing_zeros() as usize;
              let fdst; let faps; let fdps; let fdks;
              if dst % 8 >= 4 {
                fdst = hmirror(dst); faps = hmirror(aps);
                fdps = hmirror(dps); fdks = hmirror(dks);
              }
              else {
                fdst = dst; faps = aps; fdps = dps; fdks = dks;
              }
              let fakx = sq2idx(fdst);

              let entry;
              if fdst == fdps {
                let mini_subtable : &mut MiniSubtable =
                  unsafe { &mut *(mini_tb.get_unchecked_mut(x) as *mut _) };
                entry = mini_subtable[fakx][faps][fdks][1];
              }
              else {
                entry = subtable[fakx][faps][fdps][fdks][1];
              }

              if       lost(entry) { best = std::cmp::max(best, 127 - dtm(entry) as i8); }
              else if drawn(entry) { best = std::cmp::max(best, 0);                      }
              else if   won(entry) { best = std::cmp::max(best, dtm(entry) as i8 - 127); }
              dests &= dests - 1;
            }
            // Pawn moves
            let composite : u64 = (1 << aks) | (1 << aps) | (1 << dps) | (1 << dks);
            if x == 4 {
              let step = aps + 8;
              if (1 << step) & composite == 0 {
                if aps / 8 == 6 {
                  for p in 0..4 {
                    let entry;
                    if y < p {
                      let faks; let faps; let fdps; let fdks;
                      if dks % 8 >= 4 {
                        faks = hmirror(dks);  faps = hmirror(dps);
                        fdps = hmirror(step); fdks = hmirror(aks);
                      }
                      else {
                        faks = dks; faps = dps; fdps = step; fdks = aks;
                      }
                      let promotiontable : &mut Subtable =
                        unsafe { &mut *(&mut tb[1][y][p] as *mut _) };
                      entry = promotiontable[sq2idx(faks)][faps][fdps][fdks][0];
                    }
                    else {
                      let promotiontable : &mut Subtable =
                        unsafe { &mut *(&mut tb[1][p][y] as *mut _) };
                      entry = promotiontable[akx][step][dps][dks][1];
                    }
                    if       lost(entry) { best = std::cmp::max(best, 127 - dtm(entry) as i8); }
                    else if drawn(entry) { best = std::cmp::max(best, 0);                      }
                    else if   won(entry) { best = std::cmp::max(best, dtm(entry) as i8 - 127); }
                  }
                }
                else {
                  let entry = subtable[akx][step][dps][dks][1];
                  if       lost(entry) { best = std::cmp::max(best, 127 - dtm(entry) as i8); }
                  else if drawn(entry) { best = std::cmp::max(best, 0);                      }
                  else if   won(entry) { best = std::cmp::max(best, dtm(entry) as i8 - 127); }

                  let doublestep = step + 8;
                  if aps / 8 == 1 && ((1 << doublestep) & composite == 0) {
                    let entry = subtable[akx][doublestep][dps][dks][1];
                    if       lost(entry) { best = std::cmp::max(best, 127 - dtm(entry) as i8); }
                    else if drawn(entry) { best = std::cmp::max(best, 0);                      }
                    else if   won(entry) { best = std::cmp::max(best, dtm(entry) as i8 - 127); }
                  }
                }
              }
              // TODO captures and promotions by capture
            }
            // Piece moves
            else {
              let mut dests = piece_destinations(x+1, aps, composite);
              while dests != 0 {
                let dst = dests.trailing_zeros() as usize;
                let entry;
                if dst == dps {
                  let mini_subtable : &mut MiniSubtable =
                    unsafe { &mut *(mini_tb.get_unchecked_mut(x) as *mut _) };
                  entry = mini_subtable[akx][aps][dks][1];
                }
                else {
                  entry = subtable[akx][dst][dps][dks][1];
                }
                if       lost(entry) { best = std::cmp::max(best, 127 - dtm(entry) as i8); }
                else if drawn(entry) { best = std::cmp::max(best, 0);                      }
                else if   won(entry) { best = std::cmp::max(best, dtm(entry) as i8 - 127); }
                dests &= dests - 1;
              }
            }

            // Update the entry
            let update;
            if best == 0 || best == -127 { update = 0x40; }
            else if best > 0 { update = 0xC0 + ((127 - best) as u8) + 1; }
            else             { update = 0x80 + ((127 + best) as u8) + 1; }
            if update != current {
              subtable[akx][aps][dps][dks][0] = update;
              modified = true;
            }
          } // dks
        } // dps
      } // aps
    } // akx

    // On even ply, we find the best move available to the weaker side.
    for akx in 0..32 {
      for aps in 0..64 {
        for dps in 0..64 {
          for dks in 0..64 {
            let current = subtable[akx][aps][dps][dks][1];
            if !legal(current) { continue; }
            // +127 won in 0
            //   +1 won in 126
            //    0 drawn
            //   -1 lost in 126
            // -127 lost in 0
            // -128 stalemate
            let mut best : i8 = -128; // stalemate

            // King moves
            let aks = idx2sq(akx);
            let mut dests = king_destinations(dks);
            while dests != 0 {
              let dst = dests.trailing_zeros() as usize;
              let entry;
              if dst == aps {
                let faks; let faps; let fdks;
                if dst % 8 >= 4 {
                  faks = hmirror(dst); faps = hmirror(dps); fdks = hmirror(aks);
                }
                else {
                  faks = dst; faps = dps; fdks = aks;
                }
                let mini_subtable : &mut MiniSubtable =
                  unsafe { &mut *(mini_tb.get_unchecked_mut(y) as *mut _) };
                entry = mini_subtable[sq2idx(faks)][faps][fdks][1];
              }
              else {
                entry = subtable[akx][aps][dps][dst][0];
              }

              if       lost(entry) { best = std::cmp::max(best, 127 - dtm(entry) as i8); }
              else if drawn(entry) { best = std::cmp::max(best, 0);                      }
              else if   won(entry) { best = std::cmp::max(best, dtm(entry) as i8 - 127); }
              dests &= dests - 1;
            }
            // Pawn moves
            let composite : u64 = (1 << aks) | (1 << aps) | (1 << dps) | (1 << dks);
            if y == 4 {
              let step = dps - 8;
              if (1 << step) & composite == 0 {
                if dps / 8 == 1 {
                  for p in 0..4 {
                    let entry;
                    if p < x {
                      let faks; let faps; let fdps; let fdks;
                      if dks % 8 >= 4 {
                        faks = hmirror(dks);  faps = hmirror(dps);
                        fdps = hmirror(step); fdks = hmirror(aks);
                      }
                      else {
                        faks = dks; faps = dps; fdps = step; fdks = aks;
                      }
                      let promotiontable : &mut Subtable =
                        unsafe { &mut *(&mut tb[1][p][x] as *mut _) };
                      entry = promotiontable[sq2idx(faks)][faps][fdps][fdks][1];
                    }
                    else {
                      let promotiontable : &mut Subtable =
                        unsafe { &mut *(&mut tb[1][x][p] as *mut _) };
                      entry = promotiontable[akx][aps][step][dks][0];
                    }
                    if       lost(entry) { best = std::cmp::max(best, 127 - dtm(entry) as i8); }
                    else if drawn(entry) { best = std::cmp::max(best, 0);                      }
                    else if   won(entry) { best = std::cmp::max(best, dtm(entry) as i8 - 127); }
                  }
                }
                else {
                  let entry = subtable[akx][aps][step][dks][0];
                  if       lost(entry) { best = std::cmp::max(best, 127 - dtm(entry) as i8); }
                  else if drawn(entry) { best = std::cmp::max(best, 0);                      }
                  else if   won(entry) { best = std::cmp::max(best, dtm(entry) as i8 - 127); }

                  let doublestep = step - 8;
                  if aps / 8 == 6 && ((1 << doublestep) & composite == 0) {
                    let entry = subtable[akx][aps][doublestep][dks][0];
                    if       lost(entry) { best = std::cmp::max(best, 127 - dtm(entry) as i8); }
                    else if drawn(entry) { best = std::cmp::max(best, 0);                      }
                    else if   won(entry) { best = std::cmp::max(best, dtm(entry) as i8 - 127); }
                  }
                }
              }
              // TODO captures and promotions by capture
            }
            // Piece moves
            else {
              let mut dests = piece_destinations(y+1, dps, composite);
              while dests != 0 {
                let dst = dests.trailing_zeros() as usize;
                let entry;
                if dst == aps {

                  let faks; let faps; let fdks;
                  if dks % 8 >= 4 {
                    faks = hmirror(dks); faps = hmirror(dst); fdks = hmirror(aks);
                  }
                  else {
                    faks = dks; faps = dst; fdks = aks;
                  }
                  let mini_subtable : &mut MiniSubtable =
                    unsafe { &mut *(mini_tb.get_unchecked_mut(y) as *mut _) };
                  entry = mini_subtable[sq2idx(faks)][faps][fdks][1];
                }
                else {
                  entry = subtable[akx][aps][dst][dks][0];
                }
                if       lost(entry) { best = std::cmp::max(best, 127 - dtm(entry) as i8); }
                else if drawn(entry) { best = std::cmp::max(best, 0);                      }
                else if   won(entry) { best = std::cmp::max(best, dtm(entry) as i8 - 127); }
                dests &= dests - 1;
              }
            }

            // Update the entry
            let update;
            if best == 0 || best == -127 { update = 0x40; }
            else if best > 0 { update = 0xC0 + ((127 - best) as u8) + 1; }
            else             { update = 0x80 + ((127 + best) as u8) + 1; }
            if update != current {
              subtable[akx][aps][dps][dks][1] = update;
              modified = true;
            }
          } // dks
        } // dps
      } // aps
    } // akx

    if !modified { eprintln!("{movenumber}"); break; }
  }
}

pub fn build_4man() {
  // Pawnless endgames
  for x in 0..4 { for y in x..4 { build_KXYvK_subtable(x, y); } }
  for x in 0..4 { for y in x..4 { build_KXvKY_subtable(x, y); } }
  // Pawn endgames
  for x in 0..5 { build_KXYvK_subtable(x, 4); }
  for x in 0..5 { build_KXvKY_subtable(x, 4); }
}

pub fn probe_4man(state : &State, height : i16) -> i16 {
  // NOTE that this should only be called once we've
  //   checked there are in fact four men on the board.

  let white_king_sq = state.boards[WHITE+KING].trailing_zeros() as usize;
  let black_king_sq = state.boards[BLACK+KING].trailing_zeros() as usize;

  let composite    = state.sides[W] | state.sides[B];
  let piece_board  = composite ^ (state.boards[WHITE+KING] | state.boards[BLACK+KING]);
  let piece_fst_sq = piece_board.trailing_zeros() as usize;
  let piece_board  = piece_board & (piece_board - 1);
  let piece_snd_sq = piece_board.trailing_zeros() as usize;

  let piece_fst = state.squares[piece_fst_sq];
  let piece_snd = state.squares[piece_snd_sq];

  let piece_x; let mut piece_x_sq;
  let piece_y; let mut piece_y_sq;
  if piece_snd.kind() < piece_fst.kind() {
    piece_x = piece_snd; piece_x_sq = piece_snd_sq;
    piece_y = piece_fst; piece_y_sq = piece_fst_sq;
  }
  else {
    piece_x = piece_fst; piece_x_sq = piece_fst_sq;
    piece_y = piece_snd; piece_y_sq = piece_snd_sq;
  }

  let black_winning = piece_x.color() != Color::White;

  let s = if piece_x.color() == piece_y.color() { 0 } else { 1 };
  let mut atk_king_sq = if black_winning { black_king_sq } else { white_king_sq };
  let mut def_king_sq = if black_winning { white_king_sq } else { black_king_sq };

  if atk_king_sq % 8 >= 4 {
    atk_king_sq = hmirror(atk_king_sq);
    piece_x_sq  = hmirror(piece_x_sq);
    piece_y_sq  = hmirror(piece_y_sq);
    def_king_sq = hmirror(def_king_sq);
  }
  if black_winning {
    atk_king_sq = vmirror(atk_king_sq);
    piece_x_sq  = vmirror(piece_x_sq);
    piece_y_sq  = vmirror(piece_y_sq);
    def_king_sq = vmirror(def_king_sq);
  }
  let atk_king_idx = sq2idx(atk_king_sq);

  let atk_to_move = (state.turn as usize) ^ (black_winning as usize);

  let tb = unsafe { Tb.assume_init_mut() };
  let subtable : &mut Subtable = &mut tb[s][piece_x.kind()-1][piece_y.kind()-1];
  let entry = subtable[atk_king_idx][piece_x_sq][piece_y_sq][def_king_sq][atk_to_move];
  let dtm = (entry & 0x3F) as i16;
  return match entry >> 6 {
    1 => 0,
    2 => PROVEN_LOSS + height + dtm,
    3 => PROVEN_MATE - height - dtm,
    _ => unreachable!()
  };
}

// ↑↑↑ INCOMPLETE ↑↑↑ */
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub fn probe_tb_line(state : &mut State) -> (i16, Vec<Move>)
{
  let metadata = state.save();
  let mut early_moves = Vec::with_capacity(16);
  let mut  late_moves = Vec::with_capacity(32);
  state.generate_legal_moves(Selectivity::Everything, &mut early_moves, &mut late_moves);
  early_moves.append(&mut late_moves);

  let mut best_score = i16::MIN;
  let mut best_move = NULL_MOVE;

  for mv in early_moves {
    state.apply(&mv);
    let men = (state.sides[W] | state.sides[B]).count_ones();
    let score = match men {
      2 => 0,
      3 => -probe_3man(state, 1),
      /* ↓↓↓ INCOMPLETE ↓↓↓ //
      4 => -probe_4man(state, 1),
      // ↑↑↑ INCOMPLETE ↑↑↑ */
      _ => unreachable!()
    };
    if score > best_score {
      best_score = score;
      best_move = mv.clone();
    }
    state.undo(&mv);
    state.restore(&metadata);
  }

  if best_score == i16::MIN {
    return (if state.incheck { PROVEN_LOSS } else { 0 }, Vec::new());
  }

  if best_score == 0 { return (0, vec!(best_move)); }

  state.apply(&best_move);
  let (_, mut pv) = probe_tb_line(state);
  state.undo(&best_move);
  state.restore(&metadata);
  pv.insert(0, best_move);
  return (best_score, pv);
}
