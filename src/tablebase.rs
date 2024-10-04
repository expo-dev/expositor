use crate::color::Color::*;
use crate::dest::*;
use crate::misc::{
  piece_destinations,
  pawn_attacks,
  vmirror,
  hmirror
};
use crate::movegen::Selectivity::Everything;
use crate::movetype::Move;
use crate::piece::Kind::*;
use crate::piece::Piece::*;
use crate::score::PovScore;
use crate::state::State;

use std::mem::MaybeUninit;

// NOTE that I've decided to abandon the 4-man tablebase, but the comments
//   here are still helpful in explaining the 3-man tablebase, so I've left
//   them here.
//
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

#[allow(non_upper_case_globals)]
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
  let subtable : &mut MiniSubtable = unsafe { &mut *(tb.get_unchecked_mut(x) as *mut _) };

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
            if pawn_attacks(White, 1 << aps) & (1 << dks) != 0 { break; }
          }
          else {
            let kind = unsafe { std::mem::transmute((x+1) as u8) };
            if piece_destinations(kind, aps, composite) & (1 << dks) != 0 { break; }
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
              if pawn_attacks(White, 1 << aps) & (1 << dks) == 0 { break; }
            }
            else {
              let kind = unsafe { std::mem::transmute((x+1) as u8) };
              if piece_destinations(kind, aps, composite) & (1 << dks) == 0 { break; }
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
            let kind = unsafe { std::mem::transmute((x+1) as u8) };
            let mut dests = piece_destinations(kind, aps, composite);
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

    if !modified { break; }

    let mut modified = false;

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
  // It's important that we build the pawn subtable last.
  for x in 0..5 { build_3man_subtable(x); }
}

pub fn probe_3man(state : &State, height : u8) -> PovScore {
  // NOTE that this should only be called once we've
  //   checked there are in fact three men on the board.

  let winning = if state.sides[White].count_ones() == 1 { Black } else { White};

  let mut atk_king = state.boards[ winning+King].trailing_zeros() as usize;
  let mut def_king = state.boards[!winning+King].trailing_zeros() as usize;

  let composite = state.sides[White] | state.sides[Black];
  let piece_board = composite ^ (state.boards[WhiteKing] | state.boards[BlackKing]);
  let mut atk_piece = piece_board.trailing_zeros() as usize;

  let piece_type = state.squares[atk_piece].kind();
  let atk_to_move = (state.turn != winning) as usize;

  if atk_king % 8 >= 4 {
    atk_king  = hmirror(atk_king);
    atk_piece = hmirror(atk_piece);
    def_king  = hmirror(def_king);
  }

  // Within the 3-man tablebase, pawns always move upward,
  //   so we flip vertically when black has the pawn.
  if winning == Black {
    atk_king  = vmirror(atk_king);
    atk_piece = vmirror(atk_piece);
    def_king  = vmirror(def_king);
  }

  atk_king = sq2idx(atk_king);

  let tb = unsafe { MiniTb.assume_init_mut() };
  let subtable : &mut MiniSubtable = &mut tb[piece_type as usize - 1];
  let entry = subtable[atk_king][atk_piece][def_king][atk_to_move];
  let dtm = (entry & 0x3F) as u8;
  let distance = (height as u16) + (dtm as u16);
  let distance = if distance > 255 { 255 } else { distance as u8 };
  return match entry >> 6 {
    1 => PovScore::ZERO,
    2 => PovScore::realized_loss(distance),
    3 => PovScore::realized_win(distance),
    _ => unreachable!()
  };
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub fn probe_tb_line(state : &mut State) -> (PovScore, Vec<Move>)
{
  let legal_moves = state.collect_legal_moves(Everything);

  let mut best_score = PovScore::SENTINEL;
  let mut best_move = Move::NULL;

  let metadata = state.save();
  for mv in legal_moves.into_iter() {
    state.apply(&mv);
    let men = (state.sides[White] | state.sides[Black]).count_ones();
    let score = match men {
      2 => PovScore::ZERO,
      3 => -probe_3man(state, 1),
      _ => unreachable!()
    };
    if score > best_score {
      best_score = score;
      best_move = mv.clone();
    }
    state.undo(&mv);
    state.restore(&metadata);
  }

  if best_score.is_sentinel() {
    return (if state.incheck { PovScore::LOST } else { PovScore::ZERO }, Vec::new());
  }

  if best_score.is_zero() { return (PovScore::ZERO, vec!(best_move)); }

  state.apply(&best_move);
  let (_, mut pv) = probe_tb_line(state);
  state.undo(&best_move);
  state.restore(&metadata);
  pv.insert(0, best_move);
  return (best_score, pv);
}
