use crate::basis::*;
use crate::color::Color::{self, *};
use crate::global::nnue_enabled;
use crate::misc::{
  WHITE_SHORT_CASTLE_ROOK,
  WHITE_LONG_CASTLE_ROOK,
  BLACK_SHORT_CASTLE_ROOK,
  BLACK_LONG_CASTLE_ROOK,
  vmirror
};
use crate::movetype::{Move, MoveType::PawnManoeuvre};
use crate::nnue::{Simd32, vN1, NETWORK, king_region};
use crate::piece::Kind::*;
use crate::piece::{KQRBNP, Piece::{self, *}};
use crate::simd::{LANES, simd_load};
use crate::state::State;
use crate::zobrist::rezobrist;

use std::simd::Simd;

// ↓↓↓ DEBUG ↓↓↓
// pub static mut KING_MOVES  : usize = 0;
// pub static mut RESET_COUNT : usize = 0;
// ↑↑↑ DEBUG ↑↑↑

impl State {
  fn update_nnue(
    &mut self,
    wk_rdx : usize, // white king region index
    bk_rdx : usize, // black king region index
    piece  : Piece,
    sq     : usize,
    add    : bool
  )
  {
    if !nnue_enabled() || self.s1.is_empty() { return; }
    let last = self.s1.len() - 1;
    let s1 = &mut self.s1[last];

    let w_region = unsafe { &NETWORK.rn[wk_rdx] };
    let b_region = unsafe { &NETWORK.rn[bk_rdx] };

    let c = piece.color();  // self.turn
    let x = match c {
      White => (piece as usize    )*64 + sq,
      Black => (piece as usize & 7)*64 + vmirror(sq),
    };
    if add {
      for n in 0..vN1 { s1[White][n] += simd_load!(w_region.w1[ c][x], n); }
      for n in 0..vN1 { s1[Black][n] += simd_load!(b_region.w1[!c][x], n); }
    }
    else {
      for n in 0..vN1 { s1[White][n] -= simd_load!(w_region.w1[ c][x], n); }
      for n in 0..vN1 { s1[Black][n] -= simd_load!(b_region.w1[!c][x], n); }
    }
  }

  fn reset_s1(&mut self, color : Color, rdx : usize)
  {
    if !nnue_enabled() || self.s1.is_empty() { return; }

    // ↓↓↓ DEBUG ↓↓↓
    // unsafe { RESET_COUNT += 1; }
    // ↑↑↑ DEBUG ↑↑↑

    let last = self.s1.len() - 1;
    let s1 = &mut self.s1[last];

    let region = unsafe { &NETWORK.rn[rdx] };
    for n in 0..vN1 { s1[color][n] = simd_load!(region.b1, n); }
    for kind in KQRBNP {
      let mut sources = self.boards[White+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize)*64 + src;

        for n in 0..vN1 { s1[color][n] += simd_load!(region.w1[color][x], n); }
        sources &= sources - 1;
      }
    }
    for kind in KQRBNP {
      let mut sources = self.boards[Black+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize)*64 + vmirror(src);
        for n in 0..vN1 { s1[color][n] += simd_load!(region.w1[!color][x], n); }
        sources &= sources - 1;
      }
    }
  }

  pub fn apply(&mut self, m : &Move)
  {
    // Step 0. Copy the accumulators
    if nnue_enabled() && !self.s1.is_empty() {
      // NOTE this is somewhat unsafe, but the only way
      //   I've found to prevent an unnecessary copy.
      self.s1.reserve(1);
      let new_len = self.s1.len() + 1;
      unsafe { self.s1.set_len(new_len); }
      let old_end : *const [[Simd32; vN1]; 2] =    & self.s1[new_len-2];
      let new_end :   *mut [[Simd32; vN1]; 2] = &mut self.s1[new_len-1];
      unsafe { std::ptr::copy_nonoverlapping(old_end, new_end, 1); }
      // Alternatively,
      //   let new_end : *mut [f32; N1] = &mut self.s1[new_len-1];
      //   let old_end = &self.s1[new_len-2];
      //   unsafe { for n in 0..N1 { (*new_end)[n] = old_end[n]; } }
    }

    let wk_rdx = king_region(        self.boards[WhiteKing].trailing_zeros() as usize );
    let bk_rdx = king_region(vmirror(self.boards[BlackKing].trailing_zeros() as usize));

    // Step 1. Swap pieces and begin calculating key diffs
    let piece_color = m.piece.color();
    let offset = (m.piece as usize) * 64;

    let mut key_diff = 0;

    if m.is_promotion() {
      let src_location = 1 << m.src;
      self.boards[m.piece]    ^= src_location;
      self.sides[piece_color] ^= src_location;

      self.squares[m.src as usize] = Null;
      self.update_nnue(wk_rdx, bk_rdx, m.piece, m.src as usize, false);

      key_diff ^= PIECE_BASIS[offset + m.src as usize];

      let dst_location = 1 << m.dst;
      self.boards[m.promotion] ^= dst_location;
      self.sides[piece_color]  ^= dst_location;

      self.squares[m.dst as usize] = m.promotion;
      self.update_nnue(wk_rdx, bk_rdx, m.promotion, m.dst as usize, true);

      key_diff ^= PIECE_BASIS[(m.promotion as usize)*64 + m.dst as usize];
    }
    else {
      let locations = (1 << m.src) | (1 << m.dst);
      self.boards[m.piece]    ^= locations;
      self.sides[piece_color] ^= locations;

      self.squares[m.src as usize] = Null;
      self.squares[m.dst as usize] = m.piece;

      self.update_nnue(wk_rdx, bk_rdx, m.piece, m.src as usize, false);
      self.update_nnue(wk_rdx, bk_rdx, m.piece, m.dst as usize, true);

      key_diff ^= PIECE_BASIS[offset + m.src as usize]
                ^ PIECE_BASIS[offset + m.dst as usize];
    }

    if m.is_castle() {
      match m.dst {
         2 => {
           self.boards[WhiteRook] ^= WHITE_LONG_CASTLE_ROOK;
           self.sides[White]      ^= WHITE_LONG_CASTLE_ROOK;
           self.squares[0] = Null;
           self.squares[3] = WhiteRook;
           self.update_nnue(wk_rdx, bk_rdx, WhiteRook, 0, false);
           self.update_nnue(wk_rdx, bk_rdx, WhiteRook, 3, true);
           key_diff ^= PIECE_BASIS[(WhiteRook as usize)*64 + 0]
                     ^ PIECE_BASIS[(WhiteRook as usize)*64 + 3];
         }
         6 => {
           self.boards[WhiteRook] ^= WHITE_SHORT_CASTLE_ROOK;
           self.sides[White]      ^= WHITE_SHORT_CASTLE_ROOK;
           self.squares[5] = WhiteRook;
           self.squares[7] = Null;
           self.update_nnue(wk_rdx, bk_rdx, WhiteRook, 5, true);
           self.update_nnue(wk_rdx, bk_rdx, WhiteRook, 7, false);
           key_diff ^= PIECE_BASIS[(WhiteRook as usize)*64 + 5]
                     ^ PIECE_BASIS[(WhiteRook as usize)*64 + 7];
         }
        58 => {
           self.boards[BlackRook] ^= BLACK_LONG_CASTLE_ROOK;
           self.sides[Black]      ^= BLACK_LONG_CASTLE_ROOK;
           self.squares[56] = Null;
           self.squares[59] = BlackRook;
           self.update_nnue(wk_rdx, bk_rdx, BlackRook, 56, false);
           self.update_nnue(wk_rdx, bk_rdx, BlackRook, 59, true);
           key_diff ^= PIECE_BASIS[(BlackRook as usize)*64 + 56]
                     ^ PIECE_BASIS[(BlackRook as usize)*64 + 59];
        }
        62 => {
           self.boards[BlackRook] ^= BLACK_SHORT_CASTLE_ROOK;
           self.sides[Black]      ^= BLACK_SHORT_CASTLE_ROOK;
           self.squares[61] = BlackRook;
           self.squares[63] = Null;
           self.update_nnue(wk_rdx, bk_rdx, BlackRook, 61, true);
           self.update_nnue(wk_rdx, bk_rdx, BlackRook, 63, false);
           key_diff ^= PIECE_BASIS[(BlackRook as usize)*64 + 61]
                     ^ PIECE_BASIS[(BlackRook as usize)*64 + 63];
        }
        _ => unreachable!()
      }
    }
    else if m.is_enpass() {
      match m.piece.color() {
        White => {
          let ep_location = 1 << (m.dst - 8);
          self.boards[BlackPawn] ^= ep_location;
          self.sides[Black]      ^= ep_location;
          self.squares[m.dst as usize - 8] = Null;
          self.update_nnue(wk_rdx, bk_rdx, BlackPawn, m.dst as usize - 8, false);
          let ep_offset = (BlackPawn as usize)*64 + m.dst as usize - 8;
          key_diff ^= PIECE_BASIS[ep_offset];
        }
        Black => {
          let ep_location = 1 << (m.dst + 8);
          self.boards[WhitePawn] ^= ep_location;
          self.sides[White]      ^= ep_location;
          self.squares[m.dst as usize + 8] = Null;
          self.update_nnue(wk_rdx, bk_rdx, WhitePawn, m.dst as usize + 8, false);
          let ep_offset = (WhitePawn as usize)*64 + m.dst as usize + 8;
          key_diff ^= PIECE_BASIS[ep_offset];
        }
      }
    }
    else if m.is_capture() {
      let dst_location = 1 << m.dst;
      self.boards[m.captured] ^= dst_location;
      self.sides[m.captured.color()] ^= dst_location;

      self.update_nnue(wk_rdx, bk_rdx, m.captured, m.dst as usize, false);

      let capture_offset = (m.captured as usize)*64 + m.dst as usize;
      key_diff ^= PIECE_BASIS[capture_offset];
    }

    if m.piece.kind() == King {
      // ↓↓↓ DEBUG ↓↓↓
      // unsafe { KING_MOVES += 1; }
      // ↑↑↑ DEBUG ↑↑↑
      match self.turn {
        White => {
          let prev_rdx = wk_rdx;
          let rdx = king_region(self.boards[WhiteKing].trailing_zeros() as usize);
          if rdx != prev_rdx { self.reset_s1(White, rdx); }
        }
        Black => {
          let prev_rdx = bk_rdx;
          let rdx = king_region(vmirror(self.boards[BlackKing].trailing_zeros() as usize));
          if rdx != prev_rdx { self.reset_s1(Black, rdx); }
        }
      }
    }

    // Step 2. Save current rights and en passant target
    let prev_rights = self.rights;
    let prev_enpass = self.enpass;

    // Step 3. Update rights
    if m.piece.kind() == King {
      self.rights &= match m.piece.color() {
        White => 0b_1100,
        Black => 0b_0011,
      };
    }
    for sq in [m.src, m.dst].iter() {
      match sq {
         0 => { self.rights &= 0b_1101; }
         7 => { self.rights &= 0b_1110; }
        56 => { self.rights &= 0b_0111; }
        63 => { self.rights &= 0b_1011; }
         _ => {}
      }
    }

    // Step 4. Update en passant target
    if m.movetype == PawnManoeuvre {
      self.enpass = match m.piece.color() {
        White => if m.src + 16 == m.dst { m.src + 8 } else { -1 }
        Black => if m.src - 16 == m.dst { m.src - 8 } else { -1 }
      };
    }
    else {
      self.enpass = -1;
    }

    // Step 5. Update check status, side to move, depth from zeroing, and ply
    self.incheck = m.gives_check();
    self.turn = !self.turn;
    self.ply += 1;
    self.dfz = if m.is_zeroing() { 0 } else { self.dfz + 1 };

    // Step 6. Perform key update
    let re_diff = rezobrist(prev_rights, prev_enpass, self.rights, self.enpass);
    self.key ^= key_diff ^ re_diff ^ TURN_BASIS;
  }

  pub fn undo(&mut self, m : &Move)
  {
    if nnue_enabled() { self.s1.pop(); }

    if m.is_promotion() {
      self.boards[m.piece] ^= 1 << m.src;
      self.squares[m.src as usize] = m.piece;
      self.boards[m.promotion] ^= 1 << m.dst;
    }
    else {
      self.boards[m.piece] ^= (1 << m.src) | (1 << m.dst);
      self.squares[m.src as usize] = m.piece;
    }

    if m.is_castle() {
      self.squares[m.dst as usize] = Null;
      match m.dst {
         2 => {
           self.boards[WhiteRook] ^= WHITE_LONG_CASTLE_ROOK;
           self.squares[0] = WhiteRook;
           self.squares[3] = Null;
         }
         6 => {
           self.boards[WhiteRook] ^= WHITE_SHORT_CASTLE_ROOK;
           self.squares[5] = Null;
           self.squares[7] = WhiteRook;
         }
        58 => {
           self.boards[BlackRook] ^= BLACK_LONG_CASTLE_ROOK;
           self.squares[56] = BlackRook;
           self.squares[59] = Null;
        }
        62 => {
           self.boards[BlackRook] ^= BLACK_SHORT_CASTLE_ROOK;
           self.squares[61] = Null;
           self.squares[63] = BlackRook;
        }
        _ => unreachable!()
      }
    }
    else if m.is_enpass() {
      self.squares[m.dst as usize] = Null;
      match m.piece.color() {
        White => {
          self.boards[BlackPawn] ^= 1 << (m.dst - 8);
          self.squares[m.dst as usize - 8] = BlackPawn;
        }
        Black => {
          self.boards[WhitePawn] ^= 1 << (m.dst + 8);
          self.squares[m.dst as usize + 8] = WhitePawn;
        }
      }
    }
    else if m.is_capture() {
      self.boards[m.captured as usize] ^= 1 << m.dst;
      self.squares[m.dst as usize] = m.captured;
    }
    else {
      self.squares[m.dst as usize] = Null;
    }
    self.turn = !self.turn;
    self.ply -= 1;
  }

  pub fn apply_null(&mut self)
  {
    if self.enpass >= 0 {
      self.key ^= ENPASS_BASIS[(self.enpass % 8) as usize];
    }
    self.enpass = -1;
    self.incheck = false;
    self.turn = !self.turn;
    self.dfz += 1;
    self.ply += 1;
    self.key ^= TURN_BASIS;
  }

  pub fn undo_null(&mut self)
  {
    self.turn = !self.turn;
    self.ply -= 1;
  }
}
