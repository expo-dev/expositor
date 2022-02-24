use crate::basis::*;
use crate::color::*;
use crate::misc::*;
use crate::movetype::*;
use crate::nnue::*;
use crate::piece::*;
use crate::simd::*;
use crate::state::*;
use crate::zobrist::*;

use std::simd::Simd;

impl State {
  fn update_nnue(&mut self, piece : usize, sq : usize, increment : bool)
  {
    if self.s1.is_empty() { return; }
    let idx = self.s1.len() - 1;
    let s1 = &mut self.s1[idx];
    let ofs; let x; let x_mirror;
    if piece < 8 {
      ofs = piece*64 + sq;
      x = ofs;
      x_mirror = Bp + ofs;
    }
    else {
      ofs = (piece-8)*64 + vmirror(sq);
      x = Bp + ofs;
      x_mirror = ofs;
    }
    unsafe {
      if increment {
        for n in 0..H1 { s1[  n ] += simd_load!(SEARCH_NETWORK.w1[x],        n); }
        for n in 0..H1 { s1[H1+n] += simd_load!(SEARCH_NETWORK.w1[x_mirror], n); }
      }
      else {
        for n in 0..H1 { s1[  n ] -= simd_load!(SEARCH_NETWORK.w1[x],        n); }
        for n in 0..H1 { s1[H1+n] -= simd_load!(SEARCH_NETWORK.w1[x_mirror], n); }
      }
    }
  }

  pub fn apply(&mut self, m : &Move)
  {
    // TODO update sides incrementally, or recalculate at the end?
    // TODO eliminate state.squares[] and use at_square() instead?
    // TODO undo fused updates to state.boards[]?
    // TODO remove key_diff and update state.key directly each line?

    // Step 0. Copy the accumulators
    if self.s1.len() != 0 {
      // NOTE this is somewhat unsafe, but the only way
      //   I've found to prevent an unnecessary copy.
      self.s1.reserve(1);
      let new_len = self.s1.len() + 1;
      unsafe { self.s1.set_len(new_len); }
      let old_end : *const [Simd32; V1] = &mut self.s1[new_len-2];
      let new_end :   *mut [Simd32; V1] = &mut self.s1[new_len-1];
      unsafe { std::ptr::copy_nonoverlapping(old_end, new_end, 1); }
      // Alternatively,
      //   let new_end : *mut [f32; N1] = &mut self.s1[new_len-1];
      //   let old_end = &self.s1[new_len-2];
      //   unsafe { for n in 0..N1 { (*new_end)[n] = old_end[n]; } }
    }

    // Step 1. Swap pieces and begin calculating key diffs
    let piece_color = m.piece.color() as usize;
    let offset = (m.piece as usize)*64;

    let mut key_diff = 0;

    if m.is_promotion() {
      let src_location = 1u64 << m.src;
      self.boards[m.piece as usize] ^= src_location;
      self.sides[piece_color]       ^= src_location;

      self.squares[m.src as usize] = Piece::NullPiece;
      self.update_nnue(m.piece as usize, m.src as usize, false);

      key_diff ^= PIECE_BASIS[offset + m.src as usize];

      let dst_location = 1u64 << m.dst;
      self.boards[m.promotion as usize] ^= dst_location;
      self.sides[piece_color]           ^= dst_location;

      self.squares[m.dst as usize] = m.promotion;
      self.update_nnue(m.promotion as usize, m.dst as usize, true);

      key_diff ^= PIECE_BASIS[(m.promotion as usize)*64 + m.dst as usize];
    }
    else {
      let locations = (1u64 << m.src) | (1u64 << m.dst);
      self.boards[m.piece as usize] ^= locations;
      self.sides[piece_color]       ^= locations;

      self.squares[m.src as usize] = Piece::NullPiece;
      self.squares[m.dst as usize] = m.piece;

      self.update_nnue(m.piece as usize, m.src as usize, false);
      self.update_nnue(m.piece as usize, m.dst as usize, true);

      key_diff ^= PIECE_BASIS[offset + m.src as usize]
                ^ PIECE_BASIS[offset + m.dst as usize];
    }

    if m.is_castle() {
      match m.dst {
         2 => {
           self.boards[WHITE+ROOK] ^= 0x0000000000000009;
           self.sides[W]           ^= 0x0000000000000009;
           self.squares[0] = Piece::NullPiece;
           self.squares[3] = Piece::WhiteRook;
           self.update_nnue(WHITE+ROOK, 0, false);
           self.update_nnue(WHITE+ROOK, 3, true);
           key_diff ^= PIECE_BASIS[(WHITE+ROOK)*64 + 0]
                     ^ PIECE_BASIS[(WHITE+ROOK)*64 + 3];
         }
         6 => {
           self.boards[WHITE+ROOK] ^= 0x00000000000000A0;
           self.sides[W]           ^= 0x00000000000000A0;
           self.squares[5] = Piece::WhiteRook;
           self.squares[7] = Piece::NullPiece;
           self.update_nnue(WHITE+ROOK, 7, false);
           self.update_nnue(WHITE+ROOK, 5, true);
           key_diff ^= PIECE_BASIS[(WHITE+ROOK)*64 + 5]
                     ^ PIECE_BASIS[(WHITE+ROOK)*64 + 7];
         }
        58 => {
           self.boards[BLACK+ROOK] ^= 0x0900000000000000;
           self.sides[B]           ^= 0x0900000000000000;
           self.squares[56] = Piece::NullPiece;
           self.squares[59] = Piece::BlackRook;
           self.update_nnue(BLACK+ROOK, 56, false);
           self.update_nnue(BLACK+ROOK, 59, true);
           key_diff ^= PIECE_BASIS[(BLACK+ROOK)*64 + 56]
                     ^ PIECE_BASIS[(BLACK+ROOK)*64 + 59];
        }
        62 => {
           self.boards[BLACK+ROOK] ^= 0xA000000000000000;
           self.sides[B]           ^= 0xA000000000000000;
           self.squares[61] = Piece::BlackRook;
           self.squares[63] = Piece::NullPiece;
           self.update_nnue(BLACK+ROOK, 63, false);
           self.update_nnue(BLACK+ROOK, 61, true);
           key_diff ^= PIECE_BASIS[(BLACK+ROOK)*64 + 61]
                     ^ PIECE_BASIS[(BLACK+ROOK)*64 + 63];
        }
        _ => {}
      }
    }
    else if m.is_enpass() {
      match m.piece.color() {
        Color::White => {
          let ep_location = 1u64 << (m.dst - 8);
          self.boards[BLACK+PAWN] ^= ep_location;
          self.sides[B]           ^= ep_location;
          self.squares[m.dst as usize - 8] = Piece::NullPiece;
          self.update_nnue(BLACK+PAWN, m.dst as usize - 8, false);
          let ep_offset = (BLACK+PAWN)*64 + m.dst as usize - 8;
          key_diff ^= PIECE_BASIS[ep_offset];
        }
        Color::Black => {
          let ep_location = 1u64 << (m.dst + 8);
          self.boards[WHITE+PAWN] ^= ep_location;
          self.sides[W]           ^= ep_location;
          self.squares[m.dst as usize + 8] = Piece::NullPiece;
          self.update_nnue(WHITE+PAWN, m.dst as usize + 8, false);
          let ep_offset = (WHITE+PAWN)*64 + m.dst as usize + 8;
          key_diff ^= PIECE_BASIS[ep_offset];
        }
      }
    }
    else if m.is_capture() {
      let dst_location = 1u64 << m.dst;
      self.boards[m.captured as usize] ^= dst_location;
      self.sides[m.captured.color() as usize] ^= dst_location;

      self.update_nnue(m.captured as usize, m.dst as usize, false);

      let capture_offset = (m.captured as usize)*64 + m.dst as usize;
      key_diff ^= PIECE_BASIS[capture_offset];
    }

    // Step 2. Save current rights and en passant target
    let prev_rights = self.rights;
    let prev_enpass = self.enpass;

    // Step 3. Update rights
    if m.piece.kind() == KING {
      self.rights &= match m.piece.color() {
        Color::White => 0b1100,
        Color::Black => 0b0011,
      };
    }
    for sq in [m.src, m.dst].iter() {
      match sq {
         0 => { self.rights &= 0b1101; }
         7 => { self.rights &= 0b1110; }
        56 => { self.rights &= 0b0111; }
        63 => { self.rights &= 0b1011; }
         _ => {}
      }
    }

    // Step 4. Update en passant target
    if m.movetype == MoveType::PawnManoeuvre {
      match m.piece.color() {
        Color::White =>
          if m.src + 16 == m.dst { self.enpass = m.src + 8; } else { self.enpass = -1; }
        Color::Black =>
          if m.src - 16 == m.dst { self.enpass = m.src - 8; } else { self.enpass = -1; }
      }
    }
    else {
      self.enpass = -1;
    }

    // Step 5. Update check status, side to move, depth from zeroing, and ply
    self.incheck = m.gives_check();
    self.turn = !self.turn;
    self.dfz  = if m.is_zeroing() { 0 } else { self.dfz + 1 };
    self.ply += 1;

    // Step 6. Perform key update
    let re_diff = rezobrist(prev_rights, prev_enpass, self.rights, self.enpass);
    self.key ^= key_diff ^ re_diff ^ TURN_BASIS;
  }

  pub fn undo(&mut self, m : &Move)
  {
    self.s1.pop();

    if m.is_promotion() {
      self.boards[m.piece as usize] ^= 1u64 << m.src;
      self.squares[m.src as usize] = m.piece;
      self.boards[m.promotion as usize] ^= 1u64 << m.dst;
    }
    else {
      self.boards[m.piece as usize] ^= (1u64 << m.src) | (1u64 << m.dst);
      self.squares[m.src as usize] = m.piece;
    }

    if m.is_castle() {
      self.squares[m.dst as usize] = Piece::NullPiece;
      match m.dst {
         2 => {
           self.boards[WHITE+ROOK] ^= 0x0000000000000009;
           self.squares[0] = Piece::WhiteRook;
           self.squares[3] = Piece::NullPiece;
         }
         6 => {
           self.boards[WHITE+ROOK] ^= 0x00000000000000A0;
           self.squares[5] = Piece::NullPiece;
           self.squares[7] = Piece::WhiteRook;
         }
        58 => {
           self.boards[BLACK+ROOK] ^= 0x0900000000000000;
           self.squares[56] = Piece::BlackRook;
           self.squares[59] = Piece::NullPiece;
        }
        62 => {
           self.boards[BLACK+ROOK] ^= 0xA000000000000000;
           self.squares[61] = Piece::NullPiece;
           self.squares[63] = Piece::BlackRook;
        }
        _ => {}
      }
    }
    else if m.is_enpass() {
      self.squares[m.dst as usize] = Piece::NullPiece;
      match m.piece.color() {
        Color::White => {
          self.boards[BLACK+PAWN] ^= 1u64 << (m.dst - 8);
          self.squares[m.dst as usize - 8] = Piece::BlackPawn;
        }
        Color::Black => {
          self.boards[WHITE+PAWN] ^= 1u64 << (m.dst + 8);
          self.squares[m.dst as usize + 8] = Piece::WhitePawn;
        }
      }
    }
    else if m.is_capture() {
      self.boards[m.captured as usize] ^= 1u64 << m.dst;
      self.squares[m.dst as usize] = m.captured;
    }
    else {
      self.squares[m.dst as usize] = Piece::NullPiece;
    }
    self.turn = !self.turn;
    self.ply -= 1;
  }

  pub fn apply_null(&mut self) {
    if self.enpass >= 0 { self.key ^= ENPASS_BASIS[(self.enpass % 8) as usize]; }
    self.enpass = -1;
    self.incheck = false;
    self.turn = !self.turn;
    self.dfz += 1;
    self.ply += 1;
    self.key ^= TURN_BASIS;
  }

  pub fn undo_null(&mut self) {
    self.turn = !self.turn;
    self.ply -= 1;
  }
}
