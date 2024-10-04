use crate::basis::*;
use crate::color::Color::{self, *};
use crate::global::nnue_enabled;
use crate::misc::{
  SHORT, LONG,
  CASTLE_ROOK,
  vmirror
};
use crate::movetype::{Move, MoveType::PawnManoeuvre};
// use crate::nnue::{N1, king_region};
use crate::nnext::{N1, king_region, QUANTIZED};
use crate::piece::Kind::*;
use crate::piece::{KQRBNP, Piece::{self, *}};
use crate::state::State;
use crate::zobrist::rezobrist;

// use std::simd::prelude::Simd;

fn simd_copy<const N : usize>(a : &mut [i16; N], b : &[i16; N])
{
  // for n in 0..N/16 {
  //   let ofs = n * 16;
  //   Simd::<i16, 16>::from_slice(    &b[ofs .. ofs+16])
  //                .copy_to_slice(&mut a[ofs .. ofs+16]);
  // }
  for n in 0..N { a[n] = b[n]; }
}

fn simd_incr(a : &mut [i16; N1], b : &[i16; N1])
{
  // for n in 0..N/16 {
  //   let ofs = n * 16;
  //   let va = Simd::<i16, 16>::from_slice(&a[ofs .. ofs+16]);
  //   let vb = Simd::<i16, 16>::from_slice(&b[ofs .. ofs+16]);
  //   (va + vb).copy_to_slice(&mut a[ofs .. ofs+16]);
  // }
  for n in 0..N1 { a[n] = a[n].wrapping_add(b[n]); }
}

fn simd_decr<const N : usize>(a : &mut [i16; N], b : &[i16; N])
{
  // for n in 0..N/16 {
  //   let ofs = n * 16;
  //   let va = Simd::<i16, 16>::from_slice(&a[ofs .. ofs+16]);
  //   let vb = Simd::<i16, 16>::from_slice(&b[ofs .. ofs+16]);
  //   (va - vb).copy_to_slice(&mut a[ofs .. ofs+16]);
  // }
  for n in 0..N { a[n] = a[n].wrapping_sub(b[n]); }
}

impl State {
  fn update_s1(
    &mut self,
    wk_rdx : usize, // white king region index
    bk_rdx : usize, // black king region index
    piece  : Piece,
    sq     : usize,
    add    : bool
  )
  {
    let last = self.s1.len() - 1;
    let s1 = &mut self.s1[last];

    let w_region = unsafe { &QUANTIZED.l1[wk_rdx] };
    let b_region = unsafe { &QUANTIZED.l1[bk_rdx] };

    let c = piece.color();  // self.turn
    let x = match c {
      White => (piece as usize    )*64 + sq,
      Black => (piece as usize & 7)*64 + vmirror(sq),
    };
    if add {
      simd_incr(&mut s1[White], &w_region.w1[ c][x]);
      simd_incr(&mut s1[Black], &b_region.w1[!c][x]);
    }
    else {
      simd_decr(&mut s1[White], &w_region.w1[ c][x]);
      simd_decr(&mut s1[Black], &b_region.w1[!c][x]);
    }
  }

  fn update_s1_manoeuvre(
    &mut self,
    wk_rdx : usize, // white king region index
    bk_rdx : usize, // black king region index
    piece  : Piece,
    src    : usize,
    dst    : usize,
  )
  {
    let last = self.s1.len() - 1;
    let s1 = &mut self.s1[last];

    let w_region = unsafe { &QUANTIZED.l1[wk_rdx] };
    let b_region = unsafe { &QUANTIZED.l1[bk_rdx] };

    let c = piece.color();  // self.turn
    let x = match c {
      White => (piece as usize    )*64 + src,
      Black => (piece as usize & 7)*64 + vmirror(src),
    };
    let y = match c {
      White => (piece as usize    )*64 + dst,
      Black => (piece as usize & 7)*64 + vmirror(dst),
    };
    for n in 0..N1 { s1[White][n] = s1[White][n].wrapping_sub(w_region.w1[ c][x][n]).wrapping_add(w_region.w1[ c][y][n]); }
    for n in 0..N1 { s1[Black][n] = s1[Black][n].wrapping_sub(b_region.w1[!c][x][n]).wrapping_add(b_region.w1[!c][y][n]); }
  }

  fn update_s1_capture(
    &mut self,
    wk_rdx : usize, // white king region index
    bk_rdx : usize, // black king region index
    piece  : Piece,
    capt   : Piece,
    src    : usize,
    dst    : usize,
  )
  {
    let last = self.s1.len() - 1;
    let s1 = &mut self.s1[last];

    let w_region = unsafe { &QUANTIZED.l1[wk_rdx] };
    let b_region = unsafe { &QUANTIZED.l1[bk_rdx] };

    let c = piece.color();  // self.turn
    let x = match c {
      White => (piece as usize    )*64 + src,
      Black => (piece as usize & 7)*64 + vmirror(src),
    };
    let y = match c {
      White => (piece as usize    )*64 + dst,
      Black => (piece as usize & 7)*64 + vmirror(dst),
    };
    let z = match !c {  // we assume capt.color() = !piece.color()
      White => (capt as usize    )*64 + dst,
      Black => (capt as usize & 7)*64 + vmirror(dst),
    };
    for n in 0..N1 { s1[White][n] = s1[White][n].wrapping_sub(w_region.w1[ c][x][n]).wrapping_add(w_region.w1[ c][y][n]).wrapping_sub(w_region.w1[!c][z][n]); }
    for n in 0..N1 { s1[Black][n] = s1[Black][n].wrapping_sub(b_region.w1[!c][x][n]).wrapping_add(b_region.w1[!c][y][n]).wrapping_sub(b_region.w1[ c][z][n]); }
  }

  fn reset_s1(&mut self, color : Color, rdx : usize)
  {
    let last = self.s1.len() - 1;
    let s1 = &mut self.s1[last];
    s1[color] = [0; N1];

    let l1 = unsafe { &QUANTIZED.l1[rdx] };
    for kind in KQRBNP {
      let mut sources = self.boards[White+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize)*64 + src;
        simd_incr(&mut s1[color], &l1.w1[color][x]);
        sources &= sources - 1;
      }
    }
    for kind in KQRBNP {
      let mut sources = self.boards[Black+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize)*64 + vmirror(src);
        simd_incr(&mut s1[color], &l1.w1[!color][x]);
        sources &= sources - 1;
      }
    }
  }

  fn copy_s1(&mut self)
  {
    // NOTE this is somewhat unsafe, but the only way
    //   I've found to prevent an unnecessary copy.
    self.s1.reserve(1);
    let new_len = self.s1.len() + 1;
    unsafe { self.s1.set_len(new_len); }
    let old_end : *const [[i16; N1]; 2] =    & self.s1[new_len-2];
    let new_end :   *mut [[i16; N1]; 2] = &mut self.s1[new_len-1];
    unsafe { std::ptr::copy_nonoverlapping(old_end, new_end, 1); }
    // Alternatively,
    //   let new_end : *mut [f32; N1] = &mut self.s1[new_len-1];
    //   let old_end = &self.s1[new_len-2];
    //   unsafe { for n in 0..N1 { (*new_end)[n] = old_end[n]; } }
  }

  pub fn apply(&mut self, m : &Move)
  {
    let wk_rdx = king_region(        self.boards[WhiteKing].trailing_zeros() as usize );
    let bk_rdx = king_region(vmirror(self.boards[BlackKing].trailing_zeros() as usize));

    // Step 0. Update the accumulators
    let valid_s1 = /* nnue_enabled() && */ !self.s1.is_empty();
    if valid_s1 {
      self.copy_s1();

      if m.is_castle() {
        self.update_s1_manoeuvre(wk_rdx, bk_rdx, m.piece, m.src as usize, m.dst as usize);
        match m.dst {
           2 => { self.update_s1_manoeuvre(wk_rdx, bk_rdx, WhiteRook,  0,  3); }
           6 => { self.update_s1_manoeuvre(wk_rdx, bk_rdx, WhiteRook,  7,  5); }
          58 => { self.update_s1_manoeuvre(wk_rdx, bk_rdx, BlackRook, 56, 59); }
          62 => { self.update_s1_manoeuvre(wk_rdx, bk_rdx, BlackRook, 63, 61); }
           _ => unreachable!()
        }
      }
      else if m.is_enpass() {
        self.update_s1_manoeuvre(wk_rdx, bk_rdx, m.piece, m.src as usize, m.dst as usize);
        match m.piece.color() {
          White => { self.update_s1(wk_rdx, bk_rdx, BlackPawn, m.dst as usize - 8, false); }
          Black => { self.update_s1(wk_rdx, bk_rdx, WhitePawn, m.dst as usize + 8, false); }
        }
      }
      else if m.is_promotion() {
        self.update_s1(wk_rdx, bk_rdx, m.piece,    m.src as usize, false);
        self.update_s1(wk_rdx, bk_rdx, m.promotion, m.dst as usize, true);
        if m.is_capture() {
          self.update_s1(wk_rdx, bk_rdx, m.captured, m.dst as usize, false);
        }
      }
      else {
        if m.is_capture() {
          self.update_s1_capture(wk_rdx, bk_rdx, m.piece, m.captured, m.src as usize, m.dst as usize);
        }
        else {
          self.update_s1_manoeuvre(wk_rdx, bk_rdx, m.piece, m.src as usize, m.dst as usize);
        }
      }
    }

    // Step 1. Swap pieces and begin calculating key diffs
    let piece_color = m.piece.color();
    let offset = (m.piece as usize) * 64;

    let mut key_diff = 0;

    if m.is_promotion() {
      let src_location = 1 << m.src;
      self.boards[m.piece]    ^= src_location;
      self.sides[piece_color] ^= src_location;

      self.squares[m.src as usize] = Null;

      key_diff ^= PIECE_BASIS[offset + m.src as usize];

      let dst_location = 1 << m.dst;
      self.boards[m.promotion] ^= dst_location;
      self.sides[piece_color]  ^= dst_location;

      self.squares[m.dst as usize] = m.promotion;

      key_diff ^= PIECE_BASIS[(m.promotion as usize)*64 + m.dst as usize];
    }
    else {
      let locations = (1 << m.src) | (1 << m.dst);
      self.boards[m.piece]    ^= locations;
      self.sides[piece_color] ^= locations;

      self.squares[m.src as usize] = Null;
      self.squares[m.dst as usize] = m.piece;

      key_diff ^= PIECE_BASIS[offset + m.src as usize]
                ^ PIECE_BASIS[offset + m.dst as usize];
    }

    if m.is_castle() {
      match m.dst {
        2 => {
          self.boards[WhiteRook] ^= CASTLE_ROOK[White][LONG];
          self.sides[White]      ^= CASTLE_ROOK[White][LONG];
          self.squares[0] = Null;
          self.squares[3] = WhiteRook;
          key_diff ^= PIECE_BASIS[(WhiteRook as usize)*64 + 0]
                    ^ PIECE_BASIS[(WhiteRook as usize)*64 + 3];
        }
        6 => {
          self.boards[WhiteRook] ^= CASTLE_ROOK[White][SHORT];
          self.sides[White]      ^= CASTLE_ROOK[White][SHORT];
          self.squares[5] = WhiteRook;
          self.squares[7] = Null;
          key_diff ^= PIECE_BASIS[(WhiteRook as usize)*64 + 5]
                    ^ PIECE_BASIS[(WhiteRook as usize)*64 + 7];
        }
        58 => {
          self.boards[BlackRook] ^= CASTLE_ROOK[Black][LONG];
          self.sides[Black]      ^= CASTLE_ROOK[Black][LONG];
          self.squares[56] = Null;
          self.squares[59] = BlackRook;
          key_diff ^= PIECE_BASIS[(BlackRook as usize)*64 + 56]
                    ^ PIECE_BASIS[(BlackRook as usize)*64 + 59];
        }
        62 => {
          self.boards[BlackRook] ^= CASTLE_ROOK[Black][SHORT];
          self.sides[Black]      ^= CASTLE_ROOK[Black][SHORT];
          self.squares[61] = BlackRook;
          self.squares[63] = Null;
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
          let ep_offset = (BlackPawn as usize)*64 + m.dst as usize - 8;
          key_diff ^= PIECE_BASIS[ep_offset];
        }
        Black => {
          let ep_location = 1 << (m.dst + 8);
          self.boards[WhitePawn] ^= ep_location;
          self.sides[White]      ^= ep_location;
          self.squares[m.dst as usize + 8] = Null;
          let ep_offset = (WhitePawn as usize)*64 + m.dst as usize + 8;
          key_diff ^= PIECE_BASIS[ep_offset];
        }
      }
    }
    else if m.is_capture() {
      let dst_location = 1 << m.dst;
      self.boards[m.captured] ^= dst_location;
      self.sides[m.captured.color()] ^= dst_location;

      let capture_offset = (m.captured as usize)*64 + m.dst as usize;
      key_diff ^= PIECE_BASIS[capture_offset];
    }

    if valid_s1 {
      if m.piece.kind() == King {
        match self.turn {
          White => {
            let prev_rdx = wk_rdx;
            let rdx = king_region(m.dst as usize);
            if rdx != prev_rdx { self.reset_s1(White, rdx); }
          }
          Black => {
            let prev_rdx = bk_rdx;
            let rdx = king_region(vmirror(m.dst as usize));
            if rdx != prev_rdx { self.reset_s1(Black, rdx); }
          }
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

    // ↓↓↓ DEBUG ↓↓↓
    /*
    if !valid_s1 { return; }

    use crate::color::WB;
    use crate::nnue::{NETWORK, SameSide, OppoSide};

    let wk_idx =         self.boards[WhiteKing].trailing_zeros() as usize ;
    let bk_idx = vmirror(self.boards[BlackKing].trailing_zeros() as usize);
    let w_region = unsafe { &NETWORK.rn[king_region(wk_idx)] };
    let b_region = unsafe { &NETWORK.rn[king_region(bk_idx)] };

    let mut s1 = [w_region.b1, b_region.b1];

    for kind in KQRBNP {
      let mut sources = self.boards[White+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize)*64 + src;
        for n in 0..N1 { s1[White][n] += w_region.w1[SameSide][x][n]; }
        for n in 0..N1 { s1[Black][n] += b_region.w1[OppoSide][x][n]; }
        sources &= sources - 1;
      }
    }
    for kind in KQRBNP {
      let mut sources = self.boards[Black+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize)*64 + vmirror(src);
        for n in 0..N1 { s1[White][n] += w_region.w1[OppoSide][x][n]; }
        for n in 0..N1 { s1[Black][n] += b_region.w1[SameSide][x][n]; }
        sources &= sources - 1;
      }
    }

    let last = self.s1.len() - 1;
    let q1 = &self.s1[last];
    for c in WB {
      for n in 0..N1 {
        let exact  = s1[c][n];
        let approx = q1[c][n];
        // i7p9 to f32
        let cast = approx as f32 / (1 << 9) as f32;
        let diff = (cast - exact).abs();
        if diff > 0.015625 {
          eprintln!("     {c} {n} {exact} {approx} {cast}");
        }
        if exact.abs() > 0.25 && (diff / exact.abs()) > (1.0 / 16.0) {
          eprintln!("1/16 {c} {n} {exact} {approx} {cast}");
        }
        if exact.abs() > 0.50 && (diff / exact.abs()) > (1.0 / 32.0) {
          eprintln!("1/32 {c} {n} {exact} {approx} {cast}");
        }
        if exact.abs() > 1.00 && (diff / exact.abs()) > (1.0 / 64.0) {
          eprintln!("1/64 {c} {n} {exact} {approx} {cast}");
        }
      }
    }
    */
    // ↑↑↑ DEBUG ↑↑↑
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
           self.boards[WhiteRook] ^= CASTLE_ROOK[White][LONG];
           self.squares[0] = WhiteRook;
           self.squares[3] = Null;
         }
         6 => {
           self.boards[WhiteRook] ^= CASTLE_ROOK[White][SHORT];
           self.squares[5] = Null;
           self.squares[7] = WhiteRook;
         }
        58 => {
           self.boards[BlackRook] ^= CASTLE_ROOK[Black][LONG];
           self.squares[56] = BlackRook;
           self.squares[59] = Null;
        }
        62 => {
           self.boards[BlackRook] ^= CASTLE_ROOK[Black][SHORT];
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
