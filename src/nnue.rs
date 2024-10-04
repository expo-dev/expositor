#![allow(non_upper_case_globals)]

use crate::color::{WB, Color::*};
use crate::default::DEFAULT_NETWORK;
use crate::misc::vmirror;
use crate::piece::{KQRBNP, Piece::{WhiteKing, BlackKing}};
use crate::rand::RandDist;
use crate::state::{MiniState, State};
use crate::simd::{LANES, simd_load, relu_ps, horizontal_sum};

use std::fs::File;
use std::io::{Read, Write, BufWriter, Error};
use std::mem::MaybeUninit;
use std::simd::Simd;

struct Guard<const B : bool> { }
impl <const B : bool> Guard<B> {
  const CHECK : () = assert!(B);
  fn assert() { let _ = Self::CHECK; }
}

macro_rules! static_assert {
  ($cond:expr) => { Guard::<{$cond}>::assert(); }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// The Expositor NNUE has 768 inputs, two hidden layers, and a single output neuron. For the
//   network to be efficiently updatable, the order of the inputs and first layer neurons (and
//   the weights between them) is fixed: inputs #0 to #383 are for white and #384 to #767 are
//   for black. However, for the sake of symmetry (and to avoid having the network learn some
//   concepts twice), if black is the side to move we swap the upper and lower banks of the
//   first layer activations before computing the second layer. In this way we ensure the
//   lower bank, #0 to #(N1-1), is for the side to move and the upper bank, #N1 to #(N1×2-1),
//   is for the side waiting.
//
// For this to work, we do two things. First, we arrange the inputs for black so that black's
//   position is flipped vertically, e.g. input #4 is hot when a white king is on e1 and the
//   corresponding input, #(Np+4), is hot when a black king is on e8. Second, we mirror the
//   weights to the banks of the first layer, so that e.g. the #4 → #0 weight is the same as
//   the #(Np+4) → #N1 weight, or e.g. the #(Np+4) → #0 weight is the same as the #4 → #N1
//   weight.
//
//           neuron-to
//              ─┴─
//     weight[x][n] = weight[x±Np][n±N1]    with signs chosen so that n±N1 and x±Np
//           ─┬─                              are not out of bounds
//       input-from
//
//   Since the weights are mirrored, we only bother storing half of them. Here is the same
//   equality as above, written out explicitly, with the canonical form on the righthand of
//   each equality:
//
//                             x = 0..Np                    x = Np..Np×2
//                   -----------------------------------------------------------
//                  |
//     n =  0..N1   |          w1[x][n]                      w1[x][n]
//                  |
//     n = N1..N1×2 |  w1[x][n] = w1[x+Np][n-N1]     w1[x][n] = w1[x-Np][x-N1]
//                  |
//
//   A last word about notation and terminology: we always use minuscule s to denote the
//   stimiulus or weighted sum of inputs to a neuron, e.g. s1[n]. The letter a is used for
//   activation, by which we always mean the output of the activation function, and so we have
//   that a1[n] := relu(s1[n]) and a2[n] := relu(s2[n]) for all n.

pub type Simd32 = Simd<f32, LANES>;

const SIMD_ZERO : Simd32 = Simd::from_array([0.0; LANES]);

pub const SideToMove  : usize = 0;
pub const SideWaiting : usize = 1;

pub const SameSide : usize = 0;
pub const OppoSide : usize = 1;

// We switch between different regions based on the position of the kings.
pub const REGIONS : usize = 5;

// We switch between different heads (layer two and output neurons)
//   based on the number of men on the board
pub const HEADS : usize = 4;

// Np must be a multiple of 2×LANES
// N1 must be a multiple of 8×LANES
pub const Np : usize = 384; // Do not modify
pub const N1 : usize = 256; // Okay to vary

// N2 must be a multiple of LANES
pub const N2 : usize = 8;  // Okay to vary
pub const N3 : usize = 1;  // Do not modify

// Number of input vectors and number of vectors per layer
pub const vNp : usize = Np / LANES;
pub const vN1 : usize = N1 / LANES;
pub const vN2 : usize = N2 / LANES;

const BODY_TOTAL : usize = (Np*2)*N1  // Weights
                         +        N1; // Biases
const HEAD_TOTAL : usize = (N1*2)*N2 + N2*N3  // Weights
                         +        N2 +    N3; // Biases

// We set the alignment of Network structs to 32 bytes so that SIMD loads and stores will be
//   aligned. The Rust reference states that the size of the struct will be a multiple of the
//   alignment, so the size of the struct in terms of single-precision floating point numbers
//   (which are 4 bytes long) is a multiple of 8.
//
pub const BODY_SIZE : usize = ((BODY_TOTAL + 7) / 8) * 8;
pub const HEAD_SIZE : usize = ((HEAD_TOTAL + 7) / 8) * 8;
    const      SIZE : usize = BODY_SIZE*REGIONS + HEAD_SIZE*HEADS;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[derive(Clone, PartialEq)]
#[repr(align(32))]
pub struct NetworkHead {
  pub w2 : [[[f32; N1]; 2]; N2],  // weight[to-second][from-fst-side][from-first]
  pub w3 :   [f32; N2],           // weight[from-second]
  pub b2 :   [f32; N2],           // bias[second]
  pub b3 :    f32,                // bias
}

#[derive(Clone, PartialEq)]
#[repr(align(32))]
pub struct NetworkBody {
  pub w1 : [[[f32; N1]; Np]; 2],  // weight[from-inp-side][from-input][to-first]
  pub b1 :   [f32; N1],           // bias[first]
}

#[derive(Clone, PartialEq)]
#[repr(align(32))]
pub struct Network {
  pub rn : [NetworkBody; REGIONS],
  pub hd : [NetworkHead; HEADS],
}

fn static_assert_block() { static_assert!(SIZE == std::mem::size_of::<Network>()); }

pub static mut NETWORK : Network = unsafe { std::mem::transmute(*DEFAULT_NETWORK) };

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
/*
pub static mut S1_MAX : Simd32 = SIMD_ZERO;
pub static mut S1_MIN : Simd32 = SIMD_ZERO;
pub static mut S2_INP_MAX   : f32 = 0.0;
pub static mut S2_INP_MIN   : f32 = 0.0;
pub static mut S2_WORST_MAX : f32 = 0.0;
pub static mut S2_WORST_MIN : f32 = 0.0;
pub static mut S2_MAX       : f32 = 0.0;
pub static mut S2_MIN       : f32 = 0.0;

pub fn print_stats()
{
  use std::simd::num::SimdFloat;
  unsafe {
    let mut w1_max = 0.0;
    let mut w1_min = 0.0;
    for r in 0..REGIONS {
      let body = &NETWORK.rn[r];
      for c in WB {
        for x in 0..Np {
          for n in 0..N1 {
            let w = body.w1[c][x][n];
            if w > w1_max { w1_max = w; }
            if w < w1_min { w1_min = w; }
          }
        }
      }
    }
    let mut w2_max = 0.0;
    let mut w2_min = 0.0;
    for h in 0..HEADS {
      let head = &NETWORK.hd[h];
      for n in 0..N2 {
        for c in WB {
          for x in 0..N1 {
            let w = head.w2[n][c][x];
            if w > w2_max { w2_max = w; }
            if w < w2_min { w2_min = w; }
          }
        }
      }
    }
    let mut w3_max = 0.0;
    let mut w3_min = 0.0;
    for h in 0..HEADS {
      let head = &NETWORK.hd[h];
      for n in 0..N2 {
        let w = head.w3[n];
        if w > w3_max { w3_max = w; }
        if w < w3_min { w3_min = w; }
      }
    }

    let upper = S1_MAX.reduce_max();
    let lower = S1_MIN.reduce_min();
    eprintln!("w1 maximum: {w1_max:+11.6}");
    eprintln!("w1 minimum: {w1_min:+11.6}");
    eprintln!();
    eprintln!("s1 maximum: {:+11.6}", upper);
    eprintln!("s1 minimum: {:+11.6}", lower);
    eprintln!();
    eprintln!("w2 maximum: {w2_max:+11.6}");
    eprintln!("w2 minimum: {w2_min:+11.6}");
    eprintln!();
    eprintln!("s2 inp max: {:+11.6}", S2_INP_MAX  );
    eprintln!("s2 inp min: {:+11.6}", S2_INP_MIN  );
    eprintln!("s2 peak mx: {:+11.6}", S2_WORST_MAX);
    eprintln!("s2 peak mn: {:+11.6}", S2_WORST_MIN);
    eprintln!("s2 maximum: {:+11.6}", S2_MAX      );
    eprintln!("s2 minimum: {:+11.6}", S2_MIN      );
    eprintln!();
    eprintln!("w3 maximum: {w3_max:+11.6}");
    eprintln!("w3 minimum: {w3_min:+11.6}");
  }
}
*/
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

impl Network {
  pub const fn zero() -> Self
  {
    // See https://github.com/rust-lang/rust/issues/62061
    //   and https://github.com/maxbla/const-zero
    const SZ : usize = std::mem::size_of::<Network>();
    union Empty {
      ary : [u8; SZ],
      net : std::mem::ManuallyDrop<Network>
    }
    const ZERO : Empty = Empty { ary: [0; SZ] };
    return std::mem::ManuallyDrop::<Network>::into_inner(unsafe { ZERO.net });
  }

  pub fn perturb_fst_snd(&mut self, n1 : usize)
  {
    let f1 = 32.0_f32.sqrt().recip();
    // for r in 0..REGIONS {
    //   let region = &mut self.rn[r];
    //   for np in 0..Np { region.w1[SameSide][np][n1] += f32::triangular() * f1; }
    //   for np in 0..Np { region.w1[OppoSide][np][n1] += f32::triangular() * f1; }
    // }
    for kind in 0..6 {
      for rank in 0..8 {
        if kind == 5 && (rank == 0 || rank == 7) { continue; }
        for file in 0..8 {
          let sq = rank*8 + file;
          let np = kind*64 + sq;
          let a = f32::triangular() * f1;
          let b = f32::triangular() * f1;
          for r in 0..REGIONS {
            if kind != 0 || king_region(sq) == r {
              self.rn[r].w1[SameSide][np][n1] += a;
            }
            self.rn[r].w1[OppoSide][np][n1] += b;
          }
        }
      }
    }
    let f2 = ((N1*2) as f32).sqrt().recip();
    // for h in 0..HEADS {
    //   let head = &mut self.hd[h];
    //   for n2 in 0..N2 {
    //     head.w2[n2][SideToMove ][n1] += f32::uniform() * f2;
    //     head.w2[n2][SideWaiting][n1] += f32::uniform() * f2;
    //   }
    // }
    for n2 in 0..N2 {
      let a = f32::uniform() * f2;
      let b = f32::uniform() * f2;
      for h in 0..HEADS {
        self.hd[h].w2[n2][SideToMove ][n1] += a;
        self.hd[h].w2[n2][SideWaiting][n1] += b;
      }
    }
  }

  pub fn perturb_thd(&mut self)
  {
    let f3 = (N2 as f32).sqrt().recip();
    for h in 0..HEADS {
      let head = &mut self.hd[h];
      for n2 in 0..N2 { head.w3[n2] += f32::uniform() * f3; }
    }
  }

  fn checksum(&self) -> u64
  {
    // Fletcher's checksum
    let mut lo : u32 = 0;
    let mut hi : u32 = 0;
    let array = unsafe { std::mem::transmute::<_,&[u32; SIZE]>(self) };
    for x in &array[0..SIZE] {
      let (sum, overflow) = lo.overflowing_add(*x);
      lo = if overflow { sum + 1 } else { sum };
      let (sum, overflow) = hi.overflowing_add(lo);
      hi = if overflow { sum + 1 } else { sum };
    }
    return ((hi as u64) << 32) | (lo as u64);
  }

  pub fn load(path : &str) -> std::io::Result<Self>
  {
    let mut fh = File::open(path)?;
    let mut sgntr = [0; 4];
    let mut array = [0; SIZE*4];
    let mut check = [0; 8];
    fh.read_exact(&mut sgntr)?;
    if sgntr != "EXPO".as_bytes() {
      return Err(Error::other("missing signature"));
    }
    fh.read_exact(&mut array)?;
    let network = unsafe { std::mem::transmute::<_,Self>(array) };
    fh.read_exact(&mut check)?;
    if network.checksum() != u64::from_le_bytes(check) {
      return Err(Error::other("checksum mismatch"));
    }
    return Ok(network);
  }

  pub fn save(&self, path : &str) -> std::io::Result<()>
  {
    let mut w = BufWriter::new(File::create(path)?);
    let bytes = unsafe { std::mem::transmute::<_,&[u8; SIZE*4]>(self) };
    w.write_all("EXPO".as_bytes())?;
    w.write_all(bytes)?;
    w.write_all(&self.checksum().to_le_bytes())?;
    return Ok(());
  }

  pub fn save_default(&self) -> std::io::Result<()>
  {
    let mut w = BufWriter::new(File::create("default.nnue")?);
    let bytes = unsafe { std::mem::transmute::<_,&[u8; SIZE*4]>(self) };
    w.write_all(bytes)?;
    return Ok(());
  }

  pub fn evaluate(&self, state : &State, head_idx : usize) -> f32
  {
    // This method is slower, but does not mutate state.

    let wk_idx =         state.boards[WhiteKing].trailing_zeros() as usize ;
    let bk_idx = vmirror(state.boards[BlackKing].trailing_zeros() as usize);
    let w_region = &self.rn[king_region(wk_idx)];
    let b_region = &self.rn[king_region(bk_idx)];
    let head = &self.hd[head_idx];

    let mut s1 : [[MaybeUninit<Simd32>; vN1]; 2] =
      [[MaybeUninit::uninit(); vN1], [MaybeUninit::uninit(); vN1]];

    for n in 0..vN1 { s1[White][n].write(simd_load!(w_region.b1, n)); }
    for n in 0..vN1 { s1[Black][n].write(simd_load!(b_region.b1, n)); }

    let s1 = unsafe { std::mem::transmute::<_,&mut [[Simd32; vN1]; 2]>(&mut s1) };

    for kind in KQRBNP {
      let mut sources = state.boards[White+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize)*64 + src;
        for n in 0..vN1 { s1[White][n] += simd_load!(w_region.w1[SameSide][x], n); }
        for n in 0..vN1 { s1[Black][n] += simd_load!(b_region.w1[OppoSide][x], n); }
        sources &= sources - 1;
      }
    }
    for kind in KQRBNP {
      let mut sources = state.boards[Black+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize)*64 + vmirror(src);
        for n in 0..vN1 { s1[White][n] += simd_load!(w_region.w1[OppoSide][x], n); }
        for n in 0..vN1 { s1[Black][n] += simd_load!(b_region.w1[SameSide][x], n); }
        sources &= sources - 1;
      }
    }

    for c in WB { for n in 0..vN1 { s1[c][n] = relu_ps(s1[c][n]); } }
    let a1 = s1;

    let mut s2 : [MaybeUninit<f32>; N2] = [MaybeUninit::uninit(); N2];
    for n in 0..N2 {
      let mut s = SIMD_ZERO;
      let c = state.turn;
      for x in 0..vN1 { s += a1[ c][x] * simd_load!(head.w2[n][SideToMove ], x); }
      for x in 0..vN1 { s += a1[!c][x] * simd_load!(head.w2[n][SideWaiting], x); }
      s2[n].write(head.b2[n] + horizontal_sum(s));
    }
    let s2 = unsafe { std::mem::transmute::<_,&mut [f32; N2]>(&mut s2) };

    let mut s = SIMD_ZERO;
    for x in 0..vN2 { s += relu_ps(simd_load!(s2, x)) * simd_load!(head.w3, x); }
    let s3 = head.b3 + horizontal_sum(s);
    return s3;
  }
}

pub const fn king_region(idx : usize) -> usize {
  // This function appears in Mantissa's source (with a different constant),
  //   but it isn't borrowed from Mantissa; it's code that I wrote. After a
  //   discussion with Jeremy about king regions, for fun I spent some time
  //   trying to optimize an implementation and ultimately came up with what
  //   you see here. (It's used with permission in Mantissa; that was the
  //   original purpose.)
  if idx ==  4 { return 0; }
  if idx >= 24 { return 4; }
  const MAP : usize = 256597072250967;
  return ((MAP >> (idx*2)) & 3) + 1;
}

impl MiniState {
  pub const fn head_index(&self) -> usize
  {
    // This is abhorrent, but "for" cannot be used in const functions.
    let w = unsafe { std::mem::transmute::<_,&[u64; 2]>(&self.positions[0]) };
    let b = unsafe { std::mem::transmute::<_,&[u64; 2]>(&self.positions[1]) };
    let absent = (w[0] & 0x_80_80_80_80_80_80_80_80).count_ones()
               + (w[1] & 0x_80_80_80_80_80_80_80_80).count_ones()
               + (b[0] & 0x_80_80_80_80_80_80_80_80).count_ones()
               + (b[1] & 0x_80_80_80_80_80_80_80_80).count_ones();
    let mut men = 32 - (absent as usize);
    if self.variable[0].1 >= 0 { men += 1; }
    if self.variable[1].1 >= 0 { men += 1; }
    if      HEADS == 8 { return (men - 1) /  4; }
    else if HEADS == 4 { return (men - 1) /  8; }
    else if HEADS == 2 { return (men - 1) / 16; }
    else               { return 0;              }
  }
}

impl State {
  pub const fn head_index(&self) -> usize
  {
    let men = (self.sides[0] | self.sides[1]).count_ones() as usize;
    if men > 32 { return HEADS - 1; }
    if      HEADS == 8 { return (men - 1) /  4; }
    else if HEADS == 4 { return (men - 1) /  8; }
    else if HEADS == 2 { return (men - 1) / 16; }
    else               { return 0;              }
  }
/*
  pub fn initialize_nnue(&mut self)
  {
    let wk_idx =         self.boards[WhiteKing].trailing_zeros() as usize ;
    let bk_idx = vmirror(self.boards[BlackKing].trailing_zeros() as usize);
    let w_region = unsafe { &NETWORK.rn[king_region(wk_idx)] };
    let b_region = unsafe { &NETWORK.rn[king_region(bk_idx)] };

    self.s1.clear();

    // NOTE this is somewhat unsafe, but the only way
    //   I've found to prevent an unnecessary copy.
    self.s1.reserve(1);
    unsafe { self.s1.set_len(1); }
    let s1 = &mut self.s1[0];

    for n in 0..vN1 { s1[White][n] = simd_load!(w_region.b1, n); }
    for n in 0..vN1 { s1[Black][n] = simd_load!(b_region.b1, n); }

    for kind in KQRBNP {
      let mut sources = self.boards[White+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize)*64 + src;
        for n in 0..vN1 { s1[White][n] += simd_load!(w_region.w1[SameSide][x], n); }
        for n in 0..vN1 { s1[Black][n] += simd_load!(b_region.w1[OppoSide][x], n); }
        sources &= sources - 1;
      }
    }
    for kind in KQRBNP {
      let mut sources = self.boards[Black+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize)*64 + vmirror(src);
        for n in 0..vN1 { s1[White][n] += simd_load!(w_region.w1[OppoSide][x], n); }
        for n in 0..vN1 { s1[Black][n] += simd_load!(b_region.w1[SameSide][x], n); }
        sources &= sources - 1;
      }
    }
  }

  pub fn evaluate(&self) -> f32
  {
    unsafe {
      let head = &NETWORK.hd[self.head_index()];

      let s1 = &self.s1[self.s1.len()-1];

      let mut a1 : [[MaybeUninit<Simd32>; vN1]; 2] =
        [[MaybeUninit::uninit(); vN1], [MaybeUninit::uninit(); vN1]];
      for c in WB { for n in 0..vN1 { a1[c][n].write(relu_ps(s1[c][n])); } }
      let a1 = std::mem::transmute::<_,&mut [[Simd32; vN1]; 2]>(&mut a1);

      let mut s2 : [MaybeUninit<f32>; N2] = [MaybeUninit::uninit(); N2];
      let c = self.turn;
      for n in 0..N2 {
        let mut s_a = SIMD_ZERO;
        let mut s_b = SIMD_ZERO;
        let mut s_c = SIMD_ZERO;
        let mut s_d = SIMD_ZERO;
        for x in 0..vN1/4 {
          s_a += a1[c][x*4+0] * simd_load!(head.w2[n][SideToMove], x*4+0);
          s_b += a1[c][x*4+1] * simd_load!(head.w2[n][SideToMove], x*4+1);
          s_c += a1[c][x*4+2] * simd_load!(head.w2[n][SideToMove], x*4+2);
          s_d += a1[c][x*4+3] * simd_load!(head.w2[n][SideToMove], x*4+3);
        }
        for x in 0..vN1/4 {
          s_a += a1[!c][x*4+0] * simd_load!(head.w2[n][SideWaiting], x*4+0);
          s_b += a1[!c][x*4+1] * simd_load!(head.w2[n][SideWaiting], x*4+1);
          s_c += a1[!c][x*4+2] * simd_load!(head.w2[n][SideWaiting], x*4+2);
          s_d += a1[!c][x*4+3] * simd_load!(head.w2[n][SideWaiting], x*4+3);
        }
        let s = (s_a + s_b) + (s_c + s_d);
        s2[n].write(head.b2[n] + horizontal_sum(s));

        {
          let mut pos = 0.0;
          let mut neg = 0.0;
          let sm = std::mem::transmute::<_, &[f32; N1]>(&a1[ c]);
          for x in 0..N1 {
            let i = sm[x] * head.w2[n][SideToMove][x];
            if i > S2_INP_MAX { S2_INP_MAX = i; }
            if i < S2_INP_MIN { S2_INP_MIN = i; }
            if i > 0.0 { pos += i; }
            if i < 0.0 { neg += i; }
          }
          let sw = std::mem::transmute::<_, &[f32; N1]>(&a1[!c]);
          for x in 0..N1 {
            let i = sw[x] * head.w2[n][SideWaiting][x];
            if i > S2_INP_MAX { S2_INP_MAX = i; }
            if i < S2_INP_MIN { S2_INP_MIN = i; }
            if i > 0.0 { pos += i; }
            if i < 0.0 { neg += i; }
          }
          let b = head.b2[n];
          if b > 0.0 { pos += b; }
          if b < 0.0 { neg += b; }
          if pos > S2_WORST_MAX { S2_WORST_MAX = pos; }
          if neg < S2_WORST_MIN { S2_WORST_MIN = neg; }
          let s = pos + neg;
          if s > S2_MAX { S2_MAX = s; }
          if s < S2_MIN { S2_MIN = s; }
        }
      }
      let s2 = std::mem::transmute::<_,&mut [f32; N2]>(&mut s2);

      let mut s = SIMD_ZERO;
      for x in 0..vN2 { s += relu_ps(simd_load!(s2, x)) * simd_load!(head.w3, x); }
      let s3 = head.b3 + horizontal_sum(s);
      return s3;
    }
  }
*/
}

impl Network {
  pub fn save_image(&self, file : &str, unif : bool, vis : i8) -> std::io::Result<()>
  {
    let all = vis < 0;
    let region = if all { 0 } else { vis as usize };

    let aspect = if all { 32 } else { 8 };  // width of image in neurons
    let upscale = 2;                        // number of pixels per square

    let wasp = aspect;
    let hasp = N1 / aspect;

    let wnum = wasp;
    let hnum = hasp * if all { REGIONS } else { 1 };

    let width   = (6*8*upscale)*wnum + (6*wnum - 1) + (wasp - 1)*2;
    let height  = (2*8*upscale)*hnum + (2*hnum - 1) + (hasp - 1)*2;

    let border = [0, 0, 32];

    let mut scale = [0.0; N1];
    let mut unif_scale : f32 = 0.0;
    let mut count = 0;
    for n in 0..N1 {
      let mut bound : f32 = 0.0;
      let rs = if all || unif { 0..REGIONS } else { region..region+1 };
      for r in rs {
        for side in 0..2 {
          for x in 0..Np {
            let z = self.rn[r].w1[side][x][n].abs();
            bound = bound.max(z);
            unif_scale += z;
            if z != 0.0 { count += 1; }
          }
        }
      }
      scale[n] = bound;
    }
    unif_scale /= count as f32;

    let mut w = BufWriter::new(File::create(format!("{}.ppm", file))?);
    writeln!(&mut w, "P6")?;
    writeln!(&mut w, "{} {}", width, height)?;
    writeln!(&mut w, "255")?;

    for tile_row in 0..hasp {
      let rs = if all { 0..REGIONS } else { 0..1 };
      for subtile_row in rs {
        for subneuron_row in 0..2 {
          let topmost = tile_row == 0 && subtile_row == 0 && subneuron_row == 0;
          if !topmost {
            let pixels = if subtile_row == 0 && subneuron_row == 0 { 3 } else { 1 };
            for _ in 0..(width*pixels) { w.write(&border)?; }
          }
          for rank in (0..8).rev() {
            for _ in 0..upscale {
              for tile_column in 0..wasp {
                let n = tile_row * aspect + tile_column;

                for kind in 0..6 {
                  let leftmost = tile_column == 0 && kind == 0;
                  if !leftmost {
                    let pixels = if kind == 0 { 3 } else { 1 };
                    for _ in 0..pixels { w.write(&border)?; }
                  }
                  for file in 0..8 {
                    let r = if all { (REGIONS-1) - subtile_row } else { region };

                    let square = rank*8 + file;
                    let side = subneuron_row ^ 1;
                    let square = if side != 0 { vmirror(square) } else { square };
                    let x : usize = kind*64 + square;

                    let s = if unif { unif_scale } else { scale[n] };
                    if s == 0.0 {
                      for _ in 0..upscale { w.write(&[192, 64, 64])?; }
                      continue;
                    }

                    let w1 = self.rn[r].w1[side][x][n] / s;

                    let normed =
                      if unif { (1.0 + (w1 * -0.5).exp2()).recip() }
                      else    { (w1 + 1.0) * 0.5                   };
                    debug_assert!(
                      1.0 >= normed && normed >= 0.0,
                      "out of range ({} {} {} {} {})",
                      normed, w1, self.rn[r].w1[side][x][n], s,
                      if unif { "unif" } else { "indp" }
                    );

                    const C : f32 = 31.5;
                    let b1 = self.rn[r].b1[n];
                    let bias = if b1 > 0.0 { (b1/scale[n]) * (C*2.0) } else { 0.0 };
                    let red = bias + normed * (255.0 - bias*2.0);
                    let grn =        normed *  255.0            ;
                    let blu =    C + normed * (255.0 -    C*2.0);
                    let red = red.round() as u8;
                    let grn = grn.round() as u8;
                    let blu = blu.round() as u8;
                    for _ in 0..upscale { w.write(&[red, grn, blu])?; }
                  }
                }
              }
            }
          }
        }
      }
    }
    w.flush()?;
    let status = std::process::Command::new("magick")
      .arg(&format!("{}.ppm", file)).arg(&format!("{}.png", file)).status()?;
    if status.success() { std::fs::remove_file(format!("{}.ppm", file))?; }
    return Ok(());
  }

  /*
  pub fn stat(&self)
  {
    use std::mem::transmute;

    for r in 0..REGIONS {
      let region = &self.rn[r];
      println!("r{r} w1"); stat_slice(unsafe { transmute::<_,&[f32; Np*N1*2]>(&region.w1) });
      println!("r{r} b1"); stat_slice(&region.b1);
    }
    for h in 0..HEADS {
      let head = &self.hd[h];
      println!("h{h} w2"); stat_slice(unsafe { transmute::<_,&[f32; N1*N2*2]>(&head.w2) });
      println!("h{h} b2"); stat_slice(&head.b2);
      println!("h{h} w3"); stat_slice(&head.w3);
    }

    let mut fst_lo   = [0.0; N1*REGIONS];
    let mut fst_hi   = [0.0; N1*REGIONS];
    let mut fst_lo_b = [0.0; N1*REGIONS];
    let mut fst_hi_b = [0.0; N1*REGIONS];

    for r in 0..REGIONS {
      let region = &self.rn[r];
      for n in 0..N1 {
        let mut inp = [[0.0; Np]; 2];
        for x in 0..Np { inp[0][x] = region.w1[0][x][n]; }
        for x in 0..Np { inp[1][x] = region.w1[1][x][n]; }
        inp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Strictly speaking, we should check that inp[0..16] only contains only
        //   negative weights and inp[Np-16..Np] contains only positive weights.
        let lo : f32 = inp[0][  0  ..16].iter().sum::<f32>()
                     + inp[1][  0  ..16].iter().sum::<f32>();
        let hi : f32 = inp[0][Np-16..Np].iter().sum::<f32>()
                     + inp[1][Np-16..Np].iter().sum::<f32>();
        fst_lo  [N1*r + n] = lo;
        fst_hi  [N1*r + n] = hi;
        fst_lo_b[N1*r + n] = lo + region.b1[n];
        fst_hi_b[N1*r + n] = hi + region.b1[n];
      }
    }

    println!("first layer");
    let lo_min   = fst_lo  .iter().fold(0.0f32, |a, x| a.min(*x));
    let hi_max   = fst_hi  .iter().fold(0.0f32, |a, x| a.max(*x));
    let lo_min_b = fst_lo_b.iter().fold(0.0f32, |a, x| a.min(*x));
    let hi_max_b = fst_hi_b.iter().fold(0.0f32, |a, x| a.max(*x));
    println!("  {lo_min:+10.3} {hi_max:+10.3}");
    println!("  {lo_min_b:+10.3} {hi_max_b:+10.3}");

    /*
    for h in 0..HEADS {
      let head = &self.hd[h];

      // This assumes the first layer activations are clamped to [0, 1].

      let mut snd_lo   = [0.0; N2];
      let mut snd_hi   = [0.0; N2];
      let mut snd_lo_b = [0.0; N2];
      let mut snd_hi_b = [0.0; N2];

      for n in 0..N2 {
        let mut lo = 0.0;
        let mut hi = 0.0;
        for x in 0..N1 {
          let w = head.w2[n][x];
          if w < 0.0 { lo += w; }
          if w > 0.0 { hi += w; }
        }
        snd_lo  [n] = lo;
        snd_hi  [n] = hi;
        snd_lo_b[n] = lo + head.b2[n];
        snd_hi_b[n] = hi + head.b2[n];
      }

      println!("second layer head {h}");
      let lo_min   = snd_lo  .iter().fold(0.0f32, |a, x| a.min(*x));
      let hi_max   = snd_hi  .iter().fold(0.0f32, |a, x| a.max(*x));
      let lo_min_b = snd_lo_b.iter().fold(0.0f32, |a, x| a.min(*x));
      let hi_max_b = snd_hi_b.iter().fold(0.0f32, |a, x| a.max(*x));
      println!("  {lo_min:+10.3} {hi_max:+10.3}");
      println!("  {lo_min_b:+10.3} {hi_max_b:+10.3}");

      // This assumes the second layer activations are clamped to [0, 1].

      let mut thd_lo = 0.0;
      let mut thd_hi = 0.0;
      for n in 0..N2 {
        let w = head.w3[n];
        if w < 0.0 { thd_lo += w; }
        if w > 0.0 { thd_hi += w; }
      }

      println!("third layer head {h}");
      println!("  {thd_lo:+10.3} {thd_hi:+10.3}");
      println!("  {:+10.3} {:+10.3}", thd_lo+head.b3, thd_hi+head.b3);
    }
    */
  }
  */
}

/*
fn stat_slice(xs : &[f32])
{
  let len = xs.len();
  let mut xs = xs.to_vec();

  xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let min = xs[0];
  let p05 = xs[len/20];
  let med = xs[len/2];
  let p95 = xs[len*19/20];
  let max = xs[len-1];
  let avg = xs.iter().sum::<f32>() / (len as f32);
  println!("  {min:+10.6} {p05:+10.6} {med:+10.6} {p95:+10.6} {max:+10.6} ({avg:+10.6})");

  for x in xs.iter_mut() { *x = x.abs(); }
  xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let min = xs[0];
  let p05 = xs[len/20];
  let med = xs[len/2];
  let p95 = xs[len*19/20];
  let max = xs[len-1];
  let avg = xs.iter().sum::<f32>() / (len as f32);
  println!("  {min:+10.6} {p05:+10.6} {med:+10.6} {p95:+10.6} {max:+10.6} ({avg:+10.6})");
}
*/
