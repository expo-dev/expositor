#![allow(non_upper_case_globals)]

use crate::color::{WB, Color::*};
use crate::misc::vmirror;
use crate::nnue::{
  SideToMove, SideWaiting,
  SameSide, OppoSide,
  REGIONS, HEADS,
  Np, N1, N2,
  NetworkHead, NetworkBody,
  Network,
  king_region
};
use crate::piece::{KQRBNP, Piece::{WhiteKing, BlackKing}};
use crate::state::State;

use std::mem::MaybeUninit;
use std::simd::prelude::*;
use std::arch::x86_64::*;

trait ArrayMap<U, T, const N : usize> {
  fn mapary<F>(&self, f : F) -> [T; N] where F : Fn(&U) -> T;
}

impl<U, T, const N : usize> ArrayMap<U, T, N> for [U; N] {
  fn mapary<F>(&self, f : F) -> [T; N] where F : Fn(&U) -> T
  {
    unsafe {
      let mut ts : [MaybeUninit::<T>; N] = [const { MaybeUninit::uninit() }; N];
      for n in 0..N { ts[n].write(f(&self[n])); }
      return MaybeUninit::array_assume_init(ts);
    }
  }
}

#[cfg(target_feature="avx2")]
macro_rules! vpmulhrsw {
  ($x:expr, $y:expr) => {
    Simd::from(_mm256_mulhrs_epi16(__m256i::from($x), __m256i::from($y)))
  }
}

#[cfg(not(target_feature="avx2"))]
macro_rules! vpmulhrsw {
  ($x:expr, $y:expr) => {
    Simd::from(_mm_mulhrs_epi16(__m128i::from($x), __m128i::from($y)))
  }
}

fn f32_to_i7p9(x : f32) -> i16
{
  // max +63.998047
  // min -64.000000
  // eps  +0.001953
  return (x * (1 << 9) as f32).round() as i16;
}

fn f32_to_i5p11(x : f32) -> i16
{
  // max +15.999512
  // min -16.000000
  // eps  +0.000488
  return (x * (1 << 11) as f32).round() as i16;
}

fn f32_to_i11p5(x : f32) -> i16
{
  // max +1023.968750
  // min -1024.000000
  // eps    +0.031250
  return (x * (1 << 5) as f32).round() as i16;
}

fn i11p5_to_f32(x : i16) -> f32
{
  return x as f32 / (1 << 5) as f32;
}

#[inline]
pub fn simd_copy<const N : usize>(a : &mut [i16; N], b : &[i16; N])
{
  for n in 0..N/16 {
    let ofs = n * 16;
    Simd::<i16, 16>::from_slice(    &b[ofs .. ofs+16])
                 .copy_to_slice(&mut a[ofs .. ofs+16]);
  }
}

#[inline]
pub fn simd_incr<const N : usize>(a : &mut [i16; N], b : &[i16; N])
{
  for n in 0..N/16 {
    let ofs = n * 16;
    let va = Simd::<i16, 16>::from_slice(&a[ofs .. ofs+16]);
    let vb = Simd::<i16, 16>::from_slice(&b[ofs .. ofs+16]);
    (va + vb).copy_to_slice(&mut a[ofs .. ofs+16]);
  }
}

#[inline]
pub fn simd_decr<const N : usize>(a : &mut [i16; N], b : &[i16; N])
{
  for n in 0..N/16 {
    let ofs = n * 16;
    let va = Simd::<i16, 16>::from_slice(&a[ofs .. ofs+16]);
    let vb = Simd::<i16, 16>::from_slice(&b[ofs .. ofs+16]);
    (va - vb).copy_to_slice(&mut a[ofs .. ofs+16]);
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[derive(Clone, PartialEq)]
#[repr(align(32))]
pub struct QuantizedBody {
  pub w1 : [[[i16; N1]; Np]; 2],  // weight[from-inp-side][from-input][to-first]
  pub b1 :   [i16; N1],           // bias[first]
}

#[derive(Clone, PartialEq)]
#[repr(align(32))]
pub struct QuantizedHead {
  pub w2 : [[[i16; N1]; 2]; N2],  // weight[to-second][from-fst-side][from-first]
  pub b2 :   [i16; N2],           // bias[second]

  pub w3 :   [f32; N2],           // weight[from-second]
  pub b3 :    f32,                // bias
}

pub struct QuantizedNetwork {
  pub rn : [QuantizedBody; REGIONS],
  pub hd : [QuantizedHead; HEADS]
}

pub static mut QUANTIZED : QuantizedNetwork = QuantizedNetwork::zero();

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

impl QuantizedHead {
  pub fn from(hd : &NetworkHead) -> Self
  {
    return Self {
      w2: hd.w2.mapary(|n| n.mapary(|c| c.mapary(|x| f32_to_i5p11(*x)))),
      b2: hd.b2.mapary(|n| f32_to_i11p5(*n)),
      w3: hd.w3,
      b3: hd.b3
    };
  }
}

impl QuantizedBody {
  pub fn from(rn : &NetworkBody) -> Self
  {
    return Self {
      w1: rn.w1.mapary(|c| c.mapary(|x| x.mapary(|n| f32_to_i7p9(*n)))),
      b1: rn.b1.mapary(|n| f32_to_i7p9(*n))
    };
  }
}

impl QuantizedNetwork {
  pub const fn zero() -> Self
  {
    const SZ : usize = std::mem::size_of::<QuantizedNetwork>();
    union Empty {
      ary : [u8; SZ],
      net : std::mem::ManuallyDrop<QuantizedNetwork>
    }
    const ZERO : Empty = Empty { ary: [0; SZ] };
    return std::mem::ManuallyDrop::<QuantizedNetwork>::into_inner(unsafe { ZERO.net });
  }

  pub fn from(network : &Network) -> Self
  {
    return Self {
      rn: network.rn.mapary(|rn| QuantizedBody::from(&rn)),
      hd: network.hd.mapary(|hd| QuantizedHead::from(&hd))
    };
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

impl State {
  pub fn initialize_nnue(&mut self)
  {
    let wk_idx =         self.boards[WhiteKing].trailing_zeros() as usize ;
    let bk_idx = vmirror(self.boards[BlackKing].trailing_zeros() as usize);
    let w_region = unsafe { &QUANTIZED.rn[king_region(wk_idx)] };
    let b_region = unsafe { &QUANTIZED.rn[king_region(bk_idx)] };

    self.s1.clear();

    self.s1.reserve(1);
    unsafe { self.s1.set_len(1); }
    let s1 = &mut self.s1[0];

    simd_copy(&mut s1[White], &w_region.b1);
    simd_copy(&mut s1[Black], &b_region.b1);

    for kind in KQRBNP {
      let mut sources = self.boards[White+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize)*64 + src;
        simd_incr(&mut s1[White], &w_region.w1[SameSide][x]);
        simd_incr(&mut s1[Black], &b_region.w1[OppoSide][x]);
        sources &= sources - 1;
      }
    }
    for kind in KQRBNP {
      let mut sources = self.boards[Black+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize)*64 + vmirror(src);
        simd_incr(&mut s1[White], &w_region.w1[OppoSide][x]);
        simd_incr(&mut s1[Black], &b_region.w1[SameSide][x]);
        sources &= sources - 1;
      }
    }
  }

  pub fn evaluate(&self) -> f32
  {
    type Simd256 = Simd<i16, 16>;
    unsafe {
      // ↓↓↓ DEBUG ↓↓↓
      /*
      let fa1;
      let fs2;
      let fs3;
      {
        use crate::nnue::NETWORK;

        let wk_idx =         self.boards[WhiteKing].trailing_zeros() as usize ;
        let bk_idx = vmirror(self.boards[BlackKing].trailing_zeros() as usize);
        let w_region = &NETWORK.rn[king_region(wk_idx)];
        let b_region = &NETWORK.rn[king_region(bk_idx)];
        let head = &NETWORK.hd[self.head_index()];

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

        for c in WB { for n in 0..N1 { s1[c][n] = s1[c][n].max(0.0); } }
        fa1 = s1;

        let mut s2 = [0.0; N2];
        for n in 0..N2 {
          let mut s = 0.0;
          let c = self.turn;
          for x in 0..N1 { s += fa1[ c][x] * head.w2[n][SideToMove ][x]; }
          for x in 0..N1 { s += fa1[!c][x] * head.w2[n][SideWaiting][x]; }
          s2[n] = head.b2[n] + s;
        }
        fs2 = s2;

        let mut s = 0.0;
        for x in 0..N2 { s += fs2[x].max(0.0) * head.w3[x]; }
        fs3 = head.b3 + s;
      }
      */
      // ↑↑↑ DEBUG ↑↑↑

      let head = &QUANTIZED.hd[self.head_index()];

      let s1 = &self.s1[self.s1.len()-1];

      let mut a1 : [[MaybeUninit<Simd256>; N1/16]; 2] =
        [[MaybeUninit::uninit(); N1/16], [MaybeUninit::uninit(); N1/16]];

      for c in WB {
        for n in 0..N1/16 {
          let ofs = n * 16;
          a1[c][n].write(
            Simd256::from_slice(&s1[c][ofs .. ofs+16]).simd_max(Simd::splat(0))
          );
        }
      }

      let mut s2 : [MaybeUninit<i16>; N2] = [MaybeUninit::uninit(); N2];
      let c = self.turn;

      #[cfg(target_feature="avx2")]
      {
        let a1 = std::mem::transmute::<_,&mut [[Simd256; N1/16]; 2]>(&mut a1);
        const SIMD_ZERO : Simd256 = Simd::from_array([0; 16]);
        for n in 0..N2 {
          let mut s_a = SIMD_ZERO;
          let mut s_b = SIMD_ZERO;
          let mut s_c = SIMD_ZERO;
          let mut s_d = SIMD_ZERO;
          for x in 0..N1/64 {
            let ofs = x * 64;
            s_a += vpmulhrsw!(a1[ c][x*4+0], Simd256::from_slice(&head.w2[n][SideToMove ][ofs    .. ofs+16]));
            s_b += vpmulhrsw!(a1[ c][x*4+1], Simd256::from_slice(&head.w2[n][SideToMove ][ofs+16 .. ofs+32]));
            s_c += vpmulhrsw!(a1[ c][x*4+2], Simd256::from_slice(&head.w2[n][SideToMove ][ofs+32 .. ofs+48]));
            s_d += vpmulhrsw!(a1[ c][x*4+3], Simd256::from_slice(&head.w2[n][SideToMove ][ofs+48 .. ofs+64]));
          }
          for x in 0..N1/64 {
            let ofs = x * 64;
            s_a += vpmulhrsw!(a1[!c][x*4+0], Simd256::from_slice(&head.w2[n][SideWaiting][ofs    .. ofs+16]));
            s_b += vpmulhrsw!(a1[!c][x*4+1], Simd256::from_slice(&head.w2[n][SideWaiting][ofs+16 .. ofs+32]));
            s_c += vpmulhrsw!(a1[!c][x*4+2], Simd256::from_slice(&head.w2[n][SideWaiting][ofs+32 .. ofs+48]));
            s_d += vpmulhrsw!(a1[!c][x*4+3], Simd256::from_slice(&head.w2[n][SideWaiting][ofs+48 .. ofs+64]));
          }
          let s = (s_a + s_b) + (s_c + s_d);
          s2[n].write(head.b2[n] + s.reduce_sum()); // TODO optimize horizontal sum
        }
      }
      #[cfg(not(target_feature="avx2"))]
      {
        type Simd128 = Simd<i16, 8>;
        let a1 = std::mem::transmute::<_,&mut [[Simd128; N1/8]; 2]>(&mut a1);
        const SIMD_ZERO : Simd128 = Simd::from_array([0; 8]);
        for n in 0..N2 {
          let mut s_a = SIMD_ZERO;
          let mut s_b = SIMD_ZERO;
          let mut s_c = SIMD_ZERO;
          let mut s_d = SIMD_ZERO;
          for x in 0..N1/32 {
            let ofs = x * 32;
            s_a += vpmulhrsw!(a1[ c][x*4+0], Simd128::from_slice(&head.w2[n][SideToMove ][ofs    .. ofs+ 8]));
            s_b += vpmulhrsw!(a1[ c][x*4+1], Simd128::from_slice(&head.w2[n][SideToMove ][ofs+ 8 .. ofs+16]));
            s_c += vpmulhrsw!(a1[ c][x*4+2], Simd128::from_slice(&head.w2[n][SideToMove ][ofs+16 .. ofs+24]));
            s_d += vpmulhrsw!(a1[ c][x*4+3], Simd128::from_slice(&head.w2[n][SideToMove ][ofs+24 .. ofs+32]));
          }
          for x in 0..N1/32 {
            let ofs = x * 32;
            s_a += vpmulhrsw!(a1[!c][x*4+0], Simd128::from_slice(&head.w2[n][SideWaiting][ofs    .. ofs+ 8]));
            s_b += vpmulhrsw!(a1[!c][x*4+1], Simd128::from_slice(&head.w2[n][SideWaiting][ofs+ 8 .. ofs+16]));
            s_c += vpmulhrsw!(a1[!c][x*4+2], Simd128::from_slice(&head.w2[n][SideWaiting][ofs+16 .. ofs+24]));
            s_d += vpmulhrsw!(a1[!c][x*4+3], Simd128::from_slice(&head.w2[n][SideWaiting][ofs+24 .. ofs+32]));
          }
          let s = (s_a + s_b) + (s_c + s_d);
          s2[n].write(head.b2[n] + s.reduce_sum()); // TODO optimize horizontal sum
        }
      }
      let s2 = std::mem::transmute::<_,&mut [i16; N2]>(&mut s2);

      let mut s = 0.0;
      for x in 0..N2 { s += i11p5_to_f32(s2[x]).max(0.0) * head.w3[x]; }
      let s3 = head.b3 + s;
      // ↓↓↓ DEBUG ↓↓↓
      /*
      for c in WB {
        for n in 0..N1 {
          let exact  = fa1[c][n];
          let approx =  a1[c][n/16][n%16];
          // i7p9 to f32
          let cast = approx as f32 / (1 << 9) as f32;
          let diff = (cast - exact).abs();
          if diff > (1.0 / 64.0) {
            eprintln!("{c} {n} {exact} {approx} {cast}");
          }
          if exact.abs() > 0.25 && (diff / exact.abs()) > (1.0 / 16.0) {
            eprintln!("1/16 {c} {n} {exact} {approx} {cast}");
          }
          if exact.abs() > 0.5 && (diff / exact.abs()) > (1.0 / 32.0) {
            eprintln!("1/32 {c} {n} {exact} {approx} {cast}");
          }
          if exact.abs() > 1.0 && (diff / exact.abs()) > (1.0 / 64.0) {
            eprintln!("1/32 {c} {n} {exact} {approx} {cast}");
          }
        }
      }
      for n in 0..N2 {
        let exact  = fs2[n];
        let approx =  s2[n];
        // i11p5 to f32
        let cast = approx as f32 / (1 << 5) as f32;
        let diff = (cast - exact).abs();
        if diff > 0.5 {
          eprintln!("     {n} {exact} {approx} {cast}");
        }
        if exact.abs() > 0.50 && (diff / exact.abs()) > (1.0 / 1.25) {
          eprintln!("1/1  {n} {exact} {approx} {cast}");
        }
        if exact.abs() > 1.00 && (diff / exact.abs()) > (1.0 / 2.50) {
          eprintln!("1/2  {n} {exact} {approx} {cast}");
        }
        if exact.abs() > 2.00 && (diff / exact.abs()) > (1.0 / 5.00) {
          eprintln!("1/5  {n} {exact} {approx} {cast}");
        }
        if exact.abs() > 4.00 && (diff / exact.abs()) > (1.0 / 10.00) {
          eprintln!("1/10 {n} {exact} {approx} {cast}");
        }
      }
      let exact  = fs3;
      let approx =  s3;
      let diff = (approx - exact).abs();
      if diff > 0.125 {
        eprintln!("{exact} {approx}");
      }
      if exact.abs() > 0.25 && (diff / exact.abs()) > (1.0 /  8.0) {
        eprintln!("1/8  {exact} {approx}");
      }
      if exact.abs() > 0.50 && (diff / exact.abs()) > (1.0 / 16.0) {
        eprintln!("1/16 {exact} {approx}");
      }
      if exact.abs() > 1.00 && (diff / exact.abs()) > (1.0 / 24.0) {
        eprintln!("1/24 {exact} {approx}");
      }
      */
      // ↑↑↑ DEBUG ↑↑↑
      return s3;
    }
  }
}
