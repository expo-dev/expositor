#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::arch::x86_64::*;
use std::simd::{*, num::SimdFloat, cmp::SimdPartialOrd};

pub const K : usize =  5;
pub const C : usize =  2; // (K - 1) / 2
pub const Z : usize = 12; // 8 + K - 1

pub type Kernel   = [f32;  40]; // 5 ranks × (5 files rounded up to 8)
pub type ZxtBoard = [f32; 152]; // ((12 ranks × 12 files) + 3 overrun) rounded up to 8×19
pub type Board    = [f32;  64]; // 8 rank × 8 files

// r,f in 0..8  (0..=7 )
// x,y in 0..5  (0..=4 )
//
//    w         w := weight
// a --> s      a := activation from previous layer
//              s := sum to next layer
//
// Forward
//   forall q
//     forall r,f { s[q][r][f] = sum p,x,y { a[p][r+x][f+y] · w[q][p][x][y] } }
//
// Backward
//   forall p,q
//     forall x,y { dw[q][p][x][y] = sum r,f { a[p][r+x][f+y] · ds[q][r][f] } }
//
// Propagation
//   forall p,q
//     forall x,y,r,f { da[p][r+x][f+y] += w[q][p][x][y] · ds[q][r][f] }

macro_rules! impossible {
  ($cond:expr) => { if $cond { unsafe { std::hint::unreachable_unchecked(); } } }
}

struct Guard<const N : usize> { }
impl <const N : usize> Guard<N> {
  const CHECK : () = assert!(N == (N / 8) * 8);
  fn assert() { let _ = Self::CHECK; }
}
macro_rules! assert_lanes {
  ($n:expr) => { Guard::<$n>::assert(); }
}

macro_rules! simd_load {
  ($a:expr, $b:expr) => { Simd::<f32, 8>::from_slice(&$a[$b .. $b+8]) }
}

macro_rules! simd_store {
  ($a:expr, $b:expr, $c:expr) => { $c.copy_to_slice(&mut $a[$b .. $b+8]) }
}

macro_rules! simd_incr {
  ($a:expr, $b:expr, $c:expr) => { simd_store!($a, $b, simd_load!($a, $b) + $c) }
}

fn horizontal_sum(x : Simd<f32, 8>) -> f32
{
  unsafe {
    let    x = __m256::from(x);
    let q_hi = _mm256_extractf128_ps(x, 1);
    let q_lo = _mm256_castps256_ps128(x);
    let d_lo = _mm_add_ps(q_hi, q_lo);
    let d_hi = _mm_movehl_ps(d_lo, d_lo);
    let s_lo = _mm_add_ps(d_lo, d_hi);
    let s_hi = _mm_shuffle_ps(s_lo, s_lo, 1);
    let    r = _mm_add_ss(s_lo, s_hi);
    return _mm_cvtss_f32(r);
  }
}

pub fn zero_extend<const N : usize>(b : &[Board; N], z : &mut [ZxtBoard; N])
{
  let ofs = C * Z + C;
  for n in 0..N {
    let nb = &b[n];
    let nz = &mut z[n];
    simd_store!(nz, ofs    , simd_load!(nb,  0));
    simd_store!(nz, ofs+Z  , simd_load!(nb,  8));
    simd_store!(nz, ofs+Z*2, simd_load!(nb, 16));
    simd_store!(nz, ofs+Z*3, simd_load!(nb, 24));
    simd_store!(nz, ofs+Z*4, simd_load!(nb, 32));
    simd_store!(nz, ofs+Z*5, simd_load!(nb, 40));
    simd_store!(nz, ofs+Z*6, simd_load!(nb, 48));
    simd_store!(nz, ofs+Z*7, simd_load!(nb, 56));
  }
}

pub fn mask_weights<const P : usize, const Q : usize>(w : &mut [[Kernel; P]; Q])
{
  for q in 0..Q {
    for p in 0..P {
      let wqp = &mut w[q][p];
      for x in 0..5 {
        wqp[x*8+5] = 0.0;
        wqp[x*8+6] = 0.0;
        wqp[x*8+7] = 0.0;
      }
    }
  }
}

/* ACTIVATION  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* pub fn sigm(x : f32) -> f32
{
  const M : f32 = 64.0;
  return (x + x.tanh() * (M - 1.0)) * (1.0 / M);
}

pub fn d_sigm(x : f32) -> f32
{
  const M : f32 = 64.0;
  let c = x.cosh();
  return ((M - 1.0) / (c * c) + 1.0) * (1.0 / M);
} */

/* pub fn sigm(x : f32) -> f32
{
  const R : f32 = 64.0;
  return
      x * ((R - 1.0) / R) / (1.0 + x * x).sqrt()
    + x * (1.0 / R);
}

pub fn d_sigm(x : f32) -> f32
{
  const R : f32 = 64.0;
  let d = 1.0 + x * x;
  return
      ((R - 1.0) / R) / (d * d * d).sqrt()
    + (1.0 / R);
} */

pub fn sigm_1(x : f32) -> f32
{
  const R : f32 = 64.0;
  let a = (R - 1.0) / R;
  let b = 1.0 / R;
  return (x * a) / (x * x + 1.0).sqrt() + x * b;
}

pub fn d_sigm_1(x : f32) -> f32
{
  const R : f32 = 64.0;
  let a = (R - 1.0) / R;
  let b = 1.0 / R;
  let d = x * x + 1.0;
  return a / (d * d * d).sqrt() + b;
}

pub fn sigm(x : Simd<f32, 8>) -> Simd<f32, 8>
{
  const R : f32 = 64.0;
  let a = Simd::<f32, 8>::splat((R - 1.0) / R);
  let b = Simd::<f32, 8>::splat(1.0 / R);
  let v1 = Simd::<f32, 8>::splat(1.0);
  return (x * a) / (x * x + v1).sqrt() + x * b;
}

pub fn d_sigm(x : Simd<f32, 8>) -> Simd<f32, 8>
{
  const R : f32 = 64.0;
  let a = Simd::<f32, 8>::splat((R - 1.0) / R);
  let b = Simd::<f32, 8>::splat(1.0 / R);
  let v1 = Simd::<f32, 8>::splat(1.0);
  let d = x * x + v1;
  return a / (d * d * d).sqrt() + b;
}

pub fn sigm_N_N<const N : usize>(
  s : &[f32; N], a : &mut [f32; N]
) {
  // assert_lanes!(N);
  let n = N / 8;
  for ofs in 0..n { simd_store!(a, ofs*8, sigm(simd_load!(s, ofs*8))); }
  if n * 8 != N { for x in (n*8)..N { a[x] = sigm_1(s[x]); } }
}

pub fn prop_sigm_N_N<const N : usize>(
  ds : &mut [f32; N], da : &[f32; N], s : &[f32; N]
) {
  // assert_lanes!(N);
  let n = N / 8;
  for ofs in 0..n {
    simd_store!(ds, ofs*8, simd_load!(da, ofs*8) * d_sigm(simd_load!(s, ofs*8)));
  }
  if n * 8 != N { for x in (n*8)..N { ds[x] = da[x] * d_sigm_1(s[x]); } }
}

fn relu(x : Simd<f32, 8>) -> Simd<f32, 8>
{
  const M : f32 = 1.0 / 64.0;
  const B : f32 = 1.0 - M;
  let vm = Simd::<f32, 8>::splat(M);
  let vb = Simd::<f32, 8>::splat(B);
  let xm = x * vm;
  return x.simd_max(xm - vb).simd_min(xm + vb);
}

fn d_relu(x : Simd<f32, 8>) -> Simd<f32, 8>
{
  const M : f32 = 1.0 / 64.0;
  let mask = x.abs().simd_gt(Simd::<f32, 8>::splat(1.0));
  return mask.select(Simd::<f32, 8>::splat(M), Simd::<f32, 8>::splat(1.0));
}

pub fn relu_88N_88N<const N : usize>(
  s : &[Board; N], a : &mut [Board; N]
) {
  for n in 0..N {
    let ns = &s[n];
    let na = &mut a[n];
    simd_store!(na,  0, relu(simd_load!(ns,  0)));
    simd_store!(na,  8, relu(simd_load!(ns,  8)));
    simd_store!(na, 16, relu(simd_load!(ns, 16)));
    simd_store!(na, 24, relu(simd_load!(ns, 24)));
    simd_store!(na, 32, relu(simd_load!(ns, 32)));
    simd_store!(na, 40, relu(simd_load!(ns, 40)));
    simd_store!(na, 48, relu(simd_load!(ns, 48)));
    simd_store!(na, 56, relu(simd_load!(ns, 56)));
  }
}

pub fn prop_relu_88N_88N<const N : usize>(
  ds : &mut [Board; N], da : &[Board; N], s : &[Board; N]
) {
  for n in 0..N {
    let ns = &mut ds[n];
    let na = &da[n];
    let ss = &s[n];
    simd_store!(ns,  0, simd_load!(na,  0) * d_relu(simd_load!(ss,  0)));
    simd_store!(ns,  8, simd_load!(na,  8) * d_relu(simd_load!(ss,  8)));
    simd_store!(ns, 16, simd_load!(na, 16) * d_relu(simd_load!(ss, 16)));
    simd_store!(ns, 24, simd_load!(na, 24) * d_relu(simd_load!(ss, 24)));
    simd_store!(ns, 32, simd_load!(na, 32) * d_relu(simd_load!(ss, 32)));
    simd_store!(ns, 40, simd_load!(na, 40) * d_relu(simd_load!(ss, 40)));
    simd_store!(ns, 48, simd_load!(na, 48) * d_relu(simd_load!(ss, 48)));
    simd_store!(ns, 56, simd_load!(na, 56) * d_relu(simd_load!(ss, 56)));
  }
}

pub fn relu_88N_z88N<const N : usize>(
  s : &[Board; N], a : &mut [ZxtBoard; N]
) {
  let ofs = C * Z + C;
  for n in 0..N {
    let ns = &s[n];
    let na = &mut a[n];
    simd_store!(na, ofs    , relu(simd_load!(ns,  0)));
    simd_store!(na, ofs+Z  , relu(simd_load!(ns,  8)));
    simd_store!(na, ofs+Z*2, relu(simd_load!(ns, 16)));
    simd_store!(na, ofs+Z*3, relu(simd_load!(ns, 24)));
    simd_store!(na, ofs+Z*4, relu(simd_load!(ns, 32)));
    simd_store!(na, ofs+Z*5, relu(simd_load!(ns, 40)));
    simd_store!(na, ofs+Z*6, relu(simd_load!(ns, 48)));
    simd_store!(na, ofs+Z*7, relu(simd_load!(ns, 56)));
  }
}

pub fn prop_relu_88N_z88N<const N : usize>(
  ds : &mut [Board; N], da : &[ZxtBoard; N], s : &[Board; N]
) {
  let ofs = C * Z + C;
  for n in 0..N {
    let ns = &mut ds[n];
    let na = &da[n];
    let ss = &s[n];
    simd_store!(ns,  0, simd_load!(na, ofs    ) * d_relu(simd_load!(ss,  0)));
    simd_store!(ns,  8, simd_load!(na, ofs+Z  ) * d_relu(simd_load!(ss,  8)));
    simd_store!(ns, 16, simd_load!(na, ofs+Z*2) * d_relu(simd_load!(ss, 16)));
    simd_store!(ns, 24, simd_load!(na, ofs+Z*3) * d_relu(simd_load!(ss, 24)));
    simd_store!(ns, 32, simd_load!(na, ofs+Z*4) * d_relu(simd_load!(ss, 32)));
    simd_store!(ns, 40, simd_load!(na, ofs+Z*5) * d_relu(simd_load!(ss, 40)));
    simd_store!(ns, 48, simd_load!(na, ofs+Z*6) * d_relu(simd_load!(ss, 48)));
    simd_store!(ns, 56, simd_load!(na, ofs+Z*7) * d_relu(simd_load!(ss, 56)));
  }
}

pub fn relu_z88N_z88N<const N : usize>(
  s : &[ZxtBoard; N], a : &mut [ZxtBoard; N]
) {
  let ofs = C * Z + C;
  for n in 0..N {
    let ns = &s[n];
    let na = &mut a[n];
    simd_store!(na, ofs    , relu(simd_load!(ns, ofs    )));
    simd_store!(na, ofs+Z  , relu(simd_load!(ns, ofs+Z  )));
    simd_store!(na, ofs+Z*2, relu(simd_load!(ns, ofs+Z*2)));
    simd_store!(na, ofs+Z*3, relu(simd_load!(ns, ofs+Z*3)));
    simd_store!(na, ofs+Z*4, relu(simd_load!(ns, ofs+Z*4)));
    simd_store!(na, ofs+Z*5, relu(simd_load!(ns, ofs+Z*5)));
    simd_store!(na, ofs+Z*6, relu(simd_load!(ns, ofs+Z*6)));
    simd_store!(na, ofs+Z*7, relu(simd_load!(ns, ofs+Z*7)));
  }
}

pub fn prop_relu_z88N_z88N<const N : usize>(
  ds : &mut [ZxtBoard; N], da : &[ZxtBoard; N], s : &[ZxtBoard; N]
) {
  let ofs = C * Z + C;
  for n in 0..N {
    let ns = &mut ds[n];
    let na = &da[n];
    let ss = &s[n];
    simd_store!(ns, ofs    , simd_load!(na, ofs    ) * d_relu(simd_load!(ss, ofs    )));
    simd_store!(ns, ofs+Z  , simd_load!(na, ofs+Z  ) * d_relu(simd_load!(ss, ofs+Z  )));
    simd_store!(ns, ofs+Z*2, simd_load!(na, ofs+Z*2) * d_relu(simd_load!(ss, ofs+Z*2)));
    simd_store!(ns, ofs+Z*3, simd_load!(na, ofs+Z*3) * d_relu(simd_load!(ss, ofs+Z*3)));
    simd_store!(ns, ofs+Z*4, simd_load!(na, ofs+Z*4) * d_relu(simd_load!(ss, ofs+Z*4)));
    simd_store!(ns, ofs+Z*5, simd_load!(na, ofs+Z*5) * d_relu(simd_load!(ss, ofs+Z*5)));
    simd_store!(ns, ofs+Z*6, simd_load!(na, ofs+Z*6) * d_relu(simd_load!(ss, ofs+Z*6)));
    simd_store!(ns, ofs+Z*7, simd_load!(na, ofs+Z*7) * d_relu(simd_load!(ss, ofs+Z*7)));
  }
}

/* CHANNEL-WIDE STATISTICS * * * * * * * * * * * * * * * * * * * * * * * * * */

pub fn pool_avg_88N_N<const N : usize>(a : &[Board; N], s : &mut [f32; N])
{
  for n in 0..N {
    let an = &a[n];
    let a0 = simd_load!(an,  0);
    let a1 = simd_load!(an,  8);
    let a2 = simd_load!(an, 16);
    let a3 = simd_load!(an, 24);
    let a4 = simd_load!(an, 32);
    let a5 = simd_load!(an, 40);
    let a6 = simd_load!(an, 48);
    let a7 = simd_load!(an, 56);
    let sum = horizontal_sum(
        ((a0 + a1) + (a2 + a3))
      + ((a4 + a5) + (a6 + a7))
    );
    s[n] = sum * (1.0 / 64.0);
  }
}

pub fn pool_max_88N_N<const N : usize>(a : &[Board; N], s : &mut [f32; N])
{
  for n in 0..N {
    let an = &a[n];
    let a0 = simd_load!(an,  0);
    let a1 = simd_load!(an,  8);
    let a2 = simd_load!(an, 16);
    let a3 = simd_load!(an, 24);
    let a4 = simd_load!(an, 32);
    let a5 = simd_load!(an, 40);
    let a6 = simd_load!(an, 48);
    let a7 = simd_load!(an, 56);
    let m01  =   a0.simd_max(a1);
    let m23  =   a2.simd_max(a3);
    let m45  =   a4.simd_max(a5);
    let m67  =   a6.simd_max(a7);
    let m0_3 =  m01.simd_max(m23);
    let m4_7 =  m45.simd_max(m67);
    let m0_8 = m0_3.simd_max(m4_7);
    s[n] = m0_8.reduce_max();
  }
}

pub fn back_pool_avg_88N_N<const N : usize>(da : &mut [Board; N], ds : &[f32; N])
{
  for n in 0..N {
    let vs = Simd::<f32, 8>::splat(ds[n] * (1.0 / 64.0));
    let an = &mut da[n];
    simd_incr!(an,  0, vs);
    simd_incr!(an,  8, vs);
    simd_incr!(an, 16, vs);
    simd_incr!(an, 24, vs);
    simd_incr!(an, 32, vs);
    simd_incr!(an, 40, vs);
    simd_incr!(an, 48, vs);
    simd_incr!(an, 56, vs);
  }
}

pub fn back_pool_max_88N_N<const N : usize>(
  da : &mut [Board; N], ds : &[f32; N], a : &[Board; N], s : &[f32; N]
) {
  // TODO optimize
  for n in 0..N {
    for x in 0..64 {
      if a[n][x] == s[n] { da[n][x] += ds[n]; }
    }
  }
}

/* FULLY-CONNECTED * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

pub fn fwd_P_1<const P : usize>(
  a : &[f32; P], w : &[f32; P]
) -> f32
{
  // assert_lanes!(P);
  let p = P / 8;
  let mut s = Simd::<f32, 8>::splat(0.0);
  for ofs in 0..p { s += simd_load!(a, ofs*8) * simd_load!(w, ofs*8); }
  if p * 8 == P { return horizontal_sum(s); }

  let s = horizontal_sum(s);
  let mut t = 0.0;
  for x in (p*8)..P { t += a[x] * w[x]; }
  return s + t;
}

pub fn fwd_P_Q<const P : usize, const Q : usize>(
  a : &[f32; P], w : &[[f32; P]; Q], s : &mut [f32; Q]
) {
  for q in 0..Q { s[q] += fwd_P_1::<P>(a, &w[q]); }
}

pub fn back_P_1<const P : usize>(
  a : &[f32; P], dw : &mut [f32; P], ds : f32
) {
  assert_lanes!(P);
  let v = Simd::<f32, 8>::splat(ds);
  for ofs in 0..(P/8) { simd_incr!(dw, ofs*8, simd_load!(a, ofs*8) * v); }
}

pub fn back_P_Q<const P : usize, const Q : usize>(
  a : &[f32; P], dw : &mut [[f32; P]; Q], ds : &[f32; Q]
) {
  // assert_lanes!(P);
  let p = P / 8;
  for ofs in 0..p {
    let va = simd_load!(a, ofs*8);
    for q in 0..Q {
      simd_incr!(dw[q], ofs*8, va * Simd::<f32, 8>::splat(ds[q]));
    }
  }
  if p * 8 == P { return; }

  for x in (p*8)..P { for q in 0..Q { dw[q][x] += a[x] * ds[q]; } }
}

pub fn back_bias_N<const N : usize>(
  db : &mut [f32; N], ds : &[f32; N]
) {
  // assert_lanes!(N);
  let n = N / 8;
  for ofs in 0..n { simd_incr!(db, ofs*8, simd_load!(ds, ofs*8)); }
  if n * 8 != N { for x in (n*8)..N { db[x] += ds[x]; } }
}

pub fn prop_P_Q<const P : usize, const Q : usize>(
  da : &mut [f32; P], w : &[[f32; P]; Q], ds : &[f32; Q]
) {
  // assert_lanes!(P);
  // TODO exchange loops (?)
  let p = P / 8;
  for q in 0..Q {
    let wq = &w[q];
    let vs = Simd::<f32, 8>::splat(ds[q]);
    for ofs in 0..p {
      simd_incr!(da, ofs*8, simd_load!(wq, ofs*8) * vs);
    }
    if p * 8 != P { for x in (p*8)..P { da[x] += wq[x] * ds[q]; } }
  }
}

/* ADDITION  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

pub fn add_88N_88N<const N : usize>(
  a : &[Board; N], b : &mut [Board; N]
) {
  for n in 0..N {
    let an = &a[n];
    let bn = &mut b[n];
    simd_incr!(bn,  0, simd_load!(an,  0));
    simd_incr!(bn,  8, simd_load!(an,  8));
    simd_incr!(bn, 16, simd_load!(an, 16));
    simd_incr!(bn, 24, simd_load!(an, 24));
    simd_incr!(bn, 32, simd_load!(an, 32));
    simd_incr!(bn, 40, simd_load!(an, 40));
    simd_incr!(bn, 48, simd_load!(an, 48));
    simd_incr!(bn, 56, simd_load!(an, 56));
  }
}

/* MULTIPLICATION  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

pub fn mul_88N_N_z88N<const N : usize>(
  a : &[Board; N], c : &[f32; N], p : &mut [ZxtBoard; N]
) {
  let ofs = C * Z + C;
  for n in 0..N {
    let an = &a[n];
    let pn = &mut p[n];
    let v = Simd::<f32, 8>::splat(c[n]);
    simd_store!(pn, ofs    , simd_load!(an,  0) * v);
    simd_store!(pn, ofs+Z  , simd_load!(an,  8) * v);
    simd_store!(pn, ofs+Z*2, simd_load!(an, 16) * v);
    simd_store!(pn, ofs+Z*3, simd_load!(an, 24) * v);
    simd_store!(pn, ofs+Z*4, simd_load!(an, 32) * v);
    simd_store!(pn, ofs+Z*5, simd_load!(an, 40) * v);
    simd_store!(pn, ofs+Z*6, simd_load!(an, 48) * v);
    simd_store!(pn, ofs+Z*7, simd_load!(an, 56) * v);
  }
}

pub fn mul_88N_N_z88N_prop_N<const N : usize>(
  a : &[Board; N], dc : &mut [f32; N], dp : &[ZxtBoard; N]
) {
  let ofs = C * Z + C;
  for n in 0..N {
    let an = &a[n];
    let pn = &dp[n];
    let r0 = simd_load!(pn, ofs    ) * simd_load!(an,  0);
    let r1 = simd_load!(pn, ofs+Z  ) * simd_load!(an,  8);
    let r2 = simd_load!(pn, ofs+Z*2) * simd_load!(an, 16);
    let r3 = simd_load!(pn, ofs+Z*3) * simd_load!(an, 24);
    let r4 = simd_load!(pn, ofs+Z*4) * simd_load!(an, 32);
    let r5 = simd_load!(pn, ofs+Z*5) * simd_load!(an, 40);
    let r6 = simd_load!(pn, ofs+Z*6) * simd_load!(an, 48);
    let r7 = simd_load!(pn, ofs+Z*7) * simd_load!(an, 56);
    let r = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    dc[n] = horizontal_sum(r);
  }
}

pub fn mul_88N_N_z88N_prop_88N<const N : usize>(
  da : &mut [Board; N], c : &[f32; N], dp : &[ZxtBoard; N]
) {
  let ofs = C * Z + C;
  for n in 0..N {
    let an = &mut da[n];
    let pn = &dp[n];
    let v = Simd::<f32, 8>::splat(c[n]);
    simd_store!(an,  0, simd_load!(pn, ofs    ) * v);
    simd_store!(an,  8, simd_load!(pn, ofs+Z  ) * v);
    simd_store!(an, 16, simd_load!(pn, ofs+Z*2) * v);
    simd_store!(an, 24, simd_load!(pn, ofs+Z*3) * v);
    simd_store!(an, 32, simd_load!(pn, ofs+Z*4) * v);
    simd_store!(an, 40, simd_load!(pn, ofs+Z*5) * v);
    simd_store!(an, 48, simd_load!(pn, ofs+Z*6) * v);
    simd_store!(an, 56, simd_load!(pn, ofs+Z*7) * v);
  }
}

pub fn dot_88N_N_88_prop_N<const N : usize>(
  a : &[Board; N], dc : &mut [f32; N], dp : &Board
) {
  for n in 0..N {
    let an = &a[n];
    let r0 = simd_load!(dp,  0) * simd_load!(an,  0);
    let r1 = simd_load!(dp,  8) * simd_load!(an,  8);
    let r2 = simd_load!(dp, 16) * simd_load!(an, 16);
    let r3 = simd_load!(dp, 24) * simd_load!(an, 24);
    let r4 = simd_load!(dp, 32) * simd_load!(an, 32);
    let r5 = simd_load!(dp, 40) * simd_load!(an, 40);
    let r6 = simd_load!(dp, 48) * simd_load!(an, 48);
    let r7 = simd_load!(dp, 56) * simd_load!(an, 56);
    let r = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    dc[n] = horizontal_sum(r);
  }
}

pub fn dot_88N_N_88_prop_88N<const N : usize>(
  da : &mut [Board; N], c : &[f32; N], dp : &Board
) {
  for n in 0..N {
    let an = &mut da[n];
    let v = Simd::<f32, 8>::splat(c[n]);
    simd_incr!(an,  0, simd_load!(dp,  0) * v);
    simd_incr!(an,  8, simd_load!(dp,  8) * v);
    simd_incr!(an, 16, simd_load!(dp, 16) * v);
    simd_incr!(an, 24, simd_load!(dp, 24) * v);
    simd_incr!(an, 32, simd_load!(dp, 32) * v);
    simd_incr!(an, 40, simd_load!(dp, 40) * v);
    simd_incr!(an, 48, simd_load!(dp, 48) * v);
    simd_incr!(an, 56, simd_load!(dp, 56) * v);
  }
}

/* 1×1 CONVOLUTION * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

pub fn fwd_88P_11P_88<const P : usize>(
  a : &[Board; P], w : &[f32; P], s : &mut Board
) {
  for p in 0..P {
    let ap = &a[p];
    let v = Simd::<f32, 8>::splat(w[p]);
    simd_incr!(s,  0, simd_load!(ap,  0) * v);
    simd_incr!(s,  8, simd_load!(ap,  8) * v);
    simd_incr!(s, 16, simd_load!(ap, 16) * v);
    simd_incr!(s, 24, simd_load!(ap, 24) * v);
    simd_incr!(s, 32, simd_load!(ap, 32) * v);
    simd_incr!(s, 40, simd_load!(ap, 40) * v);
    simd_incr!(s, 48, simd_load!(ap, 48) * v);
    simd_incr!(s, 56, simd_load!(ap, 56) * v);
  }
}

pub fn fwd_88P_11PQ_88Q<const P : usize, const Q : usize>(
  a : &[Board; P], w : &[[f32; P]; Q], s : &mut [Board; Q]
) {
  for q in 0..Q { fwd_88P_11P_88::<P>(a, &w[q], &mut s[q]); }
}

pub fn back_88P_11P_88<const P : usize>(
  a : &[Board; P], dw : &mut [f32; P], ds : &Board
) {
  for p in 0..P {
    let ap = &a[p];
    let r0 = simd_load!(ds,  0) * simd_load!(ap,  0);
    let r1 = simd_load!(ds,  8) * simd_load!(ap,  8);
    let r2 = simd_load!(ds, 16) * simd_load!(ap, 16);
    let r3 = simd_load!(ds, 24) * simd_load!(ap, 24);
    let r4 = simd_load!(ds, 32) * simd_load!(ap, 32);
    let r5 = simd_load!(ds, 40) * simd_load!(ap, 40);
    let r6 = simd_load!(ds, 48) * simd_load!(ap, 48);
    let r7 = simd_load!(ds, 56) * simd_load!(ap, 56);
    let r = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    dw[p] += horizontal_sum(r);
  }
}

pub fn back_88P_11PQ_88Q<const P : usize, const Q : usize>(
  a : &[Board; P], dw : &mut [[f32; P]; Q], ds : &[Board; Q]
) {
  for q in 0..Q { back_88P_11P_88::<P>(a, &mut dw[q], &ds[q]); }
}

// Use back_bias_88N for the 1×1 bias update

pub fn prop_88P_11PQ_88Q<const P : usize, const Q : usize>(
  da : &mut [Board; P], w : &[[f32; P]; Q], ds : &[Board; Q]
) {
  // TODO use this loop order instead?
  /* for q in 0..Q {
    let qs = &ds[q];
    let qs1 = simd_load!(qs,  0);
    let qs2 = simd_load!(qs,  8);
    let qs3 = simd_load!(qs, 16);
    let qs4 = simd_load!(qs, 24);
    let qs5 = simd_load!(qs, 32);
    let qs6 = simd_load!(qs, 40);
    let qs7 = simd_load!(qs, 48);
    let qs8 = simd_load!(qs, 56);
    let qw = &w[q];
    for p in 0..P {
      let pa = &mut da[p];
      let vw = Simd::<f32, 8>::splat(qw[p]);
      simd_incr!(pa,  0, qs1 * vw);
      simd_incr!(pa,  8, qs2 * vw);
      simd_incr!(pa, 16, qs3 * vw);
      simd_incr!(pa, 24, qs4 * vw);
      simd_incr!(pa, 32, qs5 * vw);
      simd_incr!(pa, 40, qs6 * vw);
      simd_incr!(pa, 48, qs7 * vw);
      simd_incr!(pa, 56, qs8 * vw);
    }
  } */
  for p in 0..P {
    let pa = &mut da[p];
    let mut pa1 = simd_load!(pa,  0);
    let mut pa2 = simd_load!(pa,  8);
    let mut pa3 = simd_load!(pa, 16);
    let mut pa4 = simd_load!(pa, 24);
    let mut pa5 = simd_load!(pa, 32);
    let mut pa6 = simd_load!(pa, 40);
    let mut pa7 = simd_load!(pa, 48);
    let mut pa8 = simd_load!(pa, 56);
    for q in 0..Q {
      let qs = &ds[q];
      let vw = Simd::<f32, 8>::splat(w[q][p]);
      pa1 += simd_load!(qs,  0) * vw;
      pa2 += simd_load!(qs,  8) * vw;
      pa3 += simd_load!(qs, 16) * vw;
      pa4 += simd_load!(qs, 24) * vw;
      pa5 += simd_load!(qs, 32) * vw;
      pa6 += simd_load!(qs, 40) * vw;
      pa7 += simd_load!(qs, 48) * vw;
      pa8 += simd_load!(qs, 56) * vw;
    }
    simd_store!(pa,  0, pa1);
    simd_store!(pa,  8, pa2);
    simd_store!(pa, 16, pa3);
    simd_store!(pa, 24, pa4);
    simd_store!(pa, 32, pa5);
    simd_store!(pa, 40, pa6);
    simd_store!(pa, 48, pa7);
    simd_store!(pa, 56, pa8);
  }
}

/* FORWARD * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

pub fn fwd_z88P_KKP_11<const P : usize>(
  a : &[ZxtBoard; P], w : &[Kernel; P], r : usize, f : usize
) -> f32
{
  impossible!(r > 7);
  impossible!(f > 7);
  let ofs = r * Z + f;
  let mut s = Simd::<f32, 8>::splat(0.0);
  for p in 0..P {
    let pa = &a[p];
    let pw = &w[p];
    let s0 = simd_load!(pa, ofs    ) * simd_load!(pw,  0);
    let s1 = simd_load!(pa, ofs+Z  ) * simd_load!(pw,  8);
    let s2 = simd_load!(pa, ofs+Z*2) * simd_load!(pw, 16);
    let s3 = simd_load!(pa, ofs+Z*3) * simd_load!(pw, 24);
    let s4 = simd_load!(pa, ofs+Z*4) * simd_load!(pw, 32);
    s += s2 + ((s0 + s1) + (s3 + s4));
  }
  return s.as_array()[..K].into_iter().sum::<f32>();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

fn fwd_z88P_KKP_88<const P : usize>(
  a : &[ZxtBoard; P], w : &[Kernel; P], s : &mut Board
) {
  let mut t = [[Simd::<f32, 8>::splat(0.0); 8]; 8];
  for p in 0..P {
    let pa = &a[p];
    let pw = &w[p];
    for r in 0..8 {
      for f in 0..8 {
        let ofs = r * Z + f;
        let s0 = simd_load!(pa, ofs    ) * simd_load!(pw,  0);
        let s1 = simd_load!(pa, ofs+Z  ) * simd_load!(pw,  8);
        let s2 = simd_load!(pa, ofs+Z*2) * simd_load!(pw, 16);
        let s3 = simd_load!(pa, ofs+Z*3) * simd_load!(pw, 24);
        let s4 = simd_load!(pa, ofs+Z*4) * simd_load!(pw, 32);
        t[r][f] = (t[r][f] + s2) + ((s0 + s1) + (s3 + s4));
      }
    }
  }
  for r in 0..8 {
    for f in 0..8 {
      let ofs = r * 8 + f;
      s[ofs] += t[r][f].as_array()[..K].into_iter().sum::<f32>();
    }
  }
}

pub fn fwd_z88P_KKPQ_88Q<const P : usize, const Q : usize>(
  a : &[ZxtBoard; P], w : &[[Kernel; P]; Q], s : &mut [Board; Q]
) {
  for q in 0..Q { fwd_z88P_KKP_88::<P>(a, &w[q], &mut s[q]); }
}

/* BACKWARD  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

pub fn back_z88P_KKP_11<const P : usize>(
  a : &[ZxtBoard; P], dw : &mut [Kernel; P], ds : f32, r : usize, f : usize
) {
  impossible!(r > 7);
  impossible!(f > 7);
  let ofs = r * Z + f;
  let vs = Simd::<f32, 8>::from_array([ds, ds, ds, ds, ds, 0.0, 0.0, 0.0]);
  for p in 0..P {
    let pa = &a[p];
    let pw = &mut dw[p];
    simd_incr!(pw,  0, simd_load!(pa, ofs    ) * vs);
    simd_incr!(pw,  8, simd_load!(pa, ofs+Z  ) * vs);
    simd_incr!(pw, 16, simd_load!(pa, ofs+Z*2) * vs);
    simd_incr!(pw, 24, simd_load!(pa, ofs+Z*3) * vs);
    simd_incr!(pw, 32, simd_load!(pa, ofs+Z*4) * vs);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

fn back_z88_KK_88(a : &ZxtBoard, dw : &mut Kernel, ds : &Board)
{
  let ds = unsafe { std::mem::transmute::<_, &[f32; 64]>(ds) };
  for x in 0..K {
    for y in 0..K {
      let ofs = x * Z + y;
      let dw0 = simd_load!(ds,  0) * simd_load!(a, ofs    );
      let dw1 = simd_load!(ds,  8) * simd_load!(a, ofs+Z  );
      let dw2 = simd_load!(ds, 16) * simd_load!(a, ofs+Z*2);
      let dw3 = simd_load!(ds, 24) * simd_load!(a, ofs+Z*3);
      let dw4 = simd_load!(ds, 32) * simd_load!(a, ofs+Z*4);
      let dw5 = simd_load!(ds, 40) * simd_load!(a, ofs+Z*5);
      let dw6 = simd_load!(ds, 48) * simd_load!(a, ofs+Z*6);
      let dw7 = simd_load!(ds, 56) * simd_load!(a, ofs+Z*7);
      let t = ((dw0 + dw1) + (dw2 + dw3)) + ((dw4 + dw5) + (dw6 + dw7));
      dw[x*8 + y] += horizontal_sum(t); /* NOTE “+=” rather than “=” */
    }
  }
}

pub fn back_z88P_KKPQ_88Q<const P : usize, const Q : usize>(
  a : &[ZxtBoard; P], dw : &mut [[Kernel; P]; Q], ds : &[Board; Q]
) {
  for q in 0..Q {
    for p in 0..P {
      back_z88_KK_88(&a[p], &mut dw[q][p], &ds[q]);
    }
  }
}

pub fn back_bias_88N<const N : usize>(
  db : &mut [Board; N], ds : &[Board; N]
) {
  for n in 0..N {
    let bn = &mut db[n];
    let sn = &ds[n];
    simd_incr!(bn,  0, simd_load!(sn,  0));
    simd_incr!(bn,  8, simd_load!(sn,  8));
    simd_incr!(bn, 16, simd_load!(sn, 16));
    simd_incr!(bn, 24, simd_load!(sn, 24));
    simd_incr!(bn, 32, simd_load!(sn, 32));
    simd_incr!(bn, 40, simd_load!(sn, 40));
    simd_incr!(bn, 48, simd_load!(sn, 48));
    simd_incr!(bn, 56, simd_load!(sn, 56));
  }
}

pub fn back_bias_z88N<const N : usize>(
  db : &mut [Board; N], ds : &[ZxtBoard; N]
) {
  let ofs = C * Z + C;
  for n in 0..N {
    let bn = &mut db[n];
    let sn = &ds[n];
    simd_incr!(bn,  0, simd_load!(sn, ofs    ));
    simd_incr!(bn,  8, simd_load!(sn, ofs+Z  ));
    simd_incr!(bn, 16, simd_load!(sn, ofs+Z*2));
    simd_incr!(bn, 24, simd_load!(sn, ofs+Z*3));
    simd_incr!(bn, 32, simd_load!(sn, ofs+Z*4));
    simd_incr!(bn, 40, simd_load!(sn, ofs+Z*5));
    simd_incr!(bn, 48, simd_load!(sn, ofs+Z*6));
    simd_incr!(bn, 56, simd_load!(sn, ofs+Z*7));
  }
}

/* PROPAGATION * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

pub fn prop_z88P_KKP_11<const P : usize>(
  da : &mut [ZxtBoard; P], w : &[Kernel; P], ds : f32, r : usize, f : usize
) {
  impossible!(r > 7);
  impossible!(f > 7);
  let ofs = r * Z + f;
  let vs = Simd::<f32, 8>::splat(ds);
  for p in 0..P {
    let pa = &mut da[p];
    let pw = &w[p];
    simd_incr!(pa, ofs    , simd_load!(pw,  0) * vs);
    simd_incr!(pa, ofs+Z  , simd_load!(pw,  8) * vs);
    simd_incr!(pa, ofs+Z*2, simd_load!(pw, 16) * vs);
    simd_incr!(pa, ofs+Z*3, simd_load!(pw, 24) * vs);
    simd_incr!(pa, ofs+Z*4, simd_load!(pw, 32) * vs);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

pub fn prop_z88P_KKPQ_88Q<const P : usize, const Q : usize>(
  da : &mut [ZxtBoard; P], w : &[[Kernel; P]; Q], ds : &[Board; Q]
) {
  for x in 0..K {
    for y in 0..K {
      let ofs = x * Z + y;
      for p in 0..P {
        let pa = &mut da[p];
        for q in 0..Q {
          let qs = &ds[q];
          let vw = Simd::<f32, 8>::splat(w[q][p][x * 8 + y]);
          simd_incr!(pa, ofs    , simd_load!(qs,  0) * vw);
          simd_incr!(pa, ofs+Z  , simd_load!(qs,  8) * vw);
          simd_incr!(pa, ofs+Z*2, simd_load!(qs, 16) * vw);
          simd_incr!(pa, ofs+Z*3, simd_load!(qs, 24) * vw);
          simd_incr!(pa, ofs+Z*4, simd_load!(qs, 32) * vw);
          simd_incr!(pa, ofs+Z*5, simd_load!(qs, 40) * vw);
          simd_incr!(pa, ofs+Z*6, simd_load!(qs, 48) * vw);
          simd_incr!(pa, ofs+Z*7, simd_load!(qs, 56) * vw);
        }
      }
    }
  }
}

/* BITMAPS * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
// NOTE that these weights are stored flipped along both x,y axes and that the
//   order of the p,q indices are swapped.

// NOTE that s and ∂E/∂s are zero-extended boards.

pub fn fwd_bmp_z88Q<const Q : usize>(
  a : u64, w : &[Kernel; Q], s : &mut [ZxtBoard; Q]
) {
  let mut bitset = a;
  while bitset != 0 {
    let u = bitset.trailing_zeros() as usize;
    let r = u / 8;
    let f = u % 8;
    impossible!(r > 7);
    impossible!(f > 7);
    let ofs = r * Z + f;
    for q in 0..Q {
      let qw = &w[q];
      let qs = &mut s[q];
      simd_incr!(qs, ofs    , simd_load!(qw,  0));
      simd_incr!(qs, ofs+Z  , simd_load!(qw,  8));
      simd_incr!(qs, ofs+Z*2, simd_load!(qw, 16));
      simd_incr!(qs, ofs+Z*3, simd_load!(qw, 24));
      simd_incr!(qs, ofs+Z*4, simd_load!(qw, 32));
    }
    bitset &= bitset - 1;
  }
}

pub fn back_bmp_z88Q<const Q : usize>(
  a : u64, dw : &mut [Kernel; Q], ds : &[ZxtBoard; Q]
) {
  let mut bitset = a;
  while bitset != 0 {
    let u = bitset.trailing_zeros() as usize;
    let r = u / 8;
    let f = u % 8;
    impossible!(r > 7);
    impossible!(f > 7);
    let ofs = r * Z + f;
    for q in 0..Q {
      let qw = &mut dw[q];
      let qs = &ds[q];
      simd_incr!(qw,  0, simd_load!(qs, ofs    ));
      simd_incr!(qw,  8, simd_load!(qs, ofs+Z  ));
      simd_incr!(qw, 16, simd_load!(qs, ofs+Z*2));
      simd_incr!(qw, 24, simd_load!(qs, ofs+Z*3));
      simd_incr!(qw, 32, simd_load!(qs, ofs+Z*4));
    }
    bitset &= bitset - 1;
  }
}
*/
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* fn clear_zero_extension(board : &mut ZxtBoard)
{
  for r in     0..C { for f in 0..Z { board[r][f] = 0.0; } }
  for r in (C+8)..Z { for f in 0..Z { board[r][f] = 0.0; } }
  for r in C..(C+8) {
    for f in     0..C { board[r][f] = 0.0; }
    for f in (C+8)..Z { board[r][f] = 0.0; }
  }
} */

/* fn validate_zero_extension<const N : usize>(b : &[ZxtBoard; N], msg : &str)
{
  for n in 0..N {
    let board = &b[n];
    for r in 0..C {
      for f in 0..Z {
        if board[r*Z+f] != 0.0 {
          panic!("{}: nonzero extension: {}, {}: {}", msg, r, f, board[r*Z+f]);
        }
      }
    }
    for r in (C+8)..Z {
      for f in 0..Z {
        if board[r*Z+f] != 0.0 {
          panic!("{}: nonzero extension: {}, {}: {}", msg, r, f, board[r*Z+f]);
        }
      }
    }
    for r in C..(C+8) {
      for f in 0..C {
        if board[r*Z+f] != 0.0 {
          panic!("{}: nonzero extension: {}, {}: {}", msg, r, f, board[r*Z+f]);
        }
      }
      for f in (C+8)..Z {
        if board[r*Z+f] != 0.0 {
          panic!("{}: nonzero extension: {}, {}: {}", msg, r, f, board[r*Z+f]);
        }
      }
    }
  }
} */

/* fn validate_weights<const P : usize, const Q : usize>(w : &[[Kernel; P]; Q], msg : &str)
{
  for q in 0..Q {
    for p in 0..P {
      let wqp = &w[q][p];
      for x in 0..K {
        for y in K..8 {
          if wqp[x*8+y] != 0.0 {
            panic!("{}: nonzero weight: {}, {}: {}", msg, x, y, wqp[x*8+y]);
          }
        }
      }
    }
  }
} */
