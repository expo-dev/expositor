use std::simd::Simd;
use std::simd::num::SimdFloat;

#[cfg(target_feature="avx")]
use std::arch::x86_64::*;

// Number of single-precision floating-point numbers per vector
#[cfg(target_feature="avx")]
pub const LANES : usize = 8;

#[cfg(not(target_feature="avx"))]
pub const LANES : usize = 4;

/*
macro_rules! simd_load {
  ($a:expr, $b:expr) => {
    Simd::from_slice(&$a[$b*LANES .. ($b+1)*LANES])
  }
}

pub(crate) use simd_load;
*/

// See the following:
//   https://stackoverflow.com/questions/6996764
//   https://stackoverflow.com/questions/13219146
//   https://stackoverflow.com/questions/13879609
//   https://stackoverflow.com/questions/41303780
#[cfg(target_feature="avx")]
#[inline]
pub fn horizontal_sum(x : Simd<f32, 8>) -> f32
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

#[cfg(not(target_feature="avx"))]
#[inline]
pub fn horizontal_sum(x : Simd<f32, LANES>) -> f32
{
  return x.reduce_sum();
}

// #[cfg(target_feature="avx")]
// #[inline]
// pub fn relu_ps(x : Simd<f32, 8>) -> Simd<f32, 8>
// {
//   let x = __m256::from(x);
//   // let r = unsafe { _mm256_max_ps(x, _mm256_broadcast_ss(&0.0)) };
//   let r = unsafe {
//     _mm256_min_ps(
//       _mm256_max_ps(
//         x,
//         _mm256_broadcast_ss(&0.0)
//       ),
//       _mm256_broadcast_ss(&1.0)
//     )
//   };
//   return Simd::from(r);
// }

// #[cfg(not(target_feature="avx"))]
#[inline]
pub fn relu_ps(x : Simd<f32, LANES>) -> Simd<f32, LANES>
{
  return x.simd_max(Simd::splat(0.0));
  // return x.simd_max(Simd::splat(0.0)).simd_min(Simd::splat(1.0));
}

/* LEAKY ReLU  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// With -C opt-level=3 -C target-cpu=skylake this compiles to
//
//   .LCPI0_0:
//     .long 0x3d000000
//   relu_ps:
//     vbroadcastss ymm1, dword ptr [rip + .LCPI0_0]
//     vmulps ymm1, ymm0, ymm1
//     vmaxps ymm0, ymm0, ymm1
//
// but, frustratingly, with -C opt-level=3 -C target-cpu=skylake -C target-feature=-avx2,+avx
//   this compiles to
//
//   .LCPI0_0:
//     .long 0x3d000000 .long 0x3d000000 .long 0x3d000000 .long 0x3d000000
//     .long 0x3d000000 .long 0x3d000000 .long 0x3d000000 .long 0x3d000000
//   relu_ps:
//     vmulps ymm1, ymm0, ymmword ptr [rip + .LCPI0_0]
//     vmaxps ymm0, ymm0, ymm1
//
// despite the fact that the memory-to-register variant of vbroadcastss is part of AVX and
//   not just AVX2. (The register-to-register variant is, however, exclusive to AVX2.)
//
// Notably, _mm256_broadcast_ss in Rust is actually implemented as a call to _mm256_set1_ps,
//   which – if I'm recalling correctly – itself is implemented by simply constructing an
//   instance of the Simd type from an array of repeated values. Presumably, then, the only
//   reason we get vbroadcastss at all is because LLVM recognizes the idiom.
//
// The _mm256_set1_ps intrinsic is borrowed from Intel's compiler; they note that it can
//   compile to a sequence of instructions. The _mm256_broadcast_ss, however, is meant to
//   compile directly to vbroadcastss.
//
// I'm not sure which is faster. What makes a bigger difference here – an additional
//   instruction, or increased memory bandwidth? (a 32-byte versus 4-byte load)
//
#[cfg(target_feature="avx")]
#[inline]
pub fn relu_ps(x : Simd<f32, 8>) -> Simd<f32, 8>
{
  let x = __m256::from(x);
  let r = unsafe { _mm256_max_ps(x, _mm256_mul_ps(x, _mm256_broadcast_ss(&0.03125))) };
  return Simd::from(r);
}

#[cfg(not(target_feature="avx"))]
#[inline]
pub fn relu_ps(x : Simd<f32, LANES>) -> Simd<f32, LANES>
{
  return x.simd_max(x * Simd::splat(0.03125));
}

// The same caveat in re broadcast/set1 as above applies here.
#[cfg(target_feature="avx")]
#[inline]
pub fn d_relu_ps(x : Simd<f32, 8>) -> Simd<f32, 8>
{
  unsafe {
    let x = __m256::from(x);
    let z = _mm256_setzero_ps();
    let c = _mm256_cmp_ps(x, z, 1);   // less-than, ordered, signalling
    let p = _mm256_broadcast_ss(&1.00000);
    let s = _mm256_broadcast_ss(&0.96875);
    let r = _mm256_sub_ps(p, _mm256_and_ps(s, c));
    return Simd::from(r);
  }
}

#[cfg(not(target_feature="avx"))]
#[inline]
pub fn d_relu_ps(x : Simd<f32, LANES>) -> Simd<f32, LANES>
{
  return x.simd_lt(Simd::splat(0.0)).select(Simd::splat(0.03125), Simd::splat(1.0));
}

~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ */
