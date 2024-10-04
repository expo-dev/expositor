use crate::misc::{SQRT2, SQRT3, SQRT6};

// See §3.2 of Vol. 2 (Seminumerical Algorithms) of The Art of Computer
//   Programming by Donald Knuth.
//
// This is a Linear Congruential Generator (LCG) which has the form
//     X <- (aX + c) mod m.
//
// Here m is 2⁶⁴ (the word size of AMD 64) so that the modulo operation
//   happens implicitly.
//
// The period of an LCG has length m if and only if
//     (1) c is relatively prime to m,
//     (2) a − 1 is a multiple of p for every prime p that divides m, and
//     (3) a − 1 is a multiple of 4 if m is a multiple of 4.
//
// Since in our case the only prime divisor of m is 2, these conditions are
//     (1) c is odd (not divisible by 2),
//     (2) a − 1 is a multiple of 2, and
//     (3) a − 1 is a multiple of 4,
//   or more simply, the period of our LCG has length m if and only if c is odd
//   and a − 1 is divisible by 4.
//
// It is tempting to pick a = 2.pow(k) + 1 for some k (using Rust syntax),
//   because then the multiplication can be performed by a shift and addition.
//
// However, having a period of length m does not guarantee good output. We can
//   rule out additional constants by considering another metric, potency. The
//   potency, s, is the least positive integer such that
//     (a-1).pow(s) % m == 0.
//
// A potency of at least 5 seems to be necessary for good output, but when
//   a = 2.pow(k) + 1, the potency is less than 5 for k > 8, and when k ≤ 8,
//   we have a ≤ 257, and multipliers that are too small are problematic for
//   other reasons.
//
// This rules out multipliers of the form 2.pow(k) + 1. In general, potency
//   is necessary but not sufficient for randomness.
//
// The most powerful test for LCGs is the spectral test (see §3.3.4), which is
//   rather tedious to perform, and so it's here that we defer to searches for
//   good multipliers done by other people.
//
// One of the most common multipliers that I've seen used with m = 2⁶⁴ is
//     6364136223846793005,
//   which is mentioned by Knuth (p. 108) and credited to a C. E. Haynes.
//   I've unfortunately been unable to find any further information about
//   the origin of this constant, but regardless, it would be suitable here.
//
// A list of multipliers for m = 2⁶⁴ is provided in a recent paper titled
//     Computationally Easy, Spectrally Good Multipliers
//     for Congruential Pseudorandom Number Generators
//   by Guy Steele and Sebastiano Vigna [arxiv.org/pdf/2001.05304.pdf],
//   and in §10 (p. 17) they note that "we have verified empirically that
//   in several cases our multipliers provide LCGs that in isolation perform
//   better than previous proposals (e.g., better than Knuth’s MMIX LCG with
//   multiplier 6364136223846793005)."
//
// The multiplier used in the code below is from Table 6 (p. 18), where it
//   is given in hexadecimal as d134'2543'de82'ef95. It has five distinct
//   prime factors:
//     15074714826142052245 = 5 × 17 × 1277 × 2908441 × 47750621.
//
// Interestingly, the value of c has no effect on the spectral test.
//   The addend used in the code below is an arbitrary odd number,
//     10845525579672913913 = 7 × 19 × 4201 × 340369 × 57029069.

const A : u64 = 15074714826142052245;
const C : u64 = 10845525579672913913;

static mut GLOBAL_RNG_STATE : u64 = 0;

pub fn init_rand(seed : u64)
{
  unsafe { GLOBAL_RNG_STATE = seed; }
}

fn u64_rand_with(rng : &mut u64) -> u64
{
  let nxt = rng.wrapping_mul(A).wrapping_add(C);
  *rng = nxt;
  return nxt;
}

fn u64_rand() -> u64
{
  unsafe {
    let ptr = std::ptr::addr_of_mut!(GLOBAL_RNG_STATE);
    let nxt = (*ptr).wrapping_mul(A).wrapping_add(C);
    *ptr = nxt;
    return nxt;
  }
}

pub trait Rand {
  fn rand_with(_ : &mut u64) -> Self;
  fn rand() -> Self;
}

impl Rand for u32 {
  fn rand_with(rng : &mut u64) -> Self
  {
    return (u64_rand_with(rng) >> 32) as u32;
  }

  fn rand() -> Self
  {
    return (u64_rand() >> 32) as u32;
  }
}

impl Rand for u16 {
  fn rand_with(rng : &mut u64) -> Self
  {
    return (u64_rand_with(rng) >> 32) as u16;
  }

  fn rand() -> Self
  {
    return (u64_rand() >> 32) as u16;
  }
}

pub trait RandDist {
  fn vee() -> Self;
  fn uniform() -> Self;
  fn triangular() -> Self;
}

impl RandDist for f64 {
  // -1.41 to +1.41
  // mean is 0
  // variance is 1
  fn vee() -> Self
  {
    let x = u32::rand() as f64 / 2147483647.5 - 1.0;
    return x.abs().sqrt().copysign(x) * SQRT2;
  }

  // -1.73 to +1.73
  // mean is 0
  // variance is 1
  fn uniform() -> Self
  {
    return (u32::rand() as f64 / 2147483647.5 - 1.0) * SQRT3;
  }

  // -2.45 to +2.45
  // mean is 0
  // variance is 1
  fn triangular() -> Self
  {
    let x = u32::rand() as f64 / 2147483647.5 - 1.0;
    return (1.0 - (1.0 - x.abs()).sqrt()).copysign(x) * SQRT6;
  }
}

impl RandDist for f32 {
  fn vee() -> Self
  {
    let x = u32::rand() as f32 / 2147483647.5 - 1.0;
    return x.abs().sqrt().copysign(x) * SQRT2 as f32;
  }

  fn uniform() -> Self
  {
    return (u32::rand() as f32 / 2147483647.5 - 1.0) * SQRT3 as f32;
  }

  fn triangular() -> Self
  {
    let x = u32::rand() as f32 / 2147483647.5 - 1.0;
    return (1.0 - (1.0 - x.abs()).sqrt()).copysign(x) * SQRT6 as f32;
  }
}
