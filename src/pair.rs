#![macro_use]

use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Pair32 {
  pub value : i64
}

macro_rules! P { ($a:expr, $b:expr) => { Pair32::new($a, $b) } }

pub const PAIR_ZERO : Pair32 = Pair32 { value: 0 };

// Poor Man's Vectorization
//
//   Why does this work? Despite the name, it's best to think of this as a mathematical trick
//   rather than a programming trick, or your intuition will lead you astray. It's not so much
//   storing two 32-bit numbers side-by-side in a 64-bit number as forming a linear combination
//   of two numbers in such a way that they can be recovered after performing operations.
//
//   Two integers, x and y – which would normally be stored as signed 32-bit machine numbers –
//   are stored in a single 64-bit machine number, W.
//
//     W := pair(x, y) := (sext(repr(x)) << 32) + sext(repr(y))
//
//   Note that x and y are integers, but W, repr(x), and repr(y) are bit strings.
//
//     sign(x)  sign(y)   W interpreted as u64              W interpreted as i64
//        +        +          x⋅h + y                       x⋅h + y
//        −        +      q + x⋅h + y = q−|x|⋅h +   y       x⋅h + y = −|x|⋅h +  y
//        +        −          x⋅h + y ~   x⋅h   + q−|y|     x⋅h + y =  x⋅h   - |y|
//        −        −      q + x⋅h + y ~ q−|x|⋅h + q−|y|     x⋅h + y = −|x|⋅h − |y|
//
//   where h is 2^32 and q is 2^64. The cases 0/+, +/0, and −/0 are the same as +/+, +/+,
//   and −/+, respectively, but when the signs are 0/−, W interpreted as a u64 is q+y = q−|y|.
//   In every case, W interpreted as an i64 is x⋅h + y.
//
//   An interesting effect results when we consider W as two separate halves, however:
//
//     sign(x)  sign(y)   upper(W) intrp'd as i32   lower(W) intrp'd as i32
//        +        +      x                         y
//        −        +      x = −|x|                  y
//        +        −      x−1                       y
//        −        −      x−1 = −|x|−1              y
//
//   The implication is that we have to perform some correction when extracting repr(x) and
//   repr(y) from W:
//
//     repr(x) = truncate(W >> 32) + (y < 0 ? 1 : 0)
//     repr(y) = truncate(W)
//
//   Parallel addition of these 64-bit numbers works exactly like you'd hope:
//
//         fst  snd   pair(fst, snd) interpreted as i64
//          x    y    x⋅h + y
//     +    a    b    a⋅h + b
//     =   x+a  y+b   x⋅h + a⋅h + y + b = (x+a)⋅h + (y+b)
//
//   (This is assuming no overflow occurs, i.e. both x+a and y+b fit within signed 32-bit
//   numbers.)
//
//   Multiplication also works out:
//
//         fst  snd   pair(fst, snd) interpreted as i64
//          x    y    x⋅h + y
//     ⋅    s    s
//     =   x⋅s  y⋅s   (x⋅h + y) ⋅ s = (x⋅s)⋅h + (y⋅s)
//
//   (Again assuming that no overflow occurs, i.e. both x⋅s and y⋅s fit within signed 32-bit
//   numbers.)
//
//   Note that it was crucial that we defined W := ... + ..., using addition, rather than
//   W := ... | ..., using bitwise-or! It was also necessary to perform sign-extension rather
//   than zero-extension.

impl Pair32 {
  // From https://doc.rust-lang.org/reference/expressions/operator-expr.html#semantics:
  // ‣ casting between two integers of the same size (e.g. i32 -> u32) is a no-op
  // ‣ casting from a larger integer to a smaller integer (e.g. u32 -> u8) will truncate
  // ‣ casting from a smaller integer to a larger integer (e.g. u8 -> u32) will
  //   • zero-extend if the source is unsigned
  //   • sign-extend if the source is signed

  pub const fn new(u: i32, v : i32) -> Pair32
  {
    return Pair32 { value: ((v as i64) << 32) + (u as i64) };
  }

  pub const fn lower(self) -> i32
  {
    return self.value as i32;
  }

  pub const fn upper(self) -> i32
  {
    // When the lower half is negative, this addition causes a carry, adding in the 1 we need.
    return ((self.value + 0x0000_0000_8000_0000) >> 32) as i32;
  }
}

impl Add for Pair32 {
  type Output = Self;
  fn add(self, other : Self) -> Self { return Self { value: self.value + other.value }; }
}

impl Add<i32> for Pair32 {
  type Output = Self;
  fn add(self, other : i32) -> Self { return self + Self::new(other, other); }
}

impl Sub for Pair32 {
  type Output = Self;
  fn sub(self, other : Self) -> Self { return Self { value: self.value - other.value }; }
}

impl Sub<i32> for Pair32 {
  type Output = Self;
  fn sub(self, other : i32) -> Self { return self - Self::new(other, other); }
}

impl Mul<i16> for Pair32 {
  type Output = Self;
  fn mul(self, other : i16) -> Self { return Self { value: self.value * (other as i64) }; }
}

impl Mul<i32> for Pair32 {
  type Output = Self;
  fn mul(self, other : i32) -> Self { return Self { value: self.value * (other as i64) }; }
}

impl Mul<i64> for Pair32 {
  type Output = Self;
  fn mul(self, other : i64) -> Self { return Self { value: self.value * other }; }
}

impl AddAssign for Pair32 {
  fn add_assign(&mut self, other : Self) { *self = *self + other; }
}

impl AddAssign<i32> for Pair32 {
  fn add_assign(&mut self, other : i32) { *self = *self + other; }
}

impl SubAssign for Pair32 {
  fn sub_assign(&mut self, other : Self) { *self = *self - other; }
}

impl SubAssign<i32> for Pair32 {
  fn sub_assign(&mut self, other : i32) { *self = *self - other; }
}

impl MulAssign<i16> for Pair32 {
  fn mul_assign(&mut self, other : i16) { *self = *self * other; }
}

impl MulAssign<i32> for Pair32 {
  fn mul_assign(&mut self, other : i32) { *self = *self * other; }
}

impl MulAssign<i64> for Pair32 {
  fn mul_assign(&mut self, other : i64) { *self = *self * other; }
}
