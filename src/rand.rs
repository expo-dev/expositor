use crate::misc::{SQRT2, SQRT3, SQRT6};

// NOTE that these are not thread-safe.

static mut STATE : u64 = 0;

pub fn set_rand(ofs : u64)
{
  // This is an arbitrary number (it was randomly chosen)
  unsafe { STATE = ofs.wrapping_add(1761173596); }
  for _ in 0..8 { rand(); }
}

pub fn reset_rand() { set_rand(0); }

pub fn rand() -> u32
{
  let a : u64 = 8093;
  let m : u64 = 0x0001000000000000;
  let c : u64 = 1;
  unsafe { STATE = STATE.wrapping_mul(a).wrapping_add(c) % m; }
  return (unsafe{STATE} >> 16) as u32;
}

// -1.41 to +1.41
// mean is 0
// variance is 1
pub fn vee() -> f64
{
  let x = rand() as f64 / 2147483647.5 - 1.0;
  return x.abs().sqrt().copysign(x) * SQRT2;
}

// -1.73 to +1.73
// mean is 0
// variance is 1
pub fn uniform() -> f64
{
  return (rand() as f64 / 2147483647.5 - 1.0) * SQRT3;
}

// -2.45 to +2.45
// mean is 0
// variance is 1
pub fn triangular() -> f64
{
  let x = rand() as f64 / 2147483647.5 - 1.0;
  return (1.0 - (1.0 - x.abs()).sqrt()).copysign(x) * SQRT6;
}
