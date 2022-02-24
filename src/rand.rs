// NOTE that these are not thread-safe.

static mut RAND_STATE : u64 = 0;

pub fn set_rand(ofs : u64)
{
  // This is an arbitrary number (it was randomly chosen)
  unsafe { RAND_STATE = ofs.wrapping_add(1761173596); }
  for _ in 0..8 { rand(); }
}

pub fn reset_rand() { set_rand(0); }

pub fn rand() -> u32
{
  let a : u64 = 8093;
  let m : u64 = 0x0001000000000000;
  let c : u64 = 1;
  unsafe { RAND_STATE = RAND_STATE.wrapping_mul(a).wrapping_add(c) % m; }
  return (unsafe{RAND_STATE} >> 16) as u32;
}

// 0.0 to +1.0
pub fn uniform() -> f64 { return rand() as f64 / 4294967295.0; }

// -1.0 to +1.0
pub fn symmetric_uniform() -> f64 { return (rand() as f64 / 2147483647.5) - 1.0; }

// -1.0 to +1.0 but PMF is triangular (not constant)
pub fn triangular() -> f64
{
  let x = symmetric_uniform();
  // return if x >= 0.0 { 1.0 - (1.0 - x).sqrt() } else { (1.0 + x).sqrt() - 1.0 };
  return (1.0 - (1.0 - x.abs()).sqrt()).copysign(x);
}
