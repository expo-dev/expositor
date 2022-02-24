pub const VERSION : &'static str = env!("VERSION");
pub const BUILD   : &'static str = env!("BUILD");

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub const STDOUT : u64 = 1;
pub const STDERR : u64 = 2;

static mut STDOUT_ISATTY : Option<bool> = None;
static mut STDERR_ISATTY : Option<bool> = None;

#[allow(unused_variables)]
pub fn isatty(fd : u64) -> bool
{
  #[cfg(target_os="linux")]
  unsafe {
    // Memoize to avoid making syscalls
    if fd == STDERR && STDERR_ISATTY.is_some() { return STDERR_ISATTY.unwrap(); }
    if fd == STDOUT && STDOUT_ISATTY.is_some() { return STDOUT_ISATTY.unwrap(); }
    let ret : i64;
    let ioctl : u64 = 16;
    let tcgets : u64 = 0x5401;
    let mut empty : [u64; 3] = [0; 3];
    std::arch::asm!(
      "syscall"
      , inout("rax") ioctl => ret
      ,    in("rdi") fd
      ,    in("rsi") tcgets
      ,    in("rdx") &mut empty
      ,   out("rcx") _
      ,   out("r11") _
    );
    let ret = ret == 0;
    if fd == STDERR { STDERR_ISATTY = Some(ret); }
    if fd == STDOUT { STDOUT_ISATTY = Some(ret); }
    return ret;
  }
  #[allow(unreachable_code)]
  { return false; }
}

#[allow(unused_variables)]
pub fn set_stacksize(bytes : u64) -> bool
{
  #[cfg(target_os="linux")]
  unsafe {
    let ret : i64;
    let setrlimit : u64 = 160;
    let stacksize : u64 = 3;
    let rlim_struct : [u64; 2] = [bytes, 0xFFFFFFFFFFFFFFFF];
    std::arch::asm!(
      "syscall"
      , inout("rax") setrlimit => ret
      ,    in("rdi") stacksize
      ,    in("rsi") &rlim_struct
      ,   out("rdx") _
      ,   out("rcx") _
      ,   out("r11") _
    );
    return ret == 0;
  }
  #[allow(unreachable_code)]
  { return false; }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[inline]
#[cfg(target_feature="bmi2")]
#[target_feature(enable = "bmi2")]
pub unsafe fn pdep(a : u64, mask : u64) -> u64 {
  return std::arch::x86_64::_pdep_u64(a, mask);
}

#[inline]
#[cfg(not(target_feature="bmi2"))]
pub unsafe fn pdep(mut a : u64, mut mask : u64) -> u64 {
  let mut dst = 0;
  while mask != 0 {
    let idx = mask.trailing_zeros();
    dst |= (a & 1) << idx;
    a >>= 1;
    mask &= mask - 1;
  }
  return dst;
}

#[inline]
#[cfg(target_feature="bmi2")]
#[target_feature(enable = "bmi2")]
pub unsafe fn pext(a : u64, mask : u64) -> u64 {
  return std::arch::x86_64::_pext_u64(a, mask);
}

#[inline]
#[cfg(not(target_feature="bmi2"))]
pub unsafe fn pext(a : u64, mut mask : u64) -> u64 {
  let mut dst = 0;
  let mut k = 0;
  while mask != 0 {
    let idx = mask.trailing_zeros();
    dst |= ((a >> idx) & 1) << k;
    mask &= mask - 1;
    k += 1;
  }
  return dst;
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub fn relu(x : f32) -> f32 { return x.max(x / 32.0); }

pub fn d_relu(x : f32) -> f32 { return if x < 0.0 { 0.03125 } else { 1.0 }; }

pub fn compress(x : f32) -> f32
{
  // NOTE This is an approximation of logistic(logarithmic(x)),
  //   but scaled so that compress(1) = 1 and the asymptotes are ±2.
  return (1.0 + (x.abs() - 1.0) / (x.abs() + 1.0)).copysign(x);
}

pub fn d_compress(x : f32) -> f32
{
  let denom = x.abs() + 1.0;
  return 2.0 / (denom * denom);
}

/* ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
// Unused functions

pub fn reln(x : f32) -> f32
{
  return if x > 0.0 { x.ln_1p() } else { x / 32.0 };
}

pub fn d_reln(x : f32) -> f32
{
  return if x > 0.0 { 1.0 / (x + 1.0) } else if x < 0.0 { 0.03125 } else { 0.5 };
}

pub fn logistic(x : f32) -> f32
{
  return 1.0 / (1.0 + (-x).exp2()) - 0.5;
}

pub fn d_logistic(x : f32) -> f32
{
  let expon = (-x).exp2();
  let denom = 1.0 + expon;
  return expon * (2.0f32).ln() / (denom * denom);
}

pub fn logarithmic(x : f32) -> f32
{
  return ((1.0 + x.abs()).log2() * 2.0 / (3.0f32).log2()).copysign(x);
}

pub fn d_logarithmc(x : f32) -> f32
{
  return 2.0 / ((1.0 + x.abs()) * (3.0f32).ln());
}

pub fn inv_logarithmic(x : f32) -> f32
{
  return (3.0f32).powf(x / 2.0) - 1.0;
}

// Behaves as a soft clip for centipawn scores usually in
//   the range 0 to +10_00 but that occasionally exceed those bounds
// First takes centipawn scores in the range -infinity to +infinity
//   and maps them to the range 0 to +10_00
//   (for small x, remapped(x) ~ 2_50 + 1.5x)
// Then shifts to the range +90_00 to +99_99
pub fn canonicalize(score : i16) -> i16
{
  let remapped =
    if score < 0 {
      -250.0 / (score as f64 * 0.006 - 1.0)
    }
    else {
      1000.0 - 750.0 / (score as f64 * 0.002 + 1.0)
    };
  return INEVITABLE_MATE + remapped as i16;
}
*/
