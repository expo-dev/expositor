pub const VERSION : &str = env!("VERSION");
pub const BUILD   : &str = env!("BUILD");

pub const STDOUT : u64 = 1;
pub const STDERR : u64 = 2;

pub static mut STDOUT_ISATTY : Option<bool> = None;
pub static mut STDERR_ISATTY : Option<bool> = None;

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
    let mut empty : [u64; 8] = [0; 8];
    // The rax, rdi, rsi, rdx, r10, r8, and r9 registers are caller-saved
    //   registers in the kernel syscall ABI; rax is used for the return value
    //   and the other six are used for arguments. The syscall instruction saves
    //   the current rip (which the kernel will return to after the call) in rcx
    //   and the flags are stored in r11.
    std::arch::asm!(
      "syscall"
      , inout("rax") ioctl => ret
      ,    in("rdi") fd
      ,    in("rsi") tcgets
      ,    in("rdx") &mut empty
      ,   out("r10") _
      ,   out("r8" ) _
      ,   out("r9" ) _
      ,   out("rcx") _
      ,   out("r11") _
    );
    let ret = ret == 0;
    if fd == STDERR { STDERR_ISATTY = Some(ret); }
    if fd == STDOUT { STDOUT_ISATTY = Some(ret); }
    return ret;
  }
  #[cfg(not(target_os="linux"))]
  unsafe {
    if fd == STDERR && STDERR_ISATTY.is_some() { return STDERR_ISATTY.unwrap(); }
    if fd == STDOUT && STDOUT_ISATTY.is_some() { return STDOUT_ISATTY.unwrap(); }
    return false;
  }
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
      ,   out("r10") _
      ,   out("r8" ) _
      ,   out("r9" ) _
      ,   out("rcx") _
      ,   out("r11") _
    );
    return ret == 0;
  }
  #[cfg(not(target_os="linux"))]
  { return false; }
}

// NOTE that this usually returns the number of logical (not physical) cores.
pub fn num_cores() -> usize
{
  return if let Ok(n) = std::thread::available_parallelism() { n.get() } else { 1 };
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

// swap_bytes can be used instead
/* #[inline]
pub fn bswap(a : u64) -> u64
{
  return unsafe { std::arch::x86_64::_bswap64(a as i64) as u64 };
} */

pub fn harsh_compress(x : f32) -> f32
{
  return (1.0 + (x.abs() - 1.0) / (x.abs() + 1.0)).copysign(x);
}

pub fn d_harsh_compress(x : f32) -> f32
{
  let denom = x.abs() + 1.0;
  return 2.0 / (denom * denom);
}

pub fn compress(x : f32) -> f32
{
  return (1.0 + (x.abs()*0.5 - 1.0) / (x.abs()*0.5 + 1.0)).copysign(x);
}

pub fn d_compress(x : f32) -> f32
{
  let denom = x.abs() + 2.0;
  return 4.0 / (denom * denom);
}

pub fn gentle_compress(x : f32) -> f32
{
  let thd = 0.333_333_333;
  return (1.0 + (x.abs()*thd - 1.0) / (x.abs()*thd + 1.0)).copysign(x);
}

pub fn d_gentle_compress(x : f32) -> f32
{
  let denom = x.abs() + 3.0;
  return 6.0 / (denom * denom);
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

pub struct SetU64 {
  ary : Vec<u64>,
  sz  : usize,
  lg  : u8,
}

impl SetU64 {
  pub fn new() -> Self
  {
    return Self { ary: vec![0; 8], sz: 0, lg: 3 };
  }

  pub fn len(&self) -> usize
  {
    return self.sz;
  }

  pub fn capacity(&self) -> usize
  {
    return 1 << self.lg;
  }

  pub fn insert(&mut self, x : u64) -> bool
  {
    assert!(x != 0);
    let mask = (1 << self.lg) - 1;
    let mut idx = x as usize & mask;
    loop {
      let key = self.ary[idx];
      if key == x { return false; }
      if key == 0 { break; }
      idx = (idx + 1) & mask;
    }
    self.ary[idx] = x;
    self.sz += 1;
    if self.len()*2 > self.capacity() { self.resize(); }
    return true;
  }

  #[inline]
  fn resize(&mut self)
  {
    self.lg += 1;
    let cap = 1 << self.lg;
    let mut ary = vec![0; cap];
    let mask = cap - 1;
    for &x in self.ary.iter() {
      if x == 0 { continue; }
      let mut idx = x as usize & mask;
      loop {
        let key = ary[idx];
        if key == 0 { break; }
        idx = (idx + 1) & mask;
      }
      ary[idx] = x;
    }
    self.ary = ary;
  }
}
