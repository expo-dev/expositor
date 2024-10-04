trait ArrayMap<U, T, const N : usize> {
  fn mapary<F>(&self, f : F) -> [T; N] where F : Fn(&U) -> T;
}

impl<U, T, const N : usize> ArrayMap<U, T, N> for [U; N] {
  fn mapary<F>(&self, f : F) -> [T; N] where F : Fn(&U) -> T
  {
    unsafe {
      use std::mem::MaybeUninit;
      let mut ts : [MaybeUninit::<T>; N] = [const { MaybeUninit::uninit() }; N];
      for n in 0..N { ts[n].write(f(&self[n])); }
      return MaybeUninit::array_assume_init(ts);
    }
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub fn get_window_width() -> u16
{
  #[cfg(target_os="linux")]
  unsafe {
    use crate::util::STDERR;
    let ret : i64;
    let ioctl : u64 = 16;
    let tiocgwinsz : u64 = 0x5413;
    let mut winsize : [u16; 4] = [0; 4];
    std::arch::asm!(
      "syscall"
      , inout("rax") ioctl => ret
      ,    in("rdi") STDERR
      ,    in("rsi") tiocgwinsz
      ,    in("rdx") &mut winsize
      ,   out("r10") _
      ,   out("r8" ) _
      ,   out("r9" ) _
      ,   out("rcx") _
      ,   out("r11") _
    );
    if ret != 0 { return 0; }
    // The kernel populates winsize with
    //   [
    //     number of rows,
    //     number of columns,
    //     width in pixels,
    //     height in pixels
    //   ]
    //
    // Note that width and height in pixels may not be correct; for example,
    //   with my current terminal emulator settings and my display,
    //
    //     width  in pixels / number of columns =  7.0 pixels / column
    //     height in pixels / number of rows    = 16.0 pixels / row
    //
    //   whereas the true values are
    //
    //     14.0 pixels / column
    //     32.0 pixels / row.
    //
    // We just return number of columns.
    //
    return winsize[1];
  }
  #[cfg(not(target_os="linux"))]
  return 0;
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn linearize(x : f64) -> f64
{
  return if x > 0.040_45 { ((x + 0.055) / 1.055).powf(2.4) }
    else { x / 12.92 };
}

fn delinearize(x : f64) -> f64
{
  return if x > 0.0031308 { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
    else { 12.92 * x };
}

#[derive(Copy, Clone)]
pub struct RGB { components: [f64; 3] }

impl RGB {
  pub fn new(r : f64, g : f64, b : f64) -> Self
  {
    return Self { components: [r, g, b] };
  }

  pub fn from_bytes(ary : &[u8; 3]) -> Self
  {
    return Self { components: ary.mapary(|c| *c as f64 / 255.0) };
  }

  pub fn to_bytes(&self) -> [u8; 3]
  {
    return self.components.mapary(|c| unsafe {
      (c * 255.0 + 0.5).to_int_unchecked::<u8>()
    });
  }

  pub fn inside_gamut(&self) -> bool
  {
    return self.components.iter().all(|c| 0.0 <= *c && *c <= 1.0);
  }

  pub fn clip(&self) -> Self
  {
    return Self { components: self.components.mapary(|c|
      if *c > 1.0 { 1.0 } else if *c < 0.0 { 0.0 } else { *c }
    ) };
  }

  pub fn fg(&self) -> String
  {
    let cs = self.to_bytes();
    return format!("\x1B[38;2;{};{};{}m", cs[0], cs[1], cs[2]);
  }

  pub fn bg(&self) -> String
  {
    let cs = self.to_bytes();
    return format!("\x1B[48;2;{};{};{}m", cs[0], cs[1], cs[2]);
  }

  pub fn to_lch(&self) -> Lch
  {
    let r = linearize(self.components[0]);
    let g = linearize(self.components[1]);
    let b = linearize(self.components[2]);
    // "f" for "far" instead of "l" for "long"
    //   to avoid a name collision
    let f = (r * 0.4122214708 + g * 0.5363325363 + b * 0.0514459929).cbrt();
    let m = (r * 0.2119034982 + g * 0.6806995451 + b * 0.1073969566).cbrt();
    let s = (r * 0.0883024619 + g * 0.2817188376 + b * 0.6299787005).cbrt();
    let l = f * 0.2104542553 + m * 0.7936177850 - s * 0.0040720468;
    let a = f * 1.9779984951 - m * 2.4285922050 + s * 0.4505937099;
    let b = f * 0.0259040371 + m * 0.7827717662 - s * 0.8086757660;
    // let c = (a*a + b*b).sqrt();
    // let h = b.atan2(a) / std::f64::consts::TAU;
    // let h = if h < 0.0 { h + 1.0 } else { h };
    // when adjusting for the Helmholtz-Kohlrausch effect,
    //   "Lch.l" is the brightness instead of lightness
    // let l = l.powf(1.0 / (1.0 + c * 2.0));
    return Lch { l, a, b };
  }
}

#[derive(Copy, Clone)]
pub struct Lch { l : f64, a : f64, b : f64 }

impl Lch {
  pub fn new(l : f64, c : f64, h : f64) -> Self
  {
    // h is [0, 1) rather than [0, 360) or [0, τ)
    let a = c * (h * std::f64::consts::TAU).cos();
    let b = c * (h * std::f64::consts::TAU).sin();
    return Self { l, a, b };
  }

  pub fn to_rgb(&self) -> RGB
  {
    // when adjusting for the Helmholtz-Kohlrausch effect,
    //  "self.l" is the brightness instead of lightness
    // let c = (a*a + b*b).sqrt();
    // let l = self.l.powf(1.0 + c * 2.0);
    let f = self.l + self.a * 0.3963377774 + self.b * 0.2158037573;
    let m = self.l - self.a * 0.1055613458 - self.b * 0.0638541728;
    let s = self.l - self.a * 0.0894841775 - self.b * 1.2914855480;
    let f = f * f * f;
    let m = m * m * m;
    let s = s * s * s;
    return RGB::new(
      delinearize(f *  4.0767416621 - m * 3.3077115913 + s * 0.2309699292),
      delinearize(f * -1.2684380046 + m * 2.6097574011 - s * 0.3413193965),
      delinearize(f * -0.0041960863 - m * 0.7034186147 + s * 1.7076147010),
    );
  }

  pub fn distance(&self, other : &Self) -> f64
  {
    let dl = other.l - self.l;
    let da = other.a - self.a;
    let db = other.b - self.b;
    return (dl*dl + da*da + db*db).sqrt();
  }

  pub fn to_ansi(&self) -> ANSI
  {
    let mut best_d = 1.0;
    let mut best_c = 0;
    for n in 0..16 {
      let d = self.distance(&RGB::from_bytes(&PALETTE[n]).to_lch());
      if d < best_d {
        best_d = d;
        best_c = n as u8;
      }
    }
    for n in 0..24 {
      let c = 8 + (n as u8) * 10;
      let d = self.distance(&RGB::from_bytes(&[c, c, c]).to_lch());
      if d < best_d {
        best_d = d;
        best_c = 232 + n as u8;
      }
    }
    for r in 0..6 {
      for g in 0..6 {
        for b in 0..6 {
          let d = self.distance(&RGB::from_bytes(
              &[r, g, b].mapary(|c| ENCODING[*c])
          ).to_lch());
          if d < best_d {
            best_d = d;
            best_c = 16 + 36 * (r as u8) + 6 * (g as u8) + (b as u8);
          }
        }
      }
    }
    return ANSI { inner: best_c };
  }
}

static PALETTE : [[u8; 3]; 16] = [
  [  0,   0,   0],  //  0 Black
  [255,  84,  84],  //  1 Red
  [ 84, 255,  84],  //  2 Green
  [255, 255,  84],  //  3 Yellow
  [ 84,  84, 255],  //  4 Blue
  [255,  84, 255],  //  5 Magenta
  [ 84, 255, 255],  //  6 Cyan
  [178, 178, 178],  //  7 White
  [  0,   0,   0],  //  8 Intense Black
  [255,  84,  84],  //  9 Intense Red
  [ 84, 255,  84],  // 10 Intense Green
  [255, 255,  84],  // 11 Intense Yellow
  [ 84,  84, 255],  // 12 Intense Blue
  [255,  84, 255],  // 13 Intense Magenta
  [ 84, 255, 255],  // 14 Intense Cyan
  [255, 255, 255],  // 15 Intense White
];

// fn ENCODE(n : u8) -> u8
// {
//   assert!(n < 6);
//   if n == 0 { return 0; }
//   let x = n as f64 / 5.0;
//   let y = (x + 0.275) / 1.275;
//   return unsafe {
//     (y * 255.0 + 0.5).to_int_unchecked::<u8>()
//   };
// end

static ENCODING : [u8; 6] = [0, 95, 135, 175, 215, 255];

#[derive(Copy, Clone)]
pub struct ANSI { inner : u8 }

impl ANSI {
  pub fn to_rgb(&self) -> RGB
  {
    if self.inner < 16 {
      return RGB::from_bytes(&PALETTE[self.inner as usize]);
    }
    if self.inner < 232 {
      let mut n = self.inner - 16;
      let r = n / 36; n -= r * 36;
      let g = n /  6; n -= g *  6;
      let b = n;
      return RGB::from_bytes(&[r, g, b].mapary(|c| ENCODING[*c as usize]));
    }
    let c = 8 + (self.inner - 232) * 10;
    return RGB::from_bytes(&[c, c, c]);
  }

  pub fn fg(&self) -> String
  {
    return format!("\x1B[38;5;{}m", self.inner);
  }

  pub fn bg(&self) -> String
  {
    return format!("\x1B[48;5;{}m", self.inner);
  }
}
