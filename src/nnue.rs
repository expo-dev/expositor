use crate::color::*;
use crate::misc::*;
use crate::rand::*;
use crate::state::*;
use crate::simd::*;

use std::fs::File;
use std::io::{Read, Write, BufWriter, Error, ErrorKind};
use std::mem::MaybeUninit;
use std::simd::Simd;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// The Expositor NNUE has 768 inputs, two hidden layers, and a single output neuron. For the
//   network to be efficiently updatable, the order of the inputs and first layer neurons (and
//   the weights between them) is fixed: inputs #0 to #383 are for white and #384 to #767 are
//   for black. However, for the sake of symmetry (and to avoid having the network learn some
//   concepts twice), if black is the side to move we swap the upper and lower banks of the
//   first layer activations before computing the second layer. In this way we ensure the
//   lower bank, #0 to #(B1-1), is for the side to move and the upper bank, #B1 to #(N1-1),
//   is for the side waiting.
//
// For this to work, we do two things. First, we arrange the inputs for black so that black's
//   position is flipped vertically, e.g. input #4 is hot when a white king is on e1 and the
//   corresponding input, #(Bp+4), is hot when a black king is on e8. Second, we mirror the
//   weights to the banks of the first layer, so that e.g. the #4 → #0 weight is the same as
//   the #(Bp+4) → #B1 weight, or e.g. the #(Bp+4) → #0 weight is the same as the #4 → #B1
//   weight.
//
//           neuron-to
//              ─┴─
//     weight[x][n] = weight[x±Bp][n±B1]    with signs chosen so that n±B1 and x±Bp
//           ─┬─                              are not out of bounds
//       input_from
//
//  (Note that input to first layer weights are indexed in a different order than any other
//   two-dimensional array: w2[n][x], dE_dw1[n][x], dE_dw2[n][x], m_gw1[n][x], m_gw2[n][x],
//   v_gw1[n][x], and v_gw2[n][x], whereas the input to first layer weights are w1[x][n].
//   This is unfortunate but necessary for efficient parallelization.)
//
//   Since the weights are mirrored, we only bother storing half of them. Here is the same
//   equality as above, written out explicitly, with the canonical form on the righthand of
//   each equality:
//
//                            x = 0..Bp                    x = Bp..Np
//                  -----------------------------------------------------------
//                 |
//     n =  0..B1  |          w1[x][n]                      w1[x][n]
//                 |
//     n = B1..N1  |  w1[x][n] = w1[x+Bp][n-B1]     w1[x][n] = w1[x-Bp][x-B1]
//                 |
//
//   A last word about notation and terminology: we always use `s` to denote the weighted sum of
//   inputs to a neuron, e.g. s1[n]. The letter `a` is used for activation, by which we always
//   mean the output of the activation function, and so we have that a1[n] := relu(s1[n]) and
//   a2[n] := relu(s2[n]) for all n.

pub type Simd32 = Simd<f32, LANES>;

const SIMD_ZERO : Simd32 = Simd::from_array([0.0; LANES]);

// Number of input features and number of neurons per layer
//   Np and N1 must be a multiple of 2×LANES, and N2 must be a multiple of LANES
pub const Np : usize = 768; // Do not modify
pub const N1 : usize = 512; // Okay to vary
pub const N2 : usize = 8;   // Okay to vary
pub const N3 : usize = 1;   // Do not modify

// Number of input features per side and number of neurons per bank
pub const Bp : usize = Np / 2;
pub const B1 : usize = N1 / 2;

// Number of input vectors and number of vectors per layer
pub const Vp : usize = Np / LANES;
pub const V1 : usize = N1 / LANES;
pub const V2 : usize = N2 / LANES;

// Number of input vectors per side and number of vector per bank
pub const Hp : usize = Vp / 2;
pub const H1 : usize = V1 / 2;

const TOTAL : usize = Np*B1 + N1*N2 + N2*N3  // Weights
                    +    B1 +    N2 +    N3; // Biases

// We set the alignment of Network structs to 32 bytes so that SIMD loads and stores will be
//   aligned. The Rust reference states that the size of the struct will be a multiple of the
//   alignment, so the size of the struct in terms of single-precision floating point numbers
//   (which are 4 bytes long) is a multiple of 8.
//
const SIZE : usize = ((TOTAL + 7) / 8) * 8;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[derive(Clone, PartialEq)]
#[repr(align(32))]
pub struct Network {
  pub w1 : [[f32; B1]; Np], //
  pub w2 : [[f32; N1]; N2], // weights
  pub w3 :  [f32; N2],      //
  pub b1 :  [f32; B1], //
  pub b2 :  [f32; N2], // biases
  pub b3 :   f32,      //
}

pub static mut SEARCH_NETWORK : Network = Network::zero();

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

impl Network {
  pub const fn zero() -> Self
  {
    return Self {
      w1: [[0.0; B1]; Np],
      w2: [[0.0; N1]; N2],
      w3:  [0.0; N2],
      b1:  [0.0; B1],
      b2:  [0.0; N2],
      b3:   0.0,
    };
  }

  pub fn perturb(&mut self, scale : f32)
  {
    let f1 = ((Np + N2) as f32).sqrt();
    let f2 = (   N1     as f32).sqrt();
    let f3 = (   N2     as f32).sqrt();
    for x in 0..Np { for n in 0..B1 { self.w1[x][n] += triangular() as f32 * scale / f1; } }
    for n in 0..N2 { for x in 0..N1 { self.w2[n][x] += triangular() as f32 * scale / f2; } }
    for x in 0..N2 { self.w3[x] += triangular() as f32 * scale / f3; }
  }

  pub fn new() -> Self
  {
    let mut network = Self::zero();
    network.perturb(1.0);
    return network;
  }

  fn checksum(&self) -> u64
  {
    // Fletcher's checksum
    let mut lo : u32 = 0;
    let mut hi : u32 = 0;
    let array = unsafe { std::mem::transmute::<_,&[u32; SIZE]>(self) };
    for x in &array[0..SIZE] {
      let (sum, overflow) = lo.overflowing_add(*x);
      lo = if overflow { sum + 1 } else { sum };
      let (sum, overflow) = hi.overflowing_add(lo);
      hi = if overflow { sum + 1 } else { sum };
    }
    return ((hi as u64) << 32) | (lo as u64);
  }

  pub fn load(file : &str) -> std::io::Result<Self>
  {
    let mut fh = File::open(file)?;
    let mut sgntr = [0; 4];
    let mut array = [0; SIZE*4];
    let mut check = [0; 8];
    fh.read_exact(&mut sgntr)?;
    if sgntr != "EXPO".as_bytes() {
      return Err(Error::new(ErrorKind::Other, "missing signature"));
    }
    fh.read_exact(&mut array)?;
    let network = unsafe { std::mem::transmute::<_,Self>(array) };
    fh.read_exact(&mut check)?;
    if network.checksum() != u64::from_le_bytes(check) {
      return Err(Error::new(ErrorKind::Other, "checksum mismatch"));
    }
    return Ok(network);
  }

  pub fn save(&self, file : &str) -> std::io::Result<()>
  {
    let mut w = BufWriter::new(File::create(file)?);
    let bytes = unsafe { std::mem::transmute::<_,&[u8; SIZE*4]>(self) };
    w.write_all("EXPO".as_bytes())?;
    w.write_all(bytes)?;
    w.write_all(&self.checksum().to_le_bytes())?;
    return Ok(());
  }

  pub fn evaluate(&self, state : &State) -> f32
  {
    let mut s1 : [MaybeUninit<Simd32>; V1] = MaybeUninit::uninit_array();
    for n in 0..H1 { s1[  n ].write(simd_load!(self.b1, n)); }
    for n in 0..H1 { s1[H1+n].write(simd_load!(self.b1, n)); }
    let s1 = unsafe { std::mem::transmute::<_,&mut [Simd32; V1]>(&mut s1) };
    for black in [false, true] {
      for piece in 0..6 {
        let mut sources = state.boards[(black as usize)*8 + piece];
        while sources != 0 {
          let src = sources.trailing_zeros() as usize;
          let ofs = piece*64 + if black { vmirror(src) } else { src };
          let upper = state.turn.as_bool() ^ black;
          let x        = if upper { Bp } else {  0 } + ofs;
          let x_mirror = if upper {  0 } else { Bp } + ofs;
          for n in 0..H1 { s1[  n ] += simd_load!(self.w1[x],        n); }
          for n in 0..H1 { s1[H1+n] += simd_load!(self.w1[x_mirror], n); }
          sources &= sources - 1;
        }
      }
    }
    for n in 0..V1 { s1[n] = relu_ps(s1[n]); }
    let a1 = s1;

    let mut s2 : [MaybeUninit<f32>; N2] = MaybeUninit::uninit_array();
    for n in 0..N2 {
      let mut s = SIMD_ZERO;
      for x in 0..V1 { s += a1[x] * simd_load!(self.w2[n], x); }
      s2[n].write(self.b2[n] + horizontal_sum(s));
    }
    let s2 = unsafe { std::mem::transmute::<_,&mut [f32; N2]>(&mut s2) };

    let mut s = SIMD_ZERO;
    for x in 0..V2 { s += relu_ps(simd_load!(s2, x)) * simd_load!(self.w3, x); }
    let s3 = self.b3 + horizontal_sum(s);
    return s3;
  }
}

impl State {
  pub fn initialize_nnue(&mut self)
  {
    self.s1.clear();

    // NOTE this is somewhat unsafe, but the only way
    //   I've found to prevent an unnecessary copy.
    self.s1.reserve(1);
    unsafe { self.s1.set_len(1); }
    let s1 = &mut self.s1[0];

    unsafe {
      for n in 0..H1 { s1[  n ] = simd_load!(SEARCH_NETWORK.b1, n); }
      for n in 0..H1 { s1[H1+n] = simd_load!(SEARCH_NETWORK.b1, n); }
    }
    for black in [false, true] {
      for piece in 0..6 {
        let mut sources = self.boards[(black as usize)*8 + piece];
        while sources != 0 {
          let src = sources.trailing_zeros() as usize;
          let ofs = piece*64 + if black { vmirror(src) } else { src };
          let x        = if black { Bp } else {  0 } + ofs;
          let x_mirror = if black {  0 } else { Bp } + ofs;
          unsafe {
            for n in 0..H1 { s1[  n ] += simd_load!(SEARCH_NETWORK.w1[x],        n); }
            for n in 0..H1 { s1[H1+n] += simd_load!(SEARCH_NETWORK.w1[x_mirror], n); }
          }
          sources &= sources - 1;
        }
      }
    }
  }

  pub fn evaluate(&self) -> f32
  {
    unsafe {
      let s1 = &self.s1[self.s1.len()-1];

      let mut a1 : [MaybeUninit<Simd32>; V1] = MaybeUninit::uninit_array();
      for n in 0..V1 { a1[n].write(relu_ps(s1[n])); }
      let a1 = std::mem::transmute::<_,&mut [Simd32; V1]>(&mut a1);

      let mut s2 : [MaybeUninit<f32>; N2] = MaybeUninit::uninit_array();
      match self.turn {
        Color::White => {
          for n in 0..N2 {
            let mut s = SIMD_ZERO;
            for x in 0..V1 { s += a1[x] * simd_load!(SEARCH_NETWORK.w2[n], x); }
            s2[n].write(SEARCH_NETWORK.b2[n] + horizontal_sum(s));
          }
        }
        Color::Black => {
          for n in 0..N2 {
            let mut s = SIMD_ZERO;
            for x in 0..H1 { s += a1[H1+x] * simd_load!(SEARCH_NETWORK.w2[n],   x ); }
            for x in 0..H1 { s += a1[  x ] * simd_load!(SEARCH_NETWORK.w2[n], H1+x); }
            s2[n].write(SEARCH_NETWORK.b2[n] + horizontal_sum(s));
          }
        }
      }
      let s2 = std::mem::transmute::<_,&mut [f32; N2]>(&mut s2);

      let mut s = SIMD_ZERO;
      for x in 0..V2 { s += relu_ps(simd_load!(s2, x)) * simd_load!(SEARCH_NETWORK.w3, x); }
      let s3 = SEARCH_NETWORK.b3 + horizontal_sum(s);
      return s3;
    }
  }
}

impl Network {
  pub fn save_fst_image(&self, file : &str) -> std::io::Result<()>
  {
    let aspect  = 4;
    let upscale = 8;
    let width   = 12*8*upscale*aspect + 12*aspect - 1;
    let height  = B1*8*upscale/aspect + B1/aspect - 1;

    let border  = [0, 0, 0];

    let mut w = BufWriter::new(File::create(format!("{}.ppm", file))?);
    writeln!(&mut w, "P6")?;
    writeln!(&mut w, "{} {}", width, height)?;
    writeln!(&mut w, "255")?;

    let mut scale = [0.0; B1];
    for n in 0..B1 {
      let mut upper = 0.0;
      let mut lower = 0.0;
      for x in 0..Np {
        let z = self.w1[x][n];
        if z > upper { upper = z; }
        if z < lower { lower = z; }
      }
      scale[n] = upper.max(-lower);
    }

    for tile_row in 0..(B1/aspect) {
      if tile_row != 0 { for _ in 0..width { w.write(&border)?; } }
      for rank in (0..8).rev() {
        for _ in 0..upscale {
          for tile_column in 0..aspect {
            let n = tile_row * aspect + tile_column;
            for side in 0..2 {
              for piece in 0..6 {
                if !(tile_column == 0 && side == 0 && piece == 0) { w.write(&border)?; }
                for file in 0..8 {
                  let square = rank*8 + file;
                  let square = if side != 0 { vmirror(square) } else { square };
                  let x : usize = side*384 + piece*64 + square;
                  let normed = ((self.w1[x][n] / scale[n]) + 1.0) / 2.0;
                  debug_assert!(1.0 >= normed && normed >= 0.0, "out of range");
                  let bias = if self.b1[n] > 0.0 {  self.b1[n]/scale[n] * 16.0 } else { 0.0 };
                  let r = (bias + normed * (255.0 - bias)).round() as u8;
                  let g = (       normed *  255.0        ).round() as u8;
                  let b = (32.0 + normed *  191.0        ).round() as u8;
                  for _ in 0..upscale { w.write(&[r, g, b])?; }
                }
              }
            }
          }
        }
      }
    }
    w.flush()?;
    let status = std::process::Command::new("convert")
      .arg(&format!("{}.ppm", file)).arg(&format!("{}.png", file)).status()?;
    if status.success() { std::fs::remove_file(&format!("{}.ppm", file))?; }
    return Ok(());
  }

  pub fn save_snd_image(&self, file : &str) -> std::io::Result<()>
  {
    let upscale =  8;
    let width   =  6*8*upscale +  6 - 1;
    let height  = N2*8*upscale + N2 - 1;

    let border = [0, 0, 0];

    let mut w = BufWriter::new(File::create(format!("{}.ppm", file))?);
    writeln!(&mut w, "P6")?;
    writeln!(&mut w, "{} {}", width, height)?;
    writeln!(&mut w, "255")?;

    for n in 0..N2 {
      if n != 0 { for _ in 0..width { w.write(&border)?; } }

      let mut white = [0.0; Bp];  // really, side to move
      let mut black = [0.0; Bp];  //   and side waiting

      let mut upper : f32 = 0.0;
      let mut lower : f32 = 0.0;
      for piece in 0..6 {
        for square in 0..64 {
          let x = piece*64 + square;
          let y = piece*64 + vmirror(square);
          for m in 0..B1 {
            white[x] += self.w1[x][m] * self.w2[n][m];
            black[x] += self.w1[y][m] * self.w2[n][B1+m];
          }
          upper = upper.max(white[x]).max(black[x]);
          lower = lower.min(white[x]).min(black[x]);
        }
      }
      let scale = upper.max(-lower);

      for rank in (0..8).rev() {
        for _ in 0..upscale {
          for piece in 0..6 {
            if piece != 0 { w.write(&border)?; }
            for file in 0..8 {
              let x : usize = piece*64 + rank*8 + file;

              let a = -black[x] / scale;
              let b = -white[x] / scale;

              debug_assert!(1.0 >= a && a >= -1.0, "out of range");
              debug_assert!(1.0 >= b && b >= -1.0, "out of range");

              let r = (1.0 - b) / 2.0;
              let r = r * r;
              let g = ((1.0 + a) / 2.0).powf(1.25) * ((1.0 - b) / 2.0).powf(1.25);
              let b = (1.0 + a) / 2.0;
              let b = b * b;

              let r = (r * 255.0).round() as u8;
              let g = (g * 255.0).round() as u8;
              let b = (b * 255.0).round() as u8;

              for _ in 0..upscale { w.write(&[r, g, b])?; }
            }
          }
        }
      }
    }
    w.flush()?;
    let status = std::process::Command::new("convert")
      .arg(&format!("{}.ppm", file)).arg(&format!("{}.png", file)).status()?;
    if status.success() { std::fs::remove_file(&format!("{}.ppm", file))?; }
    return Ok(());
  }

  pub fn save_source(&self, file : &str) -> std::io::Result<()>
  {
    let mut w = BufWriter::new(File::create(file)?);
    writeln!(&mut w, "use crate::nnue::*;")?;
    writeln!(&mut w, "")?;
    writeln!(&mut w, "pub const DEFAULT_NETWORK : Network = Network {{")?;

    // Input → Layer 1 Weights
    writeln!(&mut w, "  w1: [")?;
    for x in 0..Np {
      write!(&mut w, "    [")?;
      for n in 0..B1 {
        if n > 0 && n % 4 == 0 { write!(&mut w, "\n     ")?; }
        let a = self.w1[x][n];
        if 10.0 > a.abs() && a.abs() >= 0.01 {
          write!(&mut w, " {:16.13},", a)?;
        }
        else {
          write!(&mut w, " {:16.10e},", a)?;
        }
      }
      writeln!(&mut w, " ],")?;
    }
    writeln!(&mut w, "  ],")?;

    // Layer 1 → Layer 2 Weights
    writeln!(&mut w, "  w2: [")?;
    for n in 0..N2 {
      write!(&mut w, "    [")?;
      for x in 0..N1 {
        if x > 0 && x % 4 == 0 { write!(&mut w, "\n     ")?; }
        let a = self.w2[n][x];
        if 10.0 > a.abs() && a.abs() >= 0.01 {
          write!(&mut w, " {:16.13},", a)?;
        }
        else {
          write!(&mut w, " {:16.10e},", a)?;
        }
      }
      writeln!(&mut w, " ],")?;
    }
    writeln!(&mut w, "  ],")?;

    // Layer 2 → Layer 3 Weights
    writeln!(&mut w, "  w3: [")?;
    write!(&mut w, "     ")?;
    for x in 0..N2 {
      if x > 0 && x % 4 == 0 { write!(&mut w, "\n     ")?; }
      let a = self.w3[x];
      if 10.0 > a.abs() && a.abs() >= 0.01 {
        write!(&mut w, " {:16.13},", a)?;
      }
      else {
        write!(&mut w, " {:16.10e},", a)?;
      }
    }
    write!(&mut w, "\n")?;
    writeln!(&mut w, "  ],")?;

    // Layer 1 Biases
    writeln!(&mut w, "  b1: [")?;
    write!(&mut w, "     ")?;
    for n in 0..B1 {
      if n > 0 && n % 4 == 0 { write!(&mut w, "\n     ")?; }
      let a = self.b1[n];
      if 10.0 > a.abs() && a.abs() >= 0.01 {
        write!(&mut w, " {:16.13},", a)?;
      }
      else {
        write!(&mut w, " {:16.10e},", a)?;
      }
    }
    write!(&mut w, "\n")?;
    writeln!(&mut w, "  ],")?;

    // Layer 2 Biases
    writeln!(&mut w, "  b2: [")?;
    write!(&mut w, "     ")?;
    for n in 0..N2 {
      if n > 0 && n % 4 == 0 { write!(&mut w, "\n     ")?; }
      let a = self.b2[n];
      if 10.0 > a.abs() && a.abs() >= 0.01 {
        write!(&mut w, " {:16.13},", a)?;
      }
      else {
        write!(&mut w, " {:16.10e},", a)?;
      }
    }
    write!(&mut w, "\n")?;
    writeln!(&mut w, "  ],")?;

    // Layer 3 Bias
    writeln!(&mut w, "  b3:")?;
    write!(&mut w, "     ")?;
    let a = self.b3;
    if 10.0 > a.abs() && a.abs() >= 0.01 {
      writeln!(&mut w, " {:16.13},", a)?;
    }
    else {
      writeln!(&mut w, " {:16.10e},", a)?;
    }

    write!(&mut w, "}};")?;
    return Ok(());
  }
}
