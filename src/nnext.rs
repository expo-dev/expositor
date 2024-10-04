#![allow(non_upper_case_globals)]

use std::simd::Simd;
use std::simd::f32x8;
use std::simd::num::SimdFloat;
use std::simd::cmp::SimdPartialOrd;

use crate::state::{MiniState, State};

pub static mut NNETWORK  : NNetwork = NNetwork::zero();
pub static mut FNETWORK  : FNetwork = FNetwork::zero();
pub static mut QUANTIZED : QNetwork = QNetwork::zero();

pub const SideToMove  : usize = 0;
pub const SideWaiting : usize = 1;

pub const SameSide : usize = 0;
pub const OppoSide : usize = 1;

pub const REGIONS : usize = 5;
pub const PHASES  : usize = 4;

pub const Np  : usize = 384;      // 6 kinds × 64 squares
pub const N1  : usize = 800;      // multiple of 16 lanes × 2 lanes / pair

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[derive(Clone)]
#[repr(align(64))]
pub struct QNetwork1 {
  pub w1 : [[[i16; N1]; Np]; 2],  // weight[from-side][from-input][to-first]
}

#[derive(Clone)]
#[repr(align(64))]
pub struct QNetwork2 {
  pub w2 : [i16; N1],   // weight[from-side][from-first]
}

#[derive(Clone)]
#[repr(align(4096))]
pub struct QNetwork {
  pub l1 : [QNetwork1; REGIONS],
  pub l2 : [QNetwork2; PHASES],
  pub b2 : [f32; PHASES],
}

fn aplary<U, T, const N : usize, F>(
  dst : &mut [T; N],
  src : &[U; N],
  f : F
) where F : Fn(&mut T, &U)
{
  for n in 0..N { f(&mut dst[n], &src[n]); }
}

fn f32_to_i6p10(x : f32) -> i16
{
  // max +31.999023
  // min -32.000000
  // eps  +0.000977
  return (x * (1 << 10) as f32).round() as i16;
}

fn f32_to_i4p12(x : f32) -> i16
{
  // max +7.999756
  // min -8.000000
  // eps +0.000244
  return (x * (1 << 12) as f32).round() as i16;
}

fn umul_hi(x : u16, y : u16) -> u16
{
  return ((x as u32).wrapping_mul(y as u32) >> 16) as u16;
}

fn imul_hi(x : i16, y : i16) -> i16
{
  return ((x as i32).wrapping_mul(y as i32) >> 16) as i16;
}

impl QNetwork {
  pub const fn zero() -> Self
  {
    return unsafe { std::mem::zeroed() };
  }

  pub fn emulate(&mut self, network : &FNetwork)
  {
    aplary(&mut self.l1, &network.l1, |dl1, sl1|
      aplary(&mut dl1.w1, &sl1.w1, |dl1_c, sl1_c|
        aplary(dl1_c, sl1_c, |dl1_x, sl1_x|
          aplary(dl1_x, sl1_x, |dl1_n, sl1_n|
            *dl1_n = f32_to_i6p10(*sl1_n)
          )
        )
      )
    );
    aplary(&mut self.l2, &network.l2, |dl2, sl2|
      aplary(&mut dl2.w2, &sl2.w2, |dl2_n, sl2_n|
        *dl2_n = f32_to_i4p12(*sl2_n)
      )
    );
    aplary(&mut self.b2, &network.l2, |b2, sl2| *b2 = sl2.b2);
  }
}

impl State {
  pub fn initialize_nnue(&mut self)
  {
    use crate::color::Color::*;
    use crate::misc::vmirror;
    use crate::piece::{KQRBNP, Piece::*};

    let wk_sq =         self.boards[WhiteKing].trailing_zeros() as usize ;
    let bk_sq = vmirror(self.boards[BlackKing].trailing_zeros() as usize);
    let w_rn = king_region(wk_sq);
    let b_rn = king_region(bk_sq);

    self.s1.clear();
    self.s1.push([[0; N1]; 2]);
    let s1 = &mut self.s1[0];

    let w_ps = unsafe { &QUANTIZED.l1[w_rn] };
    let b_ps = unsafe { &QUANTIZED.l1[b_rn] };

    let w_ptr = &raw mut s1[White] as *mut i16;
    let b_ptr = &raw mut s1[Black] as *mut i16;

    for kind in KQRBNP {
      let mut sources = self.boards[White+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let ofs = (kind as usize) * 64 + src;
        incr::<i16, N1>(w_ptr, &raw const (*w_ps).w1[SameSide][ofs] as *const i16);
        incr::<i16, N1>(b_ptr, &raw const (*b_ps).w1[OppoSide][ofs] as *const i16);
        sources &= sources - 1;
      }
    }

    for kind in KQRBNP {
      let mut sources = self.boards[Black+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let ofs = (kind as usize) * 64 + vmirror(src);
        incr::<i16, N1>(w_ptr, &raw const (*w_ps).w1[OppoSide][ofs] as *const i16);
        incr::<i16, N1>(b_ptr, &raw const (*b_ps).w1[SameSide][ofs] as *const i16);
        sources &= sources - 1;
      }
    }
  }

  pub fn evaluate(&self) -> f32
  {
    let men = (self.sides[0] | self.sides[1]).count_ones() as usize;
    let men = std::cmp::min(men - 2, 30);
    let phase = men / 8;
    let l2_ps = unsafe { &QUANTIZED.l2[phase] };
    let b2 = unsafe { &QUANTIZED.b2[phase] };

    let s1 = &self.s1[self.s1.len()-1];

    let mut l : [i32; 16] = [0; 16];
    // let mut l : [f32; 16] = [0.0; 16];
    for side in 0..2 {
      let switched = (N1/2) * (side ^ self.turn as usize);
      for reg in 0..(N1/2)/16 {
        for ofs in 0..16 {
          let x = reg*16 + ofs;
          let n1 = (std::cmp::min(std::cmp::max(s1[side][       x], 0), 1024) as u16) << 5;
          let n2 = (std::cmp::min(std::cmp::max(s1[side][N1/2 + x], 0), 1024) as u16) << 5;
          let pn = ((n1 as u32).wrapping_mul(n2 as u32) >> 16) as i32;
          let aw = pn.wrapping_mul(l2_ps.w2[switched + x] as i32);
          l[ofs] = l[ofs].wrapping_add(aw); // or saturating_add?
          // let n1_ = crelu(s1[side][       x] as f32 / 1024.0);
          // let n2_ = crelu(s1[side][N1/2 + x] as f32 / 1024.0);
          // let pn_ = n1_ * n2_;
          // let aw_ = pn_ * (l2_ps.w2[switched + x] as f32 / 4096.0);
          // if pn_ != 0.0 {
          //   eprintln!(
          //     "{:9.3} {:9.3} {:5} {:7.1}",
          //     pn as f32 / pn_, aw as f32 / aw_, aw, aw_ * 67108864.0
          //   );
          // }
        }
      }
    }
    let pl = [
      l[0] as f32 + l[ 8] as f32,
      l[1] as f32 + l[ 9] as f32,
      l[2] as f32 + l[10] as f32,
      l[3] as f32 + l[11] as f32,
      l[4] as f32 + l[12] as f32,
      l[5] as f32 + l[13] as f32,
      l[6] as f32 + l[14] as f32,
      l[7] as f32 + l[15] as f32,
    ];
    return b2 + (pl.into_iter().sum::<f32>() / 67108864.0);
    // let pl = [
    //   l[0] + l[ 8],
    //   l[1] + l[ 9],
    //   l[2] + l[10],
    //   l[3] + l[11],
    //   l[4] + l[12],
    //   l[5] + l[13],
    //   l[6] + l[14],
    //   l[7] + l[15],
    // ];
    // return b2 + pl.into_iter().sum::<f32>();
  }
}


// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[derive(Clone)]
#[repr(align(32))]
pub struct NNetwork1 {
  pub w1 : [[[f32; N1]; Np]; 2],  // weight[from-side][from-input][to-first]
}

#[derive(Clone)]
#[repr(align(32))]
pub struct NNetwork2 {
  pub w2 : [f32; N1],             // weight[from-side][from-first]
  pub b2 :  f32
}

#[derive(Clone)]
#[repr(align(4096))]
pub struct NNetwork {
  pub l1 : [NNetwork1; REGIONS+1],
  pub l2 : [NNetwork2; PHASES],
}

#[derive(Clone)]
#[repr(align(4096))]
pub struct FNetwork {
  pub l1 : [NNetwork1; REGIONS],
  pub l2 : [NNetwork2; PHASES],
}

fn crelu(x : f32) -> f32
{
  return x.max(0.0).min(1.0);
}

const ARTIFICIAL_GRADIENT : f32 = 1.0 / 4096.0;

fn d_crelu(x : f32) -> f32
{
  if x < 0.0 { return const { ARTIFICIAL_GRADIENT }; }
  if x > 1.0 { return const { ARTIFICIAL_GRADIENT }; }
  return 1.0;
}

fn crelu_v(x : f32x8) -> f32x8
{
  return x.simd_max(Simd::splat(0.0)).simd_min(Simd::splat(1.0));
}

fn d_relu_v(x : f32x8) -> f32x8
{
  let zero : f32x8 = Simd::splat(0.0);
  let one  : f32x8 = Simd::splat(1.0);
  let rhs = x.simd_gt(one).select(zero, one);
  return x.simd_lt(zero).select(zero, rhs);
}

pub fn expectation(x : f32) -> f32
{
  // score in pawns to expected outcome (from 0 to 1)
  // expectation(1.0) = 0.750
  // expectation(2.0) = 0.875
  let near = 0.5 / (1.0 + (x * -1.697_915_8).exp());
  let far  = 0.5 / (1.0 + (x * -0.639_930_1).exp());
  return near + far;
}

fn d_expectation(x : f32) -> f32
{
  let x = x.abs();
  let e1 = (-1.697_915_8 * x).exp();
  let e2 = (-0.639_930_1 * x).exp();
  let p1 = 1.0 + e1;
  let p2 = 1.0 + e2;
  return 0.848_958 * e1 / (p1 * p1) + 0.319_965 * e2 / (p2 * p2);
}

pub const fn king_region(idx : usize) -> usize
{
  if idx >= 16 { return 4; }
  const MAP : u32 = 0x_feaa_dcaa;
  return ((MAP >> (idx*2)) & 3) as usize;
}

const fn num_subjects(mini : &MiniState) -> usize
{
  let w = unsafe { std::mem::transmute::<_, &[u64; 2]>(&mini.positions[0]) };
  let b = unsafe { std::mem::transmute::<_, &[u64; 2]>(&mini.positions[1]) };
  let absent = (w[0] & 0x_80_80_80_80_80_80_80_80).count_ones()
             + (w[1] & 0x_80_80_80_80_80_80_80_80).count_ones()
             + (b[0] & 0x_80_80_80_80_80_80_80_80).count_ones()
             + (b[1] & 0x_80_80_80_80_80_80_80_80).count_ones();
  let mut men = 30 - (absent as usize);
  if mini.variable[0].1 >= 0 { men += 1; }
  if mini.variable[1].1 >= 0 { men += 1; }
  return men;
}

const fn phase_index(mini : &MiniState) -> usize
{
  return num_subjects(mini) / 8;
}

pub fn incr<T, const N : usize>(acc : *mut T, add : *const T)
where T : Copy + std::ops::Add<Output=T>
{
  unsafe {
    for x in 0..N {
      let sum = *acc.add(x) + *add.add(x);
      std::ptr::write(acc.add(x), sum);
    }
  }
}

fn reset(network : *mut NNetwork)
{
  unsafe {
    let ary = network as *mut f32;
    for x in 0..NNetwork::NUM_PARAMS {
      std::ptr::write(ary.add(x), 0.0);
    }
  }
}

impl NNetwork {
  const NUM_PARAMS : usize = std::mem::size_of::<Self>() / std::mem::size_of::<f32>();

  pub const fn zero() -> Self
  {
    return unsafe { std::mem::zeroed() };
  }

  pub fn reset(&mut self)
  {
    reset(self as *mut Self);
  }

  pub fn perturb(&mut self)
  {
    use crate::rand::RandDist;

    let active = 32 as f32;
    let scale = active.sqrt().recip();

    // for region in 0..(REGIONS+1) {
      for side in [SameSide, OppoSide] {
        for kind in 0..6 {
          for rank in 0..8 {
            if kind == 5 && (rank == 0 || rank == 7) { continue; }
            for file in 0..8 {
              let sq = rank * 8 + file;
              let np = kind * 64 + sq;
              // if side != SameSide || kind != 0 || king_region(sq) == region || region == REGIONS {
                for n1 in 0..N1 {
                  let w = f32::triangular() * scale;
                  // self.l1[region].w1[side][np][n1] = w;
                  self.l1[REGIONS].w1[side][np][n1] = w;
                }
              // }
            }
          }
        }
        // for np in 384..FNp {
        //   for n1 in 0..N1 {
        //     let w = f32::triangular() * scale;
        //     // self.l1[region].w1[side][np][n1] = w;
        //     self.l1[REGIONS].w1[side][np][n1] = w;
        //   }
        // }
      }
    // }

    let active = N1 as f32;
    let scale = active.sqrt().recip();

    for n1 in 0..N1 {
      let w = f32::uniform() * scale;
      for phase in 0..PHASES {
        self.l2[phase].w2[n1] = w;
      }
    }
  }

  pub fn scale_from(&mut self, multiplicand : *const Self, scalar : f32)
  {
    unsafe {
      let ary = self as *mut Self as *mut f32;
      let mpd = multiplicand as *const f32;
      for x in 0..Self::NUM_PARAMS {
        let prod = *mpd.add(x) * scalar;
        std::ptr::write(ary.add(x), prod);
      }
    }
  }

  pub fn scale(&mut self, scalar : f32)
  {
    unsafe {
      let ary = self as *mut Self as *mut f32;
      for x in 0..Self::NUM_PARAMS {
        let prod = *ary.add(x) * scalar;
        std::ptr::write(ary.add(x), prod);
      }
    }
  }

  pub fn incr(&mut self, addend : *const Self)
  {
    unsafe {
      let acc_ary = self as *mut Self as *mut f32;
      let add_ary = addend as *const f32;
      for x in 0..Self::NUM_PARAMS {
        let sum = *acc_ary.add(x) + *add_ary.add(x);
        std::ptr::write(acc_ary.add(x), sum);
      }
    }
  }

  pub fn incr_scaled(&mut self, addend : *const Self, scalar : f32)
  {
    unsafe {
      let acc_ary = self as *mut Self as *mut f32;
      let add_ary = addend as *const f32;
      for x in 0..Self::NUM_PARAMS {
        let sum = *acc_ary.add(x) + *add_ary.add(x) * scalar;
        std::ptr::write(acc_ary.add(x), sum);
      }
    }
  }

  pub fn incr_square_scaled(&mut self, addend : *const Self, scalar : f32)
  {
    unsafe {
      let acc_ary = self as *mut Self as *mut f32;
      let add_ary = addend as *const f32;
      for x in 0..Self::NUM_PARAMS {
        let val = *add_ary.add(x);
        let sum = *acc_ary.add(x) + (val * val) * scalar;
        std::ptr::write(acc_ary.add(x), sum);
      }
    }
  }

  pub fn update(&mut self, means : *const Self, vars : *const Self, scalar : f32)
  {
    unsafe {
      let params_ary = self as *mut Self as *mut f32;
      let  means_ary = means as *const f32;
      let   vars_ary = vars  as *const f32;
      for x in 0..Self::NUM_PARAMS {
        let p = *params_ary.add(x);
        let m =  *means_ary.add(x);
        let v =   *vars_ary.add(x);
        let u = p - scalar * m / (v.sqrt() + epsilon);
        std::ptr::write(params_ary.add(x), u);
      }
    }
  }

  pub fn regularize(&mut self, scalar : f32)
  {
    unsafe {
      let ary = self as *mut Self as *mut f32;
      for x in 0..Self::NUM_PARAMS {
        let val = *ary.add(x);
        let adj;
        if val < 0.0 {
          adj = if val < -scalar { val + scalar } else { 0.0 };
        }
        else {
          adj = if val > scalar { val - scalar } else { 0.0 };
        }
        std::ptr::write(ary.add(x), adj);
      }
    }
  }

  pub fn load(path : &str) -> std::io::Result<Self>
  {
    use std::fs::File;
    use std::io::Read;
    const SIZE : usize = std::mem::size_of::<NNetwork>();
    let mut fh = File::open(path)?;
    let mut ary = [0; SIZE];
    fh.read_exact(&mut ary)?;
    let network = unsafe { std::mem::transmute::<_, Self>(ary) };
    return Ok(network);
  }

  pub fn save(&self, path : &str) -> std::io::Result<()>
  {
    use std::fs::File;
    use std::io::{Write, BufWriter};
    const SIZE : usize = std::mem::size_of::<NNetwork>();
    let mut wb = BufWriter::new(File::create(format!("{path}.nnue"))?);
    let array = unsafe { std::mem::transmute::<_, &[u8; SIZE]>(self) };
    wb.write_all(array)?;
    return Ok(());
  }

  pub fn image(&self, path : &str) -> std::io::Result<()>
  {
    use std::fs::File;
    use std::io::{Write, BufWriter};
    use crate::misc::vmirror;

    // block width    6 kd/nn × 8 sq/kd = 48 sq
    // block height   2 sd/nn × 8 sq/sd × 6 rn = 96 sq

    let blocks_wide = 40;
    let blocks_tall = 20;
    assert!(blocks_wide * blocks_tall == N1);

    let upscale = 1;  // pixels per square

    let width  = blocks_wide * 6 * 8 * upscale               + blocks_wide - 1;
    let height = blocks_tall * 2 * 8 * (REGIONS+1) * upscale + blocks_tall - 1;

    let border = [0, 0, 0];

    // k is chosen so that 1 / (1 + exp(2, w × k)) is 3/4 when w is 1
    let k : f32 = -1.584_962_500_721_156_3;
    let s : f32 =  1.0;

    let mut out = BufWriter::new(File::create(format!("{path}.ppm"))?);
    writeln!(&mut out, "P6")?;
    writeln!(&mut out, "{width} {height}")?;
    writeln!(&mut out, "255")?;

    for block_row in 0..blocks_tall {
      if block_row > 0 { for _ in 0..width { out.write(&border)?; } }
      for region in 0..(REGIONS+1) {
        for side in (0..2).rev() {
          for rank in (0..8).rev() {
            for _ in 0..upscale {
              for block_column in 0..blocks_wide {
                if block_column > 0 { out.write(&border)?; }
                let n = block_row * blocks_wide + block_column;
                let n = if n % 2 == 0 { n / 2 } else { N1 / 2 + (n - 1) / 2 };
                for kind in 0..6 {
                  for file in 0..8 {
                    let x = rank * 8 + file;
                    let x = if side == 0 { x } else { vmirror(x) };
                    let x = kind * 64 + x;
                    let w = self.l1[region].w1[side][x][n];
                    let normed = (1.0 + (w * k * s).exp2()).recip();
                    assert!(1.0 >= normed && normed >= 0.0);
                    let i = (normed * 256.0).round() as usize;
                    for _ in 0..upscale { out.write(&PALETTE[i])?; }
                  }
                }
              }
            }
          }
        }
      }
    }
    out.flush()?;
    let status = std::process::Command::new("magick")
      .arg(&format!("{path}.ppm")).arg(&format!("{path}.png")).status()?;
    if status.success() { std::fs::remove_file(format!("{path}.ppm"))?; }
    return Ok(());
  }

  pub fn evaluate(&self, state : &State) -> f32
  {
    use crate::color::{WB, Color::*};
    use crate::misc::vmirror;
    use crate::piece::{KQRBNP, Piece::*};

    let wk_sq =         state.boards[WhiteKing].trailing_zeros() as usize ;
    let bk_sq = vmirror(state.boards[BlackKing].trailing_zeros() as usize);
    let w_rn = king_region(wk_sq);
    let b_rn = king_region(bk_sq);

    let men = (state.sides[0] | state.sides[1]).count_ones() as usize;
    let men = std::cmp::min(men - 2, 30);
    let phase = men / 8;

    let mut s1 = [[0.0; N1]; 2];

    let w_ps = &self.l1[w_rn];
    let b_ps = &self.l1[b_rn];
    let f_ps = &self.l1[REGIONS];

    for kind in KQRBNP {
      let mut sources = state.boards[White+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize) * 64 + src;
        for n in 0..N1 { s1[White][n] += w_ps.w1[SameSide][x][n]; }
        for n in 0..N1 { s1[Black][n] += b_ps.w1[OppoSide][x][n]; }
        for n in 0..N1 { s1[White][n] += f_ps.w1[SameSide][x][n]; }
        for n in 0..N1 { s1[Black][n] += f_ps.w1[OppoSide][x][n]; }
//      let x = 384 + kind as usize;
//      for n in 0..N1 { s1[White][n] += w_ps.w1[SameSide][x][n]; }
//      for n in 0..N1 { s1[Black][n] += b_ps.w1[OppoSide][x][n]; }
//      for n in 0..N1 { s1[White][n] += f_ps.w1[SameSide][x][n]; }
//      for n in 0..N1 { s1[Black][n] += f_ps.w1[OppoSide][x][n]; }
//      if kind != crate::piece::Kind::Bishop {
//        sources &= sources - 1;
//        continue;
//      }
//      let x = 390 + (src & 1);
//      for n in 0..N1 { s1[White][n] += w_ps.w1[SameSide][x][n]; }
//      for n in 0..N1 { s1[Black][n] += b_ps.w1[OppoSide][x][n]; }
//      for n in 0..N1 { s1[White][n] += f_ps.w1[SameSide][x][n]; }
//      for n in 0..N1 { s1[Black][n] += f_ps.w1[OppoSide][x][n]; }
        sources &= sources - 1;
      }
    }
    for kind in KQRBNP {
      let mut sources = state.boards[Black+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let x = (kind as usize) * 64 + vmirror(src);
        for n in 0..N1 { s1[White][n] += w_ps.w1[OppoSide][x][n]; }
        for n in 0..N1 { s1[Black][n] += b_ps.w1[SameSide][x][n]; }
        for n in 0..N1 { s1[White][n] += f_ps.w1[OppoSide][x][n]; }
        for n in 0..N1 { s1[Black][n] += f_ps.w1[SameSide][x][n]; }
//      let x = 384 + kind as usize;
//      for n in 0..N1 { s1[White][n] += w_ps.w1[OppoSide][x][n]; }
//      for n in 0..N1 { s1[Black][n] += b_ps.w1[SameSide][x][n]; }
//      for n in 0..N1 { s1[White][n] += f_ps.w1[OppoSide][x][n]; }
//      for n in 0..N1 { s1[Black][n] += f_ps.w1[SameSide][x][n]; }
//      if kind != crate::piece::Kind::Bishop {
//        sources &= sources - 1;
//        continue;
//      }
//      let x = 390 + (vmirror(src) & 1); // the vmirror is not necessary
//      for n in 0..N1 { s1[White][n] += w_ps.w1[OppoSide][x][n]; }
//      for n in 0..N1 { s1[Black][n] += b_ps.w1[SameSide][x][n]; }
//      for n in 0..N1 { s1[White][n] += f_ps.w1[OppoSide][x][n]; }
//      for n in 0..N1 { s1[Black][n] += f_ps.w1[SameSide][x][n]; }
        sources &= sources - 1;
      }
    }

    for c in WB { for n in 0..N1 { s1[c][n] = crelu(s1[c][n]); } }
    let a1 = s1;

    let mut m1 = [0.0_f32; N1];
    let t = state.turn as usize;
    for side in 0..2 {
      let h = (side ^ t) * N1 / 2;
      for p in 0..N1/2 {
        m1[h + p] = a1[side][p] * a1[side][N1/2 + p];
      }
    }

    let mut s2_reg = [0.0_f32; 8];
    for reg in 0..N1/8 {
      let ofs = reg * 8;
      for lane in 0..8 {
        s2_reg[lane] += m1[ofs+lane] * self.l2[phase].w2[ofs+lane];
      }
    }
    let s2 = self.l2[phase].b2 + s2_reg.iter().sum::<f32>();

    let mut buf = NBuffer::zero();
    if let Some(mini) = MiniState::from(&state) {
      let spec = fwd(&raw const *self, &mut buf, &mini);
      let err = s2 - spec;
      if err.abs() > 0.000_1 {
        eprintln!("mismatch: eval {s2} fwd {spec} err {err}");
      }
    }

    return s2;
  }

  pub fn distribute(&self) -> FNetwork
  {
    let mut network = FNetwork::zero();
    for region in 0..REGIONS {
      network.l1[region] = self.l1[region].clone();
      for side in 0..2 {
        for x in 0..Np {
          for n in 0..N1 {
            network.l1[region].w1[side][x][n] += self.l1[REGIONS].w1[side][x][n];
          }
        }
      }
    }
    network.l2 = self.l2.clone();
    return network;
  }
}

pub static mut S1_MAX : f32 = 0.0;
pub static mut S1_MIN : f32 = 0.0;

impl FNetwork {
  pub const fn zero() -> Self
  {
    return unsafe { std::mem::zeroed() };
  }

  pub fn load(path : &str) -> std::io::Result<Self>
  {
    use std::fs::File;
    use std::io::Read;
    const SIZE : usize = std::mem::size_of::<FNetwork>();
    let mut fh = File::open(path)?;
    let mut ary = [0; SIZE];
    fh.read_exact(&mut ary)?;
    let network = unsafe { std::mem::transmute::<_, Self>(ary) };
    return Ok(network);
  }

  pub fn save(&self, path : &str) -> std::io::Result<()>
  {
    use std::fs::File;
    use std::io::{Write, BufWriter};
    const SIZE : usize = std::mem::size_of::<FNetwork>();
    let mut wb = BufWriter::new(File::create(format!("{path}.nnue"))?);
    let array = unsafe { std::mem::transmute::<_, &[u8; SIZE]>(self) };
    wb.write_all(array)?;
    return Ok(());
  }

  pub fn image(&self, path : &str) -> std::io::Result<()>
  {
    use std::fs::File;
    use std::io::{Write, BufWriter};
    use crate::misc::vmirror;

    // block width    6 kd/nn × 8 sq/kd = 48 sq
    // block height   2 sd/nn × 8 sq/sd × 6 rn = 96 sq

    let blocks_wide = 40;
    let blocks_tall = 20;
    assert!(blocks_wide * blocks_tall == N1);

    let upscale = 1;  // pixels per square

    let width  = blocks_wide * 6 * 8 * upscale           + blocks_wide - 1;
    let height = blocks_tall * 2 * 8 * REGIONS * upscale + blocks_tall - 1;

    let border = [0, 0, 0];

    // k is chosen so that 1 / (1 + exp(2, w × k)) is 3/4 when w is 1
    let k : f32 = -1.584_962_500_721_156_3;
    let s : f32 =  1.0;

    let mut out = BufWriter::new(File::create(format!("{path}.ppm"))?);
    writeln!(&mut out, "P6")?;
    writeln!(&mut out, "{width} {height}")?;
    writeln!(&mut out, "255")?;

    for block_row in 0..blocks_tall {
      if block_row > 0 { for _ in 0..width { out.write(&border)?; } }
      for region in 0..REGIONS {
        for side in (0..2).rev() {
          for rank in (0..8).rev() {
            for _ in 0..upscale {
              for block_column in 0..blocks_wide {
                if block_column > 0 { out.write(&border)?; }
                let n = block_row * blocks_wide + block_column;
                let n = if n % 2 == 0 { n / 2 } else { N1 / 2 + (n - 1) / 2 };
                for kind in 0..6 {
                  for file in 0..8 {
                    let x = rank * 8 + file;
                    let x = if side == 0 { x } else { vmirror(x) };
                    let x = kind * 64 + x;
                    let w = self.l1[region].w1[side][x][n];
                    let normed = (1.0 + (w * k * s).exp2()).recip();
                    assert!(1.0 >= normed && normed >= 0.0);
                    let i = (normed * 256.0).round() as usize;
                    for _ in 0..upscale { out.write(&PALETTE[i])?; }
                  }
                }
              }
            }
          }
        }
      }
    }
    out.flush()?;
    let status = std::process::Command::new("magick")
      .arg(&format!("{path}.ppm")).arg(&format!("{path}.png")).status()?;
    if status.success() { std::fs::remove_file(format!("{path}.ppm"))?; }
    return Ok(());
  }

  pub fn stats(&self)
  {
    let mut w1_max = 0.0;
    let mut w1_min = 0.0;
    for rn in 0..REGIONS {
      for side in 0..2 {
        for x in 0..Np {
          for n in 0..N1 {
            let w = self.l1[rn].w1[side][x][n];
            w1_max = if w > w1_max { w } else { w1_max };
            w1_min = if w < w1_min { w } else { w1_min };
          }
        }
      }
    }

    let mut w2_max = 0.0;
    let mut w2_min = 0.0;
    for phase in 0..PHASES {
      for n in 0..N1 {
        let w = self.l2[phase].w2[n];
        w2_max = if w > w2_max { w } else { w2_max };
        w2_min = if w < w2_min { w } else { w2_min };
      }
    }

    eprintln!("w1 max {w1_max:+11.6}");
    eprintln!("w1 min {w1_min:+11.6}");
    eprintln!();
    unsafe {
      eprintln!("s1 max {S1_MAX:+11.6}");
      eprintln!("s1 mIn {S1_MIN:+11.6}");
    }
    eprintln!();
    eprintln!("w2 max {w2_max:+11.6}");
    eprintln!("w2 min {w2_min:+11.6}");
  }

  pub fn evaluate(&self, state : &State) -> f32
  {
    use crate::color::{WB, Color::*};
    use crate::misc::vmirror;
    use crate::piece::{KQRBNP, Piece::*};

    let wk_sq =         state.boards[WhiteKing].trailing_zeros() as usize ;
    let bk_sq = vmirror(state.boards[BlackKing].trailing_zeros() as usize);
    let move_rn = king_region(match state.turn { White => wk_sq, Black => bk_sq });
    let wait_rn = king_region(match state.turn { White => bk_sq, Black => wk_sq });

    let men = (state.sides[0] | state.sides[1]).count_ones() as usize;
    let men = std::cmp::min(men - 2, 30);
    let phase = men / 8;

    let mut s1 = [[0.0; N1]; 2];

    let mv_ps = &self.l1[move_rn];
    let wt_ps = &self.l1[wait_rn];

    let sm_ptr = &raw mut s1[SideToMove ] as *mut f32;
    let sw_ptr = &raw mut s1[SideWaiting] as *mut f32;

    let rel_mv = if state.turn == White { SameSide } else { OppoSide };
    let rel_wt = if state.turn == White { OppoSide } else { SameSide };
    for kind in KQRBNP {
      let mut sources = state.boards[White+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let ofs = (kind as usize) * 64 + src;
        incr::<f32, N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
        incr::<f32, N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
        sources &= sources - 1;
      }
    }

    let rel_mv = if state.turn == White { OppoSide } else { SameSide };
    let rel_wt = if state.turn == White { SameSide } else { OppoSide };
    for kind in KQRBNP {
      let mut sources = state.boards[Black+kind];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let ofs = (kind as usize) * 64 + vmirror(src);
        incr::<f32, N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
        incr::<f32, N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
        sources &= sources - 1;
      }
    }

    unsafe {
      for c in WB {
        for n in 0..N1 {
          S1_MAX = if s1[c][n] > S1_MAX { s1[c][n] } else { S1_MAX };
          S1_MIN = if s1[c][n] < S1_MIN { s1[c][n] } else { S1_MIN };
        }
      }
    }

    for c in WB { for n in 0..N1 { s1[c][n] = crelu(s1[c][n]); } }
    let a1 = s1;

    let mut m1 = [0.0; N1];
    for side in 0..2 {
      let h = side * N1 / 2;
      for p in 0..N1/2 {
        m1[h + p] = a1[side][p] * a1[side][N1/2 + p];
      }
    }

    let l2_ps = &self.l2[phase];
    let mut s2_reg = [0.0; 8];
    for reg in 0..N1/8 {
      let ofs = reg * 8;
      for lane in 0..8 {
        s2_reg[lane] += m1[ofs+lane] * l2_ps.w2[ofs+lane];
      }
    }
    let s2 = l2_ps.b2 + s2_reg.iter().sum::<f32>();

    return s2;
  }
}

#[repr(align(32))]
#[allow(non_snake_case)]
struct NBuffer {
  s1     : [[f32;  N1]; 2],
  a1     : [[f32;  N1]; 2],
  m1     :  [f32;  N1],
  dE_dm1 :  [f32;  N1],
  dE_ds1 : [[f32;  N1]; 2],
}

impl NBuffer {
  pub const fn zero() -> Self
  {
    return unsafe { std::mem::zeroed() };
  }
}

fn fwd(
  ps     : *const NNetwork,
  buf    : &mut NBuffer,
  mini   : &MiniState,
) -> f32
{
  #![allow(non_snake_case)]
  use crate::color::{WB, Color::*};
  use crate::misc::vmirror;

  let turn = mini.turn();
  let wk_sq =         mini.positions[White][0] as usize ;
  let bk_sq = vmirror(mini.positions[Black][0] as usize);
  let move_rn = king_region(match turn { White => wk_sq, Black => bk_sq });
  let wait_rn = king_region(match turn { White => bk_sq, Black => wk_sq });
  let phase = phase_index(mini);

  buf.s1 = [[0.0_f32; N1]; 2];

  let mv_ps = unsafe { &raw const (*ps).l1[move_rn] };
  let wt_ps = unsafe { &raw const (*ps).l1[wait_rn] };
  let  f_ps = unsafe { &raw const (*ps).l1[REGIONS] };

  for color in WB {
    for slot in 0..16 {
      let posn = mini.positions[color][slot];
      if posn < 0 { continue; }
      let kind = MiniState::KIND[slot];
      let src = posn as usize;
      let ofs = kind as usize * 64
              + match color { White => src, Black => vmirror(src) };
      let rel_mv = if color == mini.turn() { SameSide } else { OppoSide };
      let rel_wt = rel_mv ^ 1;
      let sm_ptr = &raw mut buf.s1[SideToMove ] as *mut f32;
      let sw_ptr = &raw mut buf.s1[SideWaiting] as *mut f32;
      unsafe {
        incr::<f32, N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
        incr::<f32, N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
        incr::<f32, N1>(sm_ptr, &raw const  (*f_ps).w1[rel_mv][ofs] as *const f32);
        incr::<f32, N1>(sw_ptr, &raw const  (*f_ps).w1[rel_wt][ofs] as *const f32);
      }
//    let ofs = 384 + kind as usize;
//    unsafe {
//      incr::<N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
//      incr::<N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
//      incr::<N1>(sm_ptr, &raw const  (*f_ps).w1[rel_mv][ofs] as *const f32);
//      incr::<N1>(sw_ptr, &raw const  (*f_ps).w1[rel_wt][ofs] as *const f32);
//    }
//    if kind != crate::piece::Kind::Bishop { continue; }
//    let ofs = 390 + (src & 1);  // 390 if dark / 391 if light
//    unsafe {
//      incr::<N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
//      incr::<N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
//      incr::<N1>(sm_ptr, &raw const  (*f_ps).w1[rel_mv][ofs] as *const f32);
//      incr::<N1>(sw_ptr, &raw const  (*f_ps).w1[rel_wt][ofs] as *const f32);
//    }
    }
  }
  for var in 0..2 {
    let (piece, posn) = mini.variable[var];
    if posn < 0 { continue; }
    let color = piece.color();
    let kind = piece.kind();
    let src = posn as usize;
    let ofs = kind as usize * 64
            + match color { White => src, Black => vmirror(src) };
    let rel_mv = if color == mini.turn() { SameSide } else { OppoSide };
    let rel_wt = rel_mv ^ 1;
    let sm_ptr = &raw mut buf.s1[SideToMove ] as *mut f32;
    let sw_ptr = &raw mut buf.s1[SideWaiting] as *mut f32;
    unsafe {
      incr::<f32, N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
      incr::<f32, N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
      incr::<f32, N1>(sm_ptr, &raw const  (*f_ps).w1[rel_mv][ofs] as *const f32);
      incr::<f32, N1>(sw_ptr, &raw const  (*f_ps).w1[rel_wt][ofs] as *const f32);
    }
//  let ofs = 384 + kind as usize;
//  unsafe {
//    incr::<N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
//    incr::<N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
//    incr::<N1>(sm_ptr, &raw const  (*f_ps).w1[rel_mv][ofs] as *const f32);
//    incr::<N1>(sw_ptr, &raw const  (*f_ps).w1[rel_wt][ofs] as *const f32);
//  }
//  if kind != crate::piece::Kind::Bishop { continue; }
//  let ofs = 390 + (src & 1);  // 390 if dark / 391 if light
//  unsafe {
//    incr::<N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
//    incr::<N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
//    incr::<N1>(sm_ptr, &raw const  (*f_ps).w1[rel_mv][ofs] as *const f32);
//    incr::<N1>(sw_ptr, &raw const  (*f_ps).w1[rel_wt][ofs] as *const f32);
//  }
  }

  for side in 0..2 {
    for n in 0..N1 { buf.a1[side][n] = crelu(buf.s1[side][n]); }
  }

  for side in 0..2 {
    let h = side * N1 / 2;
    for p in 0..N1/2 {
      buf.m1[h + p] = buf.a1[side][p] * buf.a1[side][N1/2 + p];
    }
  }

  let l2_ps = unsafe { &raw const (*ps).l2[phase] };
  let mut s2_reg = [0.0; 8];
  for reg in 0..N1/8 {
    let ofs = reg * 8;
    for lane in 0..8 {
      s2_reg[lane] += buf.m1[ofs+lane] * unsafe { (*l2_ps).w2[ofs+lane] };
    }
  }
  let s2 = unsafe { (*l2_ps).b2 } + s2_reg.iter().sum::<f32>();

  return s2;
}

fn fwdbackprop(
  ps     : *const NNetwork,
  ds     : *mut   NNetwork,
  buf    : &mut NBuffer,
  mini   : &MiniState,
  target : f32,
) -> f32
{
  #![allow(non_snake_case)]
  use crate::color::{WB, Color::*};
  use crate::misc::vmirror;

  let turn = mini.turn();
  let wk_sq =         mini.positions[White][0] as usize ;
  let bk_sq = vmirror(mini.positions[Black][0] as usize);
  let move_rn = king_region(match turn { White => wk_sq, Black => bk_sq });
  let wait_rn = king_region(match turn { White => bk_sq, Black => wk_sq });
  let phase = phase_index(mini);

  // Forward

  // let mut inp = [[0.0_f32; FNp - Np]; 2];
  buf.s1 = [[0.0_f32; N1]; 2];

  let mv_ps = unsafe { &raw const (*ps).l1[move_rn] };
  let wt_ps = unsafe { &raw const (*ps).l1[wait_rn] };
  let  f_ps = unsafe { &raw const (*ps).l1[REGIONS] };

  for color in WB {
    for slot in 0..16 {
      let posn = mini.positions[color][slot];
      if posn < 0 { continue; }
      let kind = MiniState::KIND[slot];
      let src = posn as usize;
      let ofs = kind as usize * 64
              + match color { White => src, Black => vmirror(src) };
//    let side   = if color == mini.turn() { SideToMove } else { SideWaiting };
      let rel_mv = if color == mini.turn() {  SameSide  } else {  OppoSide   };
      let rel_wt = rel_mv ^ 1;
      let sm_ptr = &raw mut buf.s1[SideToMove ] as *mut f32;
      let sw_ptr = &raw mut buf.s1[SideWaiting] as *mut f32;
      unsafe {
        incr::<f32, N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
        incr::<f32, N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
        incr::<f32, N1>(sm_ptr, &raw const  (*f_ps).w1[rel_mv][ofs] as *const f32);
        incr::<f32, N1>(sw_ptr, &raw const  (*f_ps).w1[rel_wt][ofs] as *const f32);
      }
//    let ofs = 384 + kind as usize;
//    inp[side][kind as usize] += 1.0;
//    unsafe {
//      incr::<N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
//      incr::<N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
//      incr::<N1>(sm_ptr, &raw const  (*f_ps).w1[rel_mv][ofs] as *const f32);
//      incr::<N1>(sw_ptr, &raw const  (*f_ps).w1[rel_wt][ofs] as *const f32);
//    }
//    if kind != crate::piece::Kind::Bishop { continue; }
//    let ofs = 390 + (src & 1);  // 390 if dark / 391 if light
//    inp[side][6 + (src & 1)] += 1.0;
//    unsafe {
//      incr::<N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
//      incr::<N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
//      incr::<N1>(sm_ptr, &raw const  (*f_ps).w1[rel_mv][ofs] as *const f32);
//      incr::<N1>(sw_ptr, &raw const  (*f_ps).w1[rel_wt][ofs] as *const f32);
//    }
    }
  }
  for var in 0..2 {
    let (piece, posn) = mini.variable[var];
    if posn < 0 { continue; }
    let color = piece.color();
    let kind = piece.kind();
    let src = posn as usize;
    let ofs = kind as usize * 64
            + match color { White => src, Black => vmirror(src) };
//  let side   = if color == mini.turn() { SideToMove } else { SideWaiting };
    let rel_mv = if color == mini.turn() {  SameSide  } else {  OppoSide   };
    let rel_wt = rel_mv ^ 1;
    let sm_ptr = &raw mut buf.s1[SideToMove ] as *mut f32;
    let sw_ptr = &raw mut buf.s1[SideWaiting] as *mut f32;
    unsafe {
      incr::<f32, N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
      incr::<f32, N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
      incr::<f32, N1>(sm_ptr, &raw const  (*f_ps).w1[rel_mv][ofs] as *const f32);
      incr::<f32, N1>(sw_ptr, &raw const  (*f_ps).w1[rel_wt][ofs] as *const f32);
    }
//  let ofs = 384 + kind as usize;
//  inp[side][kind as usize] += 1.0;
//  unsafe {
//    incr::<N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
//    incr::<N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
//    incr::<N1>(sm_ptr, &raw const  (*f_ps).w1[rel_mv][ofs] as *const f32);
//    incr::<N1>(sw_ptr, &raw const  (*f_ps).w1[rel_wt][ofs] as *const f32);
//  }
//  if kind != crate::piece::Kind::Bishop { continue; }
//  let ofs = 390 + (src & 1);  // 390 if dark / 391 if light
//  inp[side][6 + (src & 1)] += 1.0;
//  unsafe {
//    incr::<N1>(sm_ptr, &raw const (*mv_ps).w1[rel_mv][ofs] as *const f32);
//    incr::<N1>(sw_ptr, &raw const (*wt_ps).w1[rel_wt][ofs] as *const f32);
//    incr::<N1>(sm_ptr, &raw const  (*f_ps).w1[rel_mv][ofs] as *const f32);
//    incr::<N1>(sw_ptr, &raw const  (*f_ps).w1[rel_wt][ofs] as *const f32);
//  }
  }

  // TODO simd crelu
  for side in 0..2 {
    for n in 0..N1 { buf.a1[side][n] = crelu(buf.s1[side][n]); }
  }

  for side in 0..2 {
    let h = side * N1 / 2;
    for p in 0..N1/2 {
      buf.m1[h + p] = buf.a1[side][p] * buf.a1[side][N1/2 + p];
    }
  }

  let l2_ps = unsafe { &raw const (*ps).l2[phase] };
  let mut s2_reg = [0.0; 8];
  for reg in 0..N1/8 {
    let ofs = reg * 8;
    for lane in 0..8 {
      s2_reg[lane] += buf.m1[ofs+lane] * unsafe { (*l2_ps).w2[ofs+lane] };
    }
  }
  let s2 = unsafe { (*l2_ps).b2 } + s2_reg.iter().sum::<f32>();

  let error = expectation(s2) - target;

  // Backward

  let dE_ds2 = error * d_expectation(s2);

  unsafe {
    let l2_ds = &raw mut (*ds).l2[phase];
    (*l2_ds).b2 += dE_ds2;

    for x in 0..N1 {
      (*l2_ds).w2[x] += buf.m1[x] * dE_ds2;
    }

    for x in 0..N1 {
      buf.dE_dm1[x] = (*l2_ps).w2[x] * dE_ds2;
    }

    for side in 0..2 {
      let h = side * N1 / 2;
      for p in 0..N1/2 {
        buf.dE_ds1[side][p] = buf.dE_dm1[h + p] * buf.a1[side][N1/2 + p];
        buf.dE_ds1[side][N1/2 + p] = buf.dE_dm1[h + p] * buf.a1[side][p];
        // let m = f32x8::from_slice(&dE_dm1[(p + 0)..(p + 8)]);
        // let aL = f32x8::from_slice(&a1[side][(y + 0)..(y +  8)]);
        // let aH = f32x8::from_slice(&a1[side][(y + 8)..(y + 16)]);
        // let dsL = &mut dE_ds1[side][(y + 0)..(y + 8)];
        // let dsH = &mut dE_ds1[side][(y + 0)..(y + 8)];
        // (m * aH).copy_to_slice(dsL);
        // (m * aL).copy_to_slice(dsH);
      }
    }
    for side in 0..2 {
      for n in 0..N1 {
        buf.dE_ds1[side][n] *= d_crelu(buf.s1[side][n]);
      }
    }

    let mv_ds = &raw mut (*ds).l1[move_rn];
    let wt_ds = &raw mut (*ds).l1[wait_rn];
    let  f_ds = &raw mut (*ds).l1[REGIONS];

    for color in WB {
      for slot in 0..16 {
        let posn = mini.positions[color][slot];
        if posn < 0 { continue; }
        let kind = MiniState::KIND[slot];
        let src = posn as usize;
        let ofs = kind as usize * 64
                + match color { White => src, Black => vmirror(src) };
        let rel_mv = if color == mini.turn() { SameSide } else { OppoSide };
        let rel_wt = rel_mv ^ 1;

        for n in 0..N1 { (*mv_ds).w1[rel_mv][ofs][n] += buf.dE_ds1[SideToMove ][n]; }
        for n in 0..N1 { (*wt_ds).w1[rel_wt][ofs][n] += buf.dE_ds1[SideWaiting][n]; }
        for n in 0..N1 { ( *f_ds).w1[rel_mv][ofs][n] += buf.dE_ds1[SideToMove ][n]; }
        for n in 0..N1 { ( *f_ds).w1[rel_wt][ofs][n] += buf.dE_ds1[SideWaiting][n]; }
      }
    }
    for var in 0..2 {
      let (piece, posn) = mini.variable[var];
      if posn < 0 { continue; }
      let color = piece.color();
      let kind = piece.kind();
      let src = posn as usize;
      let ofs = kind as usize * 64
              + match color { White => src, Black => vmirror(src) };
      let rel_mv = if color == mini.turn() { SameSide } else { OppoSide };
      let rel_wt = rel_mv ^ 1;

      for n in 0..N1 { (*mv_ds).w1[rel_mv][ofs][n] += buf.dE_ds1[SideToMove ][n]; }
      for n in 0..N1 { (*wt_ds).w1[rel_wt][ofs][n] += buf.dE_ds1[SideWaiting][n]; }
      for n in 0..N1 { ( *f_ds).w1[rel_mv][ofs][n] += buf.dE_ds1[SideToMove ][n]; }
      for n in 0..N1 { ( *f_ds).w1[rel_wt][ofs][n] += buf.dE_ds1[SideWaiting][n]; }
    }

//  for side in 0..2 {
//    for i in 0..8 {
//      for n in 0..N1 {
//        let x = 384 + i;
//        (*mv_ds).w1[side][x][n] += buf.dE_ds1[SideToMove ][n] * inp[ side ][i];
//        (*wt_ds).w1[side][x][n] += buf.dE_ds1[SideWaiting][n] * inp[side^1][i];
//        ( *f_ds).w1[side][x][n] += buf.dE_ds1[SideToMove ][n] * inp[ side ][i];
//        ( *f_ds).w1[side][x][n] += buf.dE_ds1[SideWaiting][n] * inp[side^1][i];
//      }
//    }
//  }
  }

  return error;
}

impl MiniState {
  pub fn antiphase(&self) -> u8
  {
    const SLOT_VALUE : [u8; 16] = [
      0, 12, 6, 6, 4, 4, 4, 4,
      1,  1, 1, 1, 1, 1, 1, 1
    ];
    const KIND_VALUE : [u8; 8] = [
      0, 12, 6, 4, 4, 1, 0, 0
    ];

    let mut antiphase = 0;
    for color in crate::color::WB {
      let posns = &self.positions[color];
      for ofs in 1..16 {
        if posns[ofs] < 0 { continue; }
        antiphase += SLOT_VALUE[ofs];
      }
    }
    for n in 0..2 {
      let (piece, posn) = &self.variable[n];
      if *posn < 0 { continue; }
      antiphase += KIND_VALUE[piece.kind()];
    }
    return antiphase;
  }
}

fn stats(c : &NNetwork, d : &NNetwork) -> (f64, f64) {
  let c_ary = c as *const NNetwork as *const f32;
  let d_ary = d as *const NNetwork as *const f32;
  let mut num_nonzero = 0;
  let mut num_flip = 0;
  for idx in 0..NNetwork::NUM_PARAMS {
    let c_val = unsafe { *c_ary.add(idx) };
    let d_val = unsafe { *d_ary.add(idx) };
    if c_val == 0.0 { continue; }
    if d_val == 0.0 { continue; }
    num_nonzero += 1;
    if c_val.is_sign_negative() != d_val.is_sign_negative() { num_flip += 1; }
  }
  let frac_nonzero = num_nonzero as f64 / NNetwork::NUM_PARAMS as f64;
  if num_nonzero == 0 { return (0.0, frac_nonzero); }
  let frac_flip = num_flip as f64 / num_nonzero as f64;
  return (frac_flip, frac_nonzero);
}

const alpha   : f32 = 1.0 / 64.0;
const beta    : f32 = 1.0 - 1.0 / 8.0;
const gamma   : f32 = 1.0 - 1.0 / 512.0;
const epsilon : f32 = 1.0 / 65_536.0;

const BATCH_SIZE : usize =      28_800; // (2² × 3² × 5²) × 2⁵
const EPOCH_SIZE : usize = 117_964_800; // (2² × 3² × 5²) × 2¹⁷

static mut POSITIONS : Vec<MiniState> = Vec::new();

use std::sync::atomic::{AtomicBool, AtomicUsize};

#[repr(align(64))]
struct Flag { pub inner : AtomicBool }

static mut COMPLETE : [Flag; 96] = [
  const { Flag { inner: AtomicBool::new(false) } }; 96
];

static mut NUM_ACTIVE : AtomicUsize = AtomicUsize::new(0);
pub static mut PARAMS : NNetwork = NNetwork::zero();
static mut DERIVS : Vec<NNetwork> = Vec::new();

pub fn trainn(
  path : &str,
  num_threads : usize,
  seed : u64,
  init : bool,
  skip : usize
) -> std::io::Result<()>
{
  use std::fs::File;
  use std::io::{BufReader, BufRead, Seek};
  use std::sync::atomic::Ordering;
  use crate::rand::{Rand, init_rand};

  let batch_per_epoch = EPOCH_SIZE / BATCH_SIZE;
  let posn_per_thread = BATCH_SIZE / num_threads;

  let affinity = crate::util::get_affinity();
  init_rand(seed);

  unsafe {
    if init { PARAMS.perturb(); }
    for _ in 0..num_threads {
      DERIVS.push(NNetwork::zero());
    }
  }

  let mut ds = NNetwork::zero();
  let mut ms = NNetwork::zero();
  let mut vs = NNetwork::zero();

  let mut prev_ds = NNetwork::zero();
  let mut ewma_flip = 0.0;
  let mut ewma_non0 = 0.0;

  let mut reader = BufReader::new(File::open(path)?);

  let mut update_count = 0;

  for epoch in 0..1000 {
    // Step 1. Load positions
    eprint!("  Loading...\x1B[K\r");
    let mut buffer = String::new();
    for index in 0..EPOCH_SIZE {
      if reader.read_line(&mut buffer)? == 0 {
        reader.rewind()?;
        if reader.read_line(&mut buffer)? == 0 {
          panic!("empty file");
        }
      }
      let ary = <&[u8; 40]>::try_from(&buffer.as_bytes()[..40]).unwrap();
      let mini = MiniState::from_quick(ary);
      unsafe {
        if epoch > 0 { POSITIONS[index] = mini; } else { POSITIONS.push(mini); }
      }
      buffer.clear();
    }
    // if epoch < skip { continue; }
    eprint!("  Permuting...\x1B[K\r");
    for x in 0..EPOCH_SIZE {
      let k = u32::rand() as usize % (EPOCH_SIZE - x);
      unsafe { POSITIONS.swap(x, x+k); }
    }

    // Step 2. Start the workers
    unsafe { NUM_ACTIVE.store(num_threads, Ordering::Release); }
    let mut workers = Vec::new();
    for id in 0..num_threads {
      let supervisor = std::thread::current();
      let proc = crate::util::assign_proc(&affinity, id);
      workers.push(
        std::thread::Builder::new()
        .name(format!("{id}"))
        .stack_size(2_097_152)
        .spawn(move || -> f64 {
          // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          if !(proc < 0) { crate::util::set_affinity(proc as usize); }
          let ps = &raw const PARAMS;
          let ds = unsafe { &raw mut DERIVS[id] };
          let mut buf = NBuffer::zero();
          let mut base = posn_per_thread * id;
          let mut total_error : f64 = 0.0;
          for _ in 0..batch_per_epoch {
            reset(ds);

            let mut index = base;
            for _ in 0..posn_per_thread {
              let mini = unsafe { &POSITIONS[index] };

              let mut target;
              if num_subjects(mini) > 4 {
                let cp = mini.score.from(mini.turn()).as_i16();
                let cp = std::cmp::max(cp, -100_00);
                let cp = std::cmp::min(cp,  100_00);
                let score = cp as f32 / 100.0;
                target = expectation(score);

                use crate::score::{CoOutcome, PovOutcome};
                if mini.outcome() != CoOutcome::Unknown {
                  let antiphase = mini.antiphase();
                  let phase = 96 - std::cmp::min(96, antiphase);
                  let phase = phase as f32 / 96.0;
                  let x = phase * phase;
                  let result = match mini.outcome().from(mini.turn()) {
                    PovOutcome::Unknown => panic!(),
                    PovOutcome::Loss    => 0.0,
                    PovOutcome::Draw    => 0.5,
                    PovOutcome::Win     => 1.0,
                  };
                  target = target * (1.0 - x) + result * x;
                }
              }
              else {
                use std::cmp::Ordering;
                use crate::syzygy::probe_syzygy_wdl;
                let state = State::from(mini);
                let score = probe_syzygy_wdl(&state, 0).unwrap().as_i16();
                target = match score.cmp(&0) {
                  Ordering::Less    => 0.0,
                  Ordering::Equal   => 0.5,
                  Ordering::Greater => 1.0,
                };
              }

              let error = fwdbackprop(ps, ds, &mut buf, mini, target) as f64;
              total_error += error * error;
              index += 1;
            }

            let mut pair = 1;
            let mut num_covered = 1;
            while id & pair == 0 && num_covered < num_threads {
              let conjugate = id ^ pair;
              if conjugate < num_threads {
                unsafe {
                  while !COMPLETE[conjugate].inner.load(Ordering::Acquire) {
                    std::arch::x86_64::_mm_pause();
                  }
                  incr::<f32, {NNetwork::NUM_PARAMS}>(
                    ds as *mut f32,
                    &raw const DERIVS[conjugate] as *const f32
                  );
                }
              }
              pair <<= 1;
              num_covered *= 2;
            }

            unsafe { COMPLETE[id].inner.store(true, Ordering::Release); }

            let num_active = unsafe {
              NUM_ACTIVE.fetch_sub(1, Ordering::AcqRel)
            };
            if num_active == 1 { supervisor.unpark(); }
            loop {
              std::thread::park();
              unsafe {
                if NUM_ACTIVE.load(Ordering::Acquire) > 0 { break; }
              }
            }

            base += BATCH_SIZE;
          }

          total_error
          // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        })
        .unwrap()
      );
    }

    let t = (epoch + skip) as f32;
    let t = t / 100.0;
    let var = 0.75 + 0.25 * (t * (std::f32::consts::TAU * 7.0 / 6.0)).cos();
    let exp = 0.25f32.powf(t * (1.0 / 6.0));
    let step = alpha * (var * exp);

    // Step 3. Manage each batch
    for batch in 0..batch_per_epoch {
      loop {
        std::thread::park();
        unsafe { if NUM_ACTIVE.load(Ordering::Acquire) == 0 { break; } }
      }

      unsafe {
        ds.scale_from(&raw const DERIVS[0], (BATCH_SIZE as f32).recip());
      }

      if update_count < 64 {
        let bx0 = beta.powi(update_count);
        let bx1 = bx0 * beta;
        let d_bias = (1.0 - bx1).recip();   // initial 1/β and limit 1
        let c_bias = (1.0 - bx0) * d_bias;  // initial  0  and limit 1

        ms.scale(beta * c_bias);
        ms.incr_scaled(&raw const ds, (1.0 - beta) * d_bias);

        let gx0 = gamma.powi(update_count);
        let gx1 = gx0 * gamma;
        let d_bias = (1.0 - gx1).recip();
        let c_bias = (1.0 - gx0) * d_bias;

        vs.scale(gamma * c_bias);
        vs.incr_square_scaled(&raw const ds, (1.0 - gamma) * d_bias);
      }
      else {
        ms.scale(beta);
        ms.incr_scaled(&ds, 1.0 - beta);
        vs.scale(gamma);
        vs.incr_square_scaled(&ds, 1.0 - gamma);
      }

      if update_count < 2048 {
        let u = update_count as f32 * (1.0 / 2048.0);
        unsafe { PARAMS.update(&raw const ms, &raw const vs, step * u * u); }
      }
      else {
        unsafe { PARAMS.update(&raw const ms, &raw const vs, step); }
      }

      // unsafe { PARAMS.regularize(1.0 / 32768.0); }

      unsafe {
        for id in 0..num_threads {
          COMPLETE[id].inner.store(false, Ordering::Release);
        }
        NUM_ACTIVE.store(num_threads, Ordering::Release);
        for handle in &workers { handle.thread().unpark(); }
      }

      let (frac_flip, frac_nonzero) = stats(&ds, &prev_ds);
      if update_count < 4 {
        ewma_flip = frac_flip;
        ewma_non0 = frac_nonzero;
      }
      else {
        ewma_flip = (ewma_flip * 1023.0 + frac_flip   ) / 1024.0;
        ewma_non0 = (ewma_non0 * 1023.0 + frac_nonzero) / 1024.0;
      }
      prev_ds = ds.clone();

      let epon = epoch + skip;
      let progress = (batch + 1) as f32 / batch_per_epoch as f32 * 100.0;
      eprint!("{epon:3}  {progress:7.3}%    \x1B[2m{frac_flip:8.6} {frac_nonzero:8.6}\x1B[22m\x1B[K\r");

      update_count += 1;
    }

    // Step 4. We’ve finished an epoch!
    let mut total_error : f64 = 0.0;
    for handle in workers { total_error += handle.join().unwrap(); }
    let total_error = total_error / EPOCH_SIZE as f64;

    let epon = epoch + skip;
    unsafe { PARAMS.save(&format!("tmp/{epon:03}"))?; }
    unsafe { PARAMS.image(&format!("tmp/{epon:03}"))?; }

    eprintln!("{epon:3} {total_error:12.10} \x1B[2m{ewma_flip:8.6} {ewma_non0:8.6}\x1B[22m\x1B[K");
  }

  return Ok(());
}

static PALETTE : [[u8; 3]; 257] = [
  [128, 199, 248], [124, 197, 248], [120, 195, 249], [116, 194, 250],
  [112, 192, 250], [109, 190, 251], [105, 189, 251], [102, 187, 251],
  [ 98, 185, 252], [ 95, 183, 252], [ 91, 182, 252], [ 88, 180, 253],
  [ 85, 178, 253], [ 82, 176, 253], [ 79, 174, 253], [ 76, 172, 253],
  [ 73, 170, 253], [ 70, 169, 253], [ 67, 167, 253], [ 65, 165, 253],
  [ 62, 163, 253], [ 60, 161, 253], [ 58, 159, 252], [ 56, 157, 252],
  [ 54, 155, 252], [ 52, 153, 251], [ 50, 151, 251], [ 49, 149, 250],
  [ 47, 147, 250], [ 46, 145, 249], [ 45, 143, 249], [ 44, 141, 248],
  [ 43, 138, 247], [ 42, 136, 247], [ 42, 134, 246], [ 42, 132, 245],
  [ 41, 130, 244], [ 41, 128, 243], [ 41, 126, 242], [ 41, 124, 241],
  [ 42, 122, 240], [ 42, 120, 239], [ 42, 118, 238], [ 43, 116, 236],
  [ 43, 113, 235], [ 44, 111, 234], [ 44, 109, 232], [ 45, 107, 231],
  [ 45, 105, 230], [ 46, 103, 228], [ 46, 101, 226], [ 47,  99, 225],
  [ 48,  97, 223], [ 48,  95, 221], [ 49,  93, 220], [ 49,  91, 218],
  [ 50,  89, 216], [ 51,  87, 214], [ 51,  85, 212], [ 52,  83, 210],
  [ 52,  81, 208], [ 52,  79, 206], [ 53,  77, 204], [ 53,  76, 202],
  [ 53,  74, 200], [ 54,  72, 198], [ 54,  70, 195], [ 54,  68, 193],
  [ 55,  67, 191], [ 55,  65, 188], [ 55,  63, 186], [ 55,  61, 183],
  [ 55,  60, 181], [ 55,  58, 178], [ 55,  56, 176], [ 55,  55, 173],
  [ 55,  53, 171], [ 55,  52, 168], [ 55,  50, 165], [ 54,  49, 163],
  [ 54,  47, 160], [ 54,  46, 157], [ 54,  45, 154], [ 53,  43, 151],
  [ 53,  42, 148], [ 53,  41, 146], [ 52,  39, 143], [ 52,  38, 140],
  [ 51,  37, 137], [ 51,  36, 134], [ 50,  35, 131], [ 50,  34, 128],
  [ 49,  33, 125], [ 48,  32, 122], [ 48,  31, 118], [ 47,  30, 115],
  [ 46,  29, 112], [ 46,  29, 109], [ 45,  28, 106], [ 44,  27, 103],
  [ 43,  26, 100], [ 42,  26,  96], [ 42,  25,  93], [ 41,  24,  90],
  [ 40,  24,  87], [ 39,  23,  84], [ 38,  23,  80], [ 37,  22,  77],
  [ 36,  22,  74], [ 35,  21,  71], [ 34,  21,  68], [ 32,  20,  64],
  [ 31,  20,  61], [ 30,  19,  58], [ 29,  19,  55], [ 28,  18,  52],
  [ 27,  18,  48], [ 25,  17,  45], [ 24,  17,  42], [ 23,  16,  39],
  [ 22,  16,  36], [ 20,  15,  33], [ 19,  15,  30], [ 18,  14,  26],
  [ 17,  14,  23], [ 15,  13,  20], [ 14,  12,  17], [ 12,  12,  14],
  [ 11,  11,  11],
  [ 14,  11,  11], [ 18,  12,  12], [ 21,  12,  12], [ 24,  12,  12],
  [ 27,  12,  13], [ 29,  12,  13], [ 32,  12,  13], [ 35,  11,  13],
  [ 38,  11,  14], [ 41,  11,  14], [ 44,  11,  14], [ 46,  10,  14],
  [ 49,  10,  14], [ 52,   9,  14], [ 55,   9,  14], [ 57,   8,  14],
  [ 60,   7,  14], [ 63,   7,  14], [ 66,   6,  14], [ 68,   5,  14],
  [ 71,   4,  14], [ 74,   3,  14], [ 76,   2,  14], [ 79,   1,  14],
  [ 82,   0,  14], [ 84,   0,  14], [ 87,   0,  14], [ 89,   0,  15],
  [ 91,   0,  15], [ 93,   0,  15], [ 95,   0,  15], [ 98,   0,  15],
  [100,   0,  15], [102,   0,  15], [105,   0,  16], [107,   1,  16],
  [109,   0,  16], [112,   0,  16], [114,   0,  16], [116,   1,  16],
  [119,   0,  15], [121,   0,  15], [124,   0,  15], [126,   0,  15],
  [128,   0,  14], [131,   0,  14], [133,   0,  14], [136,   0,  13],
  [138,   0,  13], [141,   0,  12], [143,   0,  12], [145,   1,  11],
  [148,   0,  10], [151,   0,   9], [153,   1,   8], [155,   1,   7],
  [158,   1,   6], [160,   1,   4], [163,   1,   3], [165,   1,   2],
  [168,   1,   0], [170,   4,   0], [172,   7,   0], [174,  10,   0],
  [176,  13,   0], [178,  16,   0], [180,  19,   0], [182,  22,   0],
  [184,  24,   0], [186,  27,   1], [188,  28,   0], [190,  31,   1],
  [192,  33,   0], [194,  35,   1], [196,  37,   0], [199,  38,   0],
  [200,  41,   1], [202,  43,   1], [204,  45,   2], [206,  48,   4],
  [207,  50,   6], [209,  53,   9], [210,  55,  11], [212,  58,  14],
  [214,  60,  16], [215,  63,  18], [216,  65,  21], [218,  68,  23],
  [219,  70,  26], [221,  73,  28], [222,  75,  30], [223,  78,  33],
  [224,  80,  35], [226,  82,  37], [227,  85,  40], [228,  87,  42],
  [229,  90,  44], [230,  92,  47], [231,  95,  49], [232,  97,  52],
  [233, 100,  54], [234, 103,  57], [235, 105,  60], [236, 108,  62],
  [237, 110,  65], [238, 113,  67], [238, 115,  70], [239, 118,  73],
  [240, 120,  76], [240, 123,  78], [241, 125,  81], [241, 128,  84],
  [242, 131,  87], [242, 133,  90], [243, 136,  93], [243, 138,  96],
  [244, 141,  99], [244, 143, 102], [244, 146, 105], [244, 149, 109],
  [244, 151, 112], [245, 154, 115], [245, 156, 118], [245, 159, 122],
  [245, 161, 125], [244, 164, 129], [244, 167, 132], [244, 169, 136],
];
