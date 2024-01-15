#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

use crate::color::{WB, Color::*};
use crate::conv::*;
use crate::dest::*;
use crate::misc::{vmirror, piece_destinations, pawn_attacks};
use crate::piece::{KQRBNP, Kind, Kind::*};
use crate::rand::{Rand, RandDist, init_rand};
use crate::state::{MiniState, State};

use std::fs::File;
use std::io::{Read, Write, BufRead, BufReader, BufWriter, Error};
use std::sync::atomic::{AtomicUsize, AtomicU32, Ordering};

struct Guard<const B : bool> { }
impl <const B : bool> Guard<B> {
  const CHECK : () = assert!(B);
  fn assert() { let _ = Self::CHECK; }
}
macro_rules! static_assert {
  ($cond:expr) => { Guard::<{$cond}>::assert(); }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn compress(x : f32) -> f32
{
  let sq = x * x;
  let sq = if x < 0.0 { sq * 16.0 } else { sq };
  return x / (3.0 + sq).sqrt();
}

fn d_compress(x : f32) -> f32
{
  let sq = x * x;
  let sq = if x < 0.0 { sq * 16.0 } else { sq };
  let dn = 3.0 + sq;
  return 3.0 / (dn * dn * dn).sqrt();
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

const INP : usize = 36;
const HL1 : usize = 64;
const OUT : usize =  7;

#[derive(Clone)]
#[repr(align(32))]
pub struct PolicyNetwork {
  pub cw1 : [[f32; INP]; HL1],
  pub cb1 : [ Board    ; HL1],
  pub gw1 : [[f32; HL1]; HL1],
  pub gb1 : [ f32      ; HL1],

  pub cw2 : [[Kernel; HL1]; OUT],
  pub cb2 : [ Board       ; OUT],
}

#[repr(align(32))]
pub struct PolicyBuffer {
  pub a1 : [ZxtBoard; HL1]
}

impl PolicyBuffer {
  pub const fn zero() -> Self
  {
    const SZ : usize = std::mem::size_of::<PolicyBuffer>();
    union Empty {
      ary : [u8; SZ],
      net : std::mem::ManuallyDrop<PolicyBuffer>
    }
    const ZERO : Empty = Empty { ary: [0; SZ] };
    return std::mem::ManuallyDrop::<PolicyBuffer>::into_inner(unsafe { ZERO.net });
  }
}

const NUM_PARAMS : usize = std::mem::size_of::<PolicyNetwork>() / std::mem::size_of::<f32>();

impl PolicyNetwork {
  pub const fn zero() -> Self
  {
    const SZ : usize = std::mem::size_of::<PolicyNetwork>();
    union Empty {
      ary : [u8; SZ],
      net : std::mem::ManuallyDrop<PolicyNetwork>
    }
    const ZERO : Empty = Empty { ary: [0; SZ] };
    return std::mem::ManuallyDrop::<PolicyNetwork>::into_inner(unsafe { ZERO.net });
  }

  pub fn reset(&mut self)
  {
    unsafe { *self = std::mem::zeroed(); }
  }

  pub fn read(&self,
    layer : usize, n : usize, x : usize, r : usize, f : usize
  ) -> f32
  {
    return match layer {
      1 => self.cw1[n][x],
      2 => self.cw2[n][x][r*8+f],
      _ => panic!()
    };
  }

  fn scale(&mut self, scalar : f32) {
    unsafe {
      let ary = self as *mut PolicyNetwork as *mut f32;
      for idx in 0..NUM_PARAMS {
        let idx = idx as isize;
        let prod = *ary.offset(idx) * scalar;
        std::ptr::write(ary.offset(idx), prod);
      }
    }
  }

  fn add(&mut self, rhs : &Self) {
    unsafe {
      let self_ary = self as *mut PolicyNetwork as *mut f32;
      let  rhs_ary = rhs as *const PolicyNetwork as *const f32;
      for idx in 0..NUM_PARAMS {
        let idx = idx as isize;
        let sum = *self_ary.offset(idx) + *rhs_ary.offset(idx);
        std::ptr::write(self_ary.offset(idx), sum);
      }
    }
  }

  fn add_scaled(&mut self, rhs : &Self, scalar : f32) {
    unsafe {
      let self_ary = self as *mut PolicyNetwork as *mut f32;
      let  rhs_ary = rhs as *const PolicyNetwork as *const f32;
      for idx in 0..NUM_PARAMS {
        let idx = idx as isize;
        let sum = *self_ary.offset(idx) + *rhs_ary.offset(idx) * scalar;
        std::ptr::write(self_ary.offset(idx), sum);
      }
    }
  }

  fn add_square_scaled(&mut self, rhs : &Self, scalar : f32) {
    unsafe {
      let self_ary = self as *mut PolicyNetwork as *mut f32;
      let  rhs_ary = rhs as *const PolicyNetwork as *const f32;
      for idx in 0..NUM_PARAMS {
        let idx = idx as isize;
        let rhs_value = *rhs_ary.offset(idx);
        let sum = *self_ary.offset(idx) + rhs_value * rhs_value * scalar;
        std::ptr::write(self_ary.offset(idx), sum);
      }
    }
  }

  fn add_diff_sq_scaled(&mut self, rhs : &Self, ofs : &Self, scalar : f32) {
    unsafe {
      let self_ary = self as *mut PolicyNetwork as *mut f32;
      let  rhs_ary = rhs as *const PolicyNetwork as *const f32;
      let  ofs_ary = ofs as *const PolicyNetwork as *const f32;
      for idx in 0..NUM_PARAMS {
        let idx = idx as isize;
        let rhs_value = *rhs_ary.offset(idx);
        let ofs_value = *ofs_ary.offset(idx);
        let diff = rhs_value - ofs_value;
        let sum = *self_ary.offset(idx) + diff * diff * scalar;
        std::ptr::write(self_ary.offset(idx), sum);
      }
    }
  }

  fn update(&mut self, means : &Self, vars : &Self, scalar : f32) {
    unsafe {
      let  self_ary = self as *mut PolicyNetwork as *mut f32;
      let means_ary = means as *const PolicyNetwork as *const f32;
      let  vars_ary = vars  as *const PolicyNetwork as *const f32;
      for idx in 0..NUM_PARAMS {
        let idx = idx as isize;
        let g =  *self_ary.offset(idx);
        let m = *means_ary.offset(idx);
        let v =  *vars_ary.offset(idx);
        let u = g - scalar * m / (v.sqrt() + epsilon);
        std::ptr::write(self_ary.offset(idx), u);
      }
    }
  }

  pub fn perturb(&mut self)
  {
    let s = (INP as f32).sqrt().recip();
    for q in 0..HL1 {
      for p in 0..INP {
        self.cw1[q][p] = f32::triangular() * s;
      }
    }

    let s = (HL1 as f32).sqrt().recip();
    for q in 0..HL1 {
      for p in 0..HL1 {
        self.gw1[q][p] = f32::triangular() * s;
      }
    }

    let s = ((HL1 * K * K) as f32).sqrt().recip();
    for q in 0..OUT {
      for p in 0..HL1 {
        for x in 0..K { for y in 0..K { self.cw2[q][p][x*8+y] = f32::triangular() * s; } }
      }
    }
  }

  fn checksum(&self) -> u64
  {
    let mut lo : u32 = 0;
    let mut hi : u32 = 0;
    let array = unsafe { std::mem::transmute::<_,&[u32; NUM_PARAMS]>(self) };
    for x in &array[0..NUM_PARAMS] {
      let (sum, overflow) = lo.overflowing_add(*x);
      lo = if overflow { sum + 1 } else { sum };
      let (sum, overflow) = hi.overflowing_add(lo);
      hi = if overflow { sum + 1 } else { sum };
    }
    return ((hi as u64) << 32) | (lo as u64);
  }

  pub fn load(path : &str) -> std::io::Result<Self>
  {
    let mut fh = File::open(path)?;
    let mut sgntr = [0; 4];
    let mut array = [0; 4*NUM_PARAMS];
    let mut check = [0; 8];
    fh.read_exact(&mut sgntr)?;
    if sgntr != "EXPO".as_bytes() {
      return Err(Error::other("missing signature"));
    }
    fh.read_exact(&mut array)?;
    let network = unsafe { std::mem::transmute::<_,Self>(array) };
    fh.read_exact(&mut check)?;
    if network.checksum() != u64::from_le_bytes(check) {
      return Err(Error::other("checksum mismatch"));
    }
    return Ok(network);
  }

  pub fn save(&self, path : &str) -> std::io::Result<()>
  {
    let mut w = BufWriter::new(File::create(path)?);
    let bytes = unsafe { std::mem::transmute::<_,&[u8; 4*NUM_PARAMS]>(self) };
    w.write_all("EXPO".as_bytes())?;
    w.write_all(bytes)?;
    w.write_all(&self.checksum().to_le_bytes())?;
    return Ok(());
  }

  pub fn save_image(&self, path : &str, layer : usize) -> std::io::Result<()>
  {
    let (feat, prev) = match layer {
      1 => (HL1, INP),
      2 => (OUT, HL1),
      _ => panic!()
    };
    let upscale = 1;
    let width  = K*upscale*prev + prev - 1;
    let height = K*upscale*feat + feat - 1;

    let border = [0, 0, 0];

    let k : f32 = -1.584_962_500_721_156_3;

    let mut out = BufWriter::new(File::create(format!("{}.ppm", path))?);
    writeln!(&mut out, "P6")?;
    writeln!(&mut out, "{} {}", width, height)?;
    writeln!(&mut out, "255")?;

    for n in 0..feat {
      if n != 0 { for _ in 0..width { out.write(&border)?; } }
      for rank in 0..K {
        for _ in 0..upscale {
          let mut leftmost = true;
          for x in 0..prev {
            if !leftmost { out.write(&border)?;}
            leftmost = false;
            for file in 0..K {
              let w = self.read(layer, n, x, rank, file);
              let normed = (1.0 + (w * k).exp2()).recip();
              debug_assert!(
                1.0 >= normed && normed >= 0.0,
                "out of range ({} {})", normed, w
              );
              let c = (normed * 256.0).round() as usize;
              for _ in 0..upscale { out.write(&PALETTE[c])?; }
            }
          }
        }
      }
    }
    out.flush()?;
    let status = std::process::Command::new("convert")
      .arg(&format!("{}.ppm", path)).arg(&format!("{}.png", path)).status()?;
    if status.success() { std::fs::remove_file(format!("{}.ppm", path))?; }

    return Ok(());
  }

  pub fn initialize(&self, state : &State, buf : &mut PolicyBuffer)
  {
    let mut sides  : [u64;  2] = [0;  2];
    let mut boards : [u64; 12] = [0; 12];

    match state.turn {
      White => {
        sides = state.sides;
        for k in KQRBNP {
          boards[(White as usize) * 6 + k as usize] = state.boards[White+k];
          boards[(Black as usize) * 6 + k as usize] = state.boards[Black+k];
        }
      }
      Black => {
        sides[0] = state.sides[1].swap_bytes();
        sides[1] = state.sides[0].swap_bytes();
        for k in KQRBNP {
          boards[(White as usize) * 6 + k as usize] = state.boards[Black+k].swap_bytes();
          boards[(Black as usize) * 6 + k as usize] = state.boards[White+k].swap_bytes();
        }
      }
    }

    let mut inp : [Board; INP] = [[0.0; 64]; INP];

    let composite = sides[White] | sides[Black];
    for color in WB {
      let ofs = color as usize * 6;
      let no_friendly_q  =     composite & !boards[ofs + Queen  as usize];
      let no_friendly_qr = no_friendly_q & !boards[ofs + Rook   as usize];
      let no_friendly_qb = no_friendly_q & !boards[ofs + Bishop as usize];
      for knd in KQRBNP {
        let mut sources = boards[ofs + knd as usize];
        while sources != 0 {
          let src = sources.trailing_zeros() as usize;

          let idx = (color as usize) * 6 + (knd as usize);
          inp[idx][src] = 1.0;

          let idx = 12 + (color as usize) * 6 + (knd as usize);
          let dests =
            if knd == Pawn { pawn_attacks(color, 1 << src) }
            else { piece_destinations(knd, src, composite) };
          let mut attacks = dests;
          while attacks != 0 {
            let atk = attacks.trailing_zeros() as usize;
            inp[idx][atk] += 1.0;
            attacks &= attacks - 1;
          }

          let idx = 24 + (color as usize) * 3 + (knd as usize) - 1;
          let mut batteries = match knd {
            Queen  =>   rook_destinations(no_friendly_qr, src)
            |         bishop_destinations(no_friendly_qb, src),
            Rook   =>   rook_destinations(no_friendly_qr, src),
            Bishop => bishop_destinations(no_friendly_qb, src),
            _ => { sources &= sources - 1; continue; }
          } & !dests;
          while batteries != 0 {
            let bat = batteries.trailing_zeros() as usize;
            inp[idx][bat] += 1.0;
            batteries &= batteries - 1;
          }

          let idx = 30 + (color as usize) * 3 + (knd as usize) - 1;
          let mut pins =
            piece_destinations(knd, src, composite & !dests)
            & !dests;
          while pins != 0 {
            let pin = pins.trailing_zeros() as usize;
            inp[idx][pin] += 1.0;
            pins &= pins - 1;
          }

          sources &= sources - 1;
        }
      }
    }
    let inp = inp;

    // Forward

    let mut cs1 = self.cb1;
    fwd_88P_11PQ_88Q::<INP, HL1>(&inp, &self.cw1, &mut cs1);
    let mut ca1 = [[0.0; 64]; HL1];
    relu_88N_88N::<HL1>(&cs1, &mut ca1);

    let mut gi1 = [0.0; HL1];
    pool_88N_N::<HL1>(&ca1, &mut gi1);

    let mut gs1 = self.gb1;
    fwd_P_Q::<HL1, HL1>(&gi1, &self.gw1, &mut gs1);
    let mut ga1 = [0.0; HL1];
    sigm_N_N::<HL1>(&gs1, &mut ga1);

    mul_88N_N_z88N::<HL1>(&ca1, &ga1, &mut buf.a1);
  }

  pub fn evaluate(&self, buf : &PolicyBuffer, kind : Kind, src : usize, dst : usize) -> f32
  {
    let dst_rank = dst / 8;
    let dst_file = dst % 8;
    let s2_dst = self.cb2[kind][dst]
               + fwd_z88P_KKP_11::<HL1>(&buf.a1, &self.cw2[kind], dst_rank, dst_file);

    let src_rank = src / 8;
    let src_file = src % 8;
    let s2_src = self.cb2[6][src]
               + fwd_z88P_KKP_11::<HL1>(&buf.a1, &self.cw2[6], src_rank, src_file);

    return s2_src + s2_dst;
  }
}

fn fwdback(
  ps   : &PolicyNetwork,
  ds   : &mut PolicyNetwork,
  mini : &MiniState,
  moves_ofs : &u32,
  num_moves : &u32
) -> f64
{
  #![allow(non_snake_case)]

  let mut sides  : [u64;  2] = [0;  2];
  let mut boards : [u64; 12] = [0; 12];

  let mut inp : [Board; INP] = [[0.0; 64]; INP];

  for c in WB {
    let color = if c == mini.turn() { White } else { Black };
    for slot in 0..16 {
      let posn = mini.positions[c][slot];
      if posn < 0 { continue; }
      let knd = MiniState::KIND[slot];
      let src = posn as usize;
      let src = match mini.turn() { White => src, Black => vmirror(src) };
      let idx = (color as usize) * 6 + knd as usize;
      sides[color] |= 1 << src;
      boards[idx]  |= 1 << src;
      inp[idx][src] = 1.0;
    }
  }
  for vdx in 0..2 {
    let (piece, posn) = mini.variable[vdx];
    if posn < 0 { continue; }
    let color = if piece.color() == mini.turn() { White } else { Black };
    let knd = piece.kind();
    let src = posn as usize;
    let src = match mini.turn() { White => src, Black => vmirror(src) };
    let idx = (color as usize) * 6 + knd as usize;
    sides[color] |= 1 << src;
    boards[idx]  |= 1 << src;
    inp[idx][src] = 1.0;
  }

  let composite = sides[White] | sides[Black];
  for color in WB {
    let ofs = color as usize * 6;
    let no_friendly_q  =     composite & !boards[ofs + Queen  as usize];
    let no_friendly_qr = no_friendly_q & !boards[ofs + Rook   as usize];
    let no_friendly_qb = no_friendly_q & !boards[ofs + Bishop as usize];
    for knd in KQRBNP {
      let mut sources = boards[ofs + knd as usize];
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;

        let idx = 12 + (color as usize) * 6 + (knd as usize);
        let dests =
          if knd == Pawn { pawn_attacks(color, 1 << src) }
          else { piece_destinations(knd, src, composite) };
        let mut attacks = dests;
        while attacks != 0 {
          let atk = attacks.trailing_zeros() as usize;
          inp[idx][atk] += 1.0;
          attacks &= attacks - 1;
        }

        let idx = 24 + (color as usize) * 3 + (knd as usize) - 1;
        let mut batteries = match knd {
          Queen  =>   rook_destinations(no_friendly_qr, src)
          |         bishop_destinations(no_friendly_qb, src),
          Rook   =>   rook_destinations(no_friendly_qr, src),
          Bishop => bishop_destinations(no_friendly_qb, src),
          _ => { sources &= sources - 1; continue; }
        } & !dests;
        while batteries != 0 {
          let bat = batteries.trailing_zeros() as usize;
          inp[idx][bat] += 1.0;
          batteries &= batteries - 1;
        }

        let idx = 30 + (color as usize) * 3 + (knd as usize) - 1;
        let mut pins =
          piece_destinations(knd, src, composite & !dests)
          & !dests;
        while pins != 0 {
          let pin = pins.trailing_zeros() as usize;
          inp[idx][pin] += 1.0;
          pins &= pins - 1;
        }

        sources &= sources - 1;
      }
    }
  }
  let inp = inp;

  // Forward

  let mut cs1 = ps.cb1;
  fwd_88P_11PQ_88Q::<INP, HL1>(&inp, &ps.cw1, &mut cs1);
  let mut ca1 = [[0.0; 64]; HL1];
  relu_88N_88N::<HL1>(&cs1, &mut ca1);

  let mut gi1 = [0.0; HL1];
  pool_88N_N::<HL1>(&ca1, &mut gi1);

  let mut gs1 = ps.gb1;
  fwd_P_Q::<HL1, HL1>(&gi1, &ps.gw1, &mut gs1);
  let mut ga1 = [0.0; HL1];
  sigm_N_N::<HL1>(&gs1, &mut ga1);

  let mut a1 = [[0.0; 152]; HL1];
  mul_88N_N_z88N::<HL1>(&ca1, &ga1, &mut a1);

  let mut dE_da1 = [[0.0; 152]; HL1];

  let mut sum_error = 0.0;
  let score = match mini.turn() { White => mini.score, Black => -mini.score } as f32 / 100.0;

  for idx in 0..(*num_moves) {
    let (m, h) = unsafe { DATASET_MOVES[(moves_ofs + idx) as usize] };
    let hyp = match mini.turn() { White => h, Black => -h } as f32 / 100.0;

    let m = m as usize;
    let kind = (m >> 12) &  7;
    let src  = (m >>  6) & 63;
    let dst  =  m        & 63;

    let dst_rank = dst / 8;
    let dst_file = dst % 8;
    let s2_dst = ps.cb2[kind][dst]
               + fwd_z88P_KKP_11::<HL1>(&a1, &ps.cw2[kind], dst_rank, dst_file);

    let src_rank = src / 8;
    let src_file = src % 8;
    let s2_src = ps.cb2[6][src]
               + fwd_z88P_KKP_11::<HL1>(&a1, &ps.cw2[6], src_rank, src_file);

    let s2 = s2_src + s2_dst;

    // Error

    let diff = compress(score - s2) - compress(hyp);
    let dE_ds2 = diff * d_compress(score - s2) * -1.0;

    ds.cb2[kind][dst] += dE_ds2;
    ds.cb2[6   ][src] += dE_ds2;
    back_z88P_KKP_11::<HL1>(&a1, &mut ds.cw2[kind], dE_ds2, dst_rank, dst_file);
    back_z88P_KKP_11::<HL1>(&a1, &mut ds.cw2[6],    dE_ds2, src_rank, src_file);

    prop_z88P_KKP_11::<HL1>(&mut dE_da1, &ps.cw2[kind], dE_ds2, dst_rank, dst_file);
    prop_z88P_KKP_11::<HL1>(&mut dE_da1, &ps.cw2[6],    dE_ds2, src_rank, src_file);

    let diff = diff as f64;
    let error = diff * diff;
    sum_error += error;
  }

  // Backward

  let mut dE_dga1 = [0.0; HL1];
  mul_88N_N_z88N_prop_N::<HL1>(&ca1, &mut dE_dga1, &dE_da1);

  let mut dE_dca1 = [[0.0; 64]; HL1];
  mul_88N_N_z88N_prop_88N::<HL1>(&mut dE_dca1, &ga1, &dE_da1);

  let mut dE_dgs1 = [0.0; HL1];
  prop_sigm_N_N::<HL1>(&mut dE_dgs1, &dE_dga1, &gs1);
  back_bias_N::<HL1>(&mut ds.gb1, &dE_dgs1);
  back_P_Q::<HL1, HL1>(&gi1, &mut ds.gw1, &dE_dgs1);
  let mut dE_dgi1 = [0.0; HL1];
  prop_P_Q::<HL1, HL1>(&mut dE_dgi1, &ps.gw1, &dE_dgs1);
  back_pool_88N_N::<HL1>(&mut dE_dca1, &dE_dgi1);

  let mut dE_dcs1 = [[0.0; 64]; HL1];
  prop_relu_88N_88N::<HL1>(&mut dE_dcs1, &dE_dca1, &cs1);

  back_bias_88N::<HL1>(&mut ds.cb1, &dE_dcs1);
  back_88P_11PQ_88Q::<INP, HL1>(&inp, &mut ds.cw1, &dE_dcs1);

  return sum_error;
}

fn stats(c : &PolicyNetwork, d : &PolicyNetwork) -> f64 {
  let c_ary = c as *const PolicyNetwork as *const f32;
  let d_ary = d as *const PolicyNetwork as *const f32;
  let mut num_nonzero = 0;
  let mut num_flip = 0;
  for idx in 0..NUM_PARAMS {
    let c_val = unsafe { *c_ary.offset(idx as isize) };
    let d_val = unsafe { *d_ary.offset(idx as isize) };
    if c_val == 0.0 { continue; }
    if d_val == 0.0 { continue; }
    num_nonzero += 1;
    if c_val.is_sign_negative() != d_val.is_sign_negative() { num_flip += 1; }
  }
  if num_nonzero == 0 { return 0.0; }
  return num_flip as f64 / num_nonzero as f64;
}

const rng_seed : u64 = 0;

const alpha   : f32 = 1.0 / 512.0;
const beta    : f32 = 1.0 - 1.0 / 8.0;
const gamma   : f32 = 1.0 - 1.0 / 512.0;
const epsilon : f32 = 1.0 / 1048576.0;

const num_threads          : usize =    4;  //   80
const initial_batch_size   : usize =  512;  //  640
const batch_size_increment : usize =   64;  //   80
const maximum_batch_size   : usize = 2048;  // 2560

static mut DATASET_POSNS : Vec<(MiniState, u32, u32)> = Vec::new();
static mut DATASET_MOVES : Vec<(u16, i16)>            = Vec::new();
static mut PARAMS : PolicyNetwork                     = PolicyNetwork::zero();
static mut DERIVS : Vec<PolicyNetwork>                = Vec::new();

static mut NUM_MOVES : Vec<AtomicU32> = Vec::new();

static mut SUPERVISOR : Option<std::thread::Thread> = None;
static mut NUM_ACTIVE : AtomicUsize = AtomicUsize::new(0);

pub fn train_policy(path : &str) -> std::io::Result<()>
{
  static_assert!((initial_batch_size   / num_threads) * num_threads == initial_batch_size  );
  static_assert!((batch_size_increment / num_threads) * num_threads == batch_size_increment);

  init_rand(rng_seed);

  unsafe { PARAMS.perturb(); }
  let mut ds = PolicyNetwork::zero();
  let mut ms = PolicyNetwork::zero();
  let mut vs = PolicyNetwork::zero();
  unsafe {
    NUM_MOVES.clear();
    for _ in 0..num_threads { NUM_MOVES.push(AtomicU32::new(0)); }
    DERIVS.clear();
    for _ in 0..num_threads { DERIVS.push(PolicyNetwork::zero()); }
  }

  let mut reader = BufReader::new(File::open(path)?);
  eprint!("Loading...\x1B[K\r");
  let mut buffer = String::new();
  loop {
    let sz = reader.read_line(&mut buffer)?;
    if sz == 0 { break; }

    let mut record = buffer.split_ascii_whitespace();
    let posn = record.next().expect("empty line");
    assert!(posn.len() == 40, "incomplete: {}", buffer);
    let ary = <&[u8; 40]>::try_from(posn.as_bytes()).unwrap();
    let mini = MiniState::from_quick(ary);

    let ofs = unsafe { DATASET_MOVES.len() } as u32;
    let mut num_moves = 0;

    for mv in record {
      assert!(mv.len() == 6, "incomplete: {}", buffer);
      let ary = <&[u8; 6]>::try_from(mv.as_bytes()).unwrap();
      use crate::quick::dec;

      let u0 = dec(ary[0]) as u16;
      let u1 = dec(ary[1]) as u16;
      let u2 = dec(ary[2]) as u16;
      assert!(u0 < 64 && u1 < 64 && u2 < 16);
      let h = (u2 << 12 | u1 << 6 | u0) as i16;
      let k = dec(ary[3]) as u16;
      let s = dec(ary[4]) as u16;
      let d = dec(ary[5]) as u16;
      assert!(k < 6 && s < 64 && d < 64);
      let m = k << 12 | s << 6 | d;
      unsafe { DATASET_MOVES.push((m, h)); }
      num_moves += 1;
    }

    assert!(num_moves > 0, "empty move list: {}", buffer);
    unsafe { DATASET_POSNS.push((mini, ofs, num_moves as u32)); }

    buffer.clear();
  }
  let epoch_size = unsafe { DATASET_POSNS.len() };

  let mut cmpltd_batches = 0;

  let mut prev_ds = PolicyNetwork::zero();
  let mut ewma_flip = 0.0;

  let mut batch_size = initial_batch_size;
  let mut decay_alpha = alpha;

  unsafe { SUPERVISOR = Some(std::thread::current()); }
  for epoch in 0.. {
    eprint!("Permuting...\x1B[K\r");
    for x in 0..epoch_size {
      let k = u32::rand() as usize % (epoch_size - x);
      unsafe { DATASET_POSNS.swap(x, x+k); }
    }

    // batch_size = std::cmp::min(
    //   batch_size, (epoch_size / num_threads) * num_threads
    // );
    let batch_size_per_thread = batch_size / num_threads;
    let batch_per_epoch = epoch_size / batch_size;

    unsafe { NUM_ACTIVE.store(num_threads, Ordering::Release); }
    let mut workers = Vec::new();
    for id in 0..num_threads {
      workers.push(
        std::thread::spawn(move || -> f64 {
          let ds = unsafe { &mut DERIVS[id] };
          let mut total_error = 0.0;
          let mut index = id;
          for _ in 0..batch_per_epoch {
            ds.reset();
            let ps = unsafe { & *std::ptr::addr_of!(PARAMS) };
            let mut total_num = 0;
            for _ in 0..batch_size_per_thread {
              let (mini, ofs, num) = unsafe { &DATASET_POSNS[index] };
              total_error += fwdback(ps, ds, mini, ofs, num);
              total_num += num;
              index += num_threads;
            }
            unsafe {
              NUM_MOVES[id].store(total_num, Ordering::Release);
              let num_active = NUM_ACTIVE.fetch_sub(1, Ordering::AcqRel);
              if num_active == 1 { SUPERVISOR.as_ref().unwrap().unpark(); }
            }
            loop {
              std::thread::park();
              unsafe { if NUM_ACTIVE.load(Ordering::Acquire) > 0 { break; } }
            }
          }
          total_error
        })
      );
    }

    let mut moves_per_epoch = 0;

    for batch_num in 0..batch_per_epoch {
      loop {
        std::thread::park();
        unsafe { if NUM_ACTIVE.load(Ordering::Acquire) == 0 { break; } }
      }

      ds.reset();
      let mut num_moves = 0;
      unsafe {
        for id in 0..num_threads { ds.add(&DERIVS[id]); }
        for id in 0..num_threads { num_moves += NUM_MOVES[id].load(Ordering::Acquire); }
      }
      moves_per_epoch += num_moves as usize;
      ds.scale((num_moves as f32).recip());

      let bx0 = beta.powi(cmpltd_batches);
      let bx1 = bx0 * beta;
      let d_bias = (1.0 - bx1).recip();
      let c_bias = (1.0 - bx0) * d_bias;

      ms.scale(              beta     * c_bias);
      ms.add_scaled(&ds, (1.0 - beta) * d_bias);

      let gx0 = gamma.powi(cmpltd_batches);
      let gx1 = gx0 * gamma;
      let d_bias = (1.0 - gx1).recip();
      let c_bias = (1.0 - gx0) * d_bias;

      vs.scale(                     gamma     * c_bias);
      vs.add_square_scaled(&ds, (1.0 - gamma) * d_bias);

      let adj_alpha;
      if cmpltd_batches < 2048 {
        let t = cmpltd_batches as f32 * (1.0 / 2048.0);
        let u = t * ((t - 1.0) * 4.0).exp2();
        adj_alpha = decay_alpha * u;
      }
      else {
        adj_alpha = decay_alpha;
      }

      unsafe { PARAMS.update(&ms, &vs, adj_alpha); }

      unsafe {
        NUM_ACTIVE.store(num_threads, Ordering::Release);
        for handle in &workers { handle.thread().unpark(); }
      }

      let frac_flip = stats(&ds, &prev_ds);
      ewma_flip = ewma_flip * (1023.0 / 1024.0) + frac_flip * (1.0 / 1024.0);
      prev_ds = ds.clone();

      let progress = batch_num as f32 * 100.0 / batch_per_epoch as f32;
      eprint!(
        "{epoch:3}  {progress:5.2}%    {ewma_flip:5.3}  {batch_size:5}  {batch_per_epoch:5}  {num_moves:5}\x1B[K\r"
      );

      cmpltd_batches += 1;
    }

    let mut total_error = 0.0;
    for handle in workers { total_error += handle.join().unwrap(); }

    unsafe {
      PARAMS.save(&format!("/tmp/epoch-{:03}.kdnn", epoch))?;
      PARAMS.save_image(&format!("/tmp/1-epoch-{:03}", epoch), 1)?;
      PARAMS.save_image(&format!("/tmp/2-epoch-{:03}", epoch), 2)?;
    }

    let training_error = total_error / moves_per_epoch as f64;
    eprintln!(
      "{epoch:3}, {training_error:8.6}, {ewma_flip:5.3}, {batch_size:5}, {batch_per_epoch:5}\x1B[K\r"
    );

    batch_size += batch_size_increment;
    if batch_size > maximum_batch_size || ewma_flip > 0.666_666_666_666_666 {
      batch_size = initial_batch_size;
      decay_alpha *= 0.5;
    }
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
