use crate::color::{WB, Color::*};
use crate::misc::vmirror;
use crate::nnue::*;
use crate::rand::{Rand, init_rand};
use crate::state::MiniState;

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufRead, BufWriter, Write, Seek};
use std::mem::MaybeUninit;
use std::ops::{AddAssign, MulAssign};
use std::sync::atomic::{AtomicUsize, Ordering};

use std::simd::Simd;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub fn expectation(x : f32) -> f32
{
  // score in pawns to expected outcome (from 0 to 1)

  return 0.31 / (1.0 + (-2.74 * x).exp()) + 0.69 / (1.0 + (-0.716 * x).exp())
}

pub fn d_expectation(x : f32) -> f32
{
  let x = x.abs();
  let e1 = (-2.74  * x).exp();
  let e2 = (-0.716 * x).exp();
  let p1 = 1.0 + e1;
  let p2 = 1.0 + e2;
  return 0.849 * e1 / (p1 * p1) + 0.494 * e2 / (p2 * p2);
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

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
    for color in WB {
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

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn relu(x : f32) -> f32
{
  // return x.max(0.0).min(1.0);
  return x.max(0.0);
}

fn d_relu(x : f32) -> f32
{
  // return if x < 0.0 || 1.0 < x { 0.0 } else { 1.0 };
  return if x < 0.0 { 0.0 } else { 1.0 };
}

struct Guard<const N : usize> { }
impl <const N : usize> Guard<N> {
  const CHECK : () = assert!(N == (N / 8) * 8);
  fn assert() { let _ = Self::CHECK; }
}

macro_rules! assert_lanes {
  ($n:expr) => { Guard::<$n>::assert(); }
}

macro_rules! simd_load {
  ($a:expr, $b:expr) => { Simd::<f32, 8>::from_slice(&$a[$b .. $b+8]) }
}

macro_rules! simd_store {
  ($a:expr, $b:expr, $c:expr) => { $c.copy_to_slice(&mut $a[$b .. $b+8]) }
}

macro_rules! simd_incr {
  ($a:expr, $b:expr, $c:expr) => { simd_store!($a, $b, simd_load!($a, $b) + $c) }
}

fn simd_incr<const N : usize>(augend : &mut [f32; N], addend : &[f32; N])
{
  assert_lanes!(N);
  let n = N / 8;
  for ofs in 0..n { simd_incr!(augend, ofs*8, simd_load!(addend, ofs*8)); }
}

impl AddAssign<&NetworkBody> for NetworkBody {
  fn add_assign(&mut self, rhs : &NetworkBody)
  {
    unsafe {
      let self_ary = self as *mut NetworkBody as *mut f32;
      let  rhs_ary = rhs as *const NetworkBody as *const f32;
      for idx in 0..BODY_SIZE {
        let sum = *self_ary.add(idx) + *rhs_ary.add(idx);
        std::ptr::write(self_ary.add(idx), sum);
      }
    }
  }
}

impl MulAssign<f32> for NetworkBody {
  fn mul_assign(&mut self, rhs : f32)
  {
    unsafe {
      let self_ary = self as *mut NetworkBody as *mut f32;
      for idx in 0..BODY_SIZE {
        let product = *self_ary.add(idx) * rhs;
        std::ptr::write(self_ary.add(idx), product);
      }
    }
  }
}

impl AddAssign<&NetworkHead> for NetworkHead {
  fn add_assign(&mut self, rhs : &NetworkHead)
  {
    unsafe {
      let self_ary = self as *mut NetworkHead as *mut f32;
      let  rhs_ary = rhs as *const NetworkHead as *const f32;
      for idx in 0..HEAD_SIZE {
        let sum = *self_ary.add(idx) + *rhs_ary.add(idx);
        std::ptr::write(self_ary.add(idx), sum);
      }
    }
  }
}

impl MulAssign<f32> for NetworkHead {
  fn mul_assign(&mut self, rhs : f32)
  {
    unsafe {
      let self_ary = self as *mut NetworkHead as *mut f32;
      for idx in 0..HEAD_SIZE {
        let product = *self_ary.add(idx) * rhs;
        std::ptr::write(self_ary.add(idx), product);
      }
    }
  }
}

// fn num_flips(ntwk : &TrainingNetwork, prev : &[NetworkBody; REGIONS]) -> usize {
//   const SZ : usize = std::mem::size_of::<NetworkBody>() / std::mem::size_of::<f32>();
//   let mut flips = 0;
//   for r in 0..REGIONS {
//     let ntwk_ary = unsafe { std::mem::transmute::<_, &[f32; SZ]>(&ntwk.rn[r].g) };
//     let prev_ary = unsafe { std::mem::transmute::<_, &[f32; SZ]>(&prev[r]) };
//     for n in 0..SZ {
//       if ntwk_ary[n].is_sign_negative() != prev_ary[n].is_sign_negative() { flips += 1; }
//     }
//   }
//   return flips;
// }

#[derive(Clone, PartialEq)]
#[repr(align(32))]
#[allow(non_snake_case)]
pub struct TrainingNetworkBody {
  g : NetworkBody,  // gradient
  m : NetworkBody,  // exponentially-weighted mean
  v : NetworkBody,  // exponentially-weighted variance
}

#[derive(Clone, PartialEq)]
#[repr(align(32))]
#[allow(non_snake_case)]
pub struct TrainingNetworkHead {
  g : NetworkHead,  // gradient
  m : NetworkHead,  // exponentially-weighted mean
  v : NetworkHead,  // exponentially-weighted variance
}

#[derive(Clone, PartialEq)]
#[repr(align(32))]
#[allow(non_snake_case)]
pub struct TrainingNetwork {
  pub params : Network,
  inputs : [[f32; Np]; 2],
      s1 : [[f32; N1]; 2],  // sums
  dE_ds1 : [[f32; N1]; 2],  // ∂E∕∂s
      s2 : [f32; N2],       // ...
  dE_ds2 : [f32; N2],       // ...
      s3 :  f32,            // ...
  dE_ds3 :  f32,            // ...
  rn     : [TrainingNetworkBody; REGIONS],
  hd     : [TrainingNetworkHead; HEADS],
}

impl TrainingNetwork {
  const fn zero() -> Self
  {
    const SZ : usize = std::mem::size_of::<TrainingNetwork>();
    union Empty {
      ary : [u8; SZ],
      net : std::mem::ManuallyDrop<TrainingNetwork>
    }
    const ZERO : Empty = Empty { ary: [0; SZ] };
    return std::mem::ManuallyDrop::<TrainingNetwork>::into_inner(unsafe { ZERO.net });
  }

  fn forward(
    &mut self,
    mini     : &MiniState,
    move_rdx : usize,
    wait_rdx : usize,
    hdx      : usize
  ) -> f32
  {
    // Since we're unable to benefit from incremental updates, we don't need
    //   to actually keep white and black in a fixed order. Let's swap the
    //   inputs rather than first layer banks.

    // let (leak, ofs) = load_relu_params();

    let move_rgn_params = &self.params.rn[move_rdx];
    let wait_rgn_params = &self.params.rn[wait_rdx];

    for n in 0..N1 { self.s1[SideToMove ][n] = move_rgn_params.b1[n]; }
    for n in 0..N1 { self.s1[SideWaiting][n] = wait_rgn_params.b1[n]; }

    self.inputs = [[0.0; Np]; 2];
    for color in WB {
      for slot in 0..16 {
        let posn = mini.positions[color][slot];
        if posn < 0 { continue; }
        let kind = MiniState::KIND[slot];
        let src = posn as usize;
        let ofs = (kind as usize)*64
                + match color { White => src, Black => vmirror(src) };
        let side        = if color == mini.turn() { SideToMove } else { SideWaiting };
        let rel_to_move = if color == mini.turn() {  SameSide  } else {  OppoSide   };
        let rel_to_wait = rel_to_move ^ 1;
        self.inputs[side][ofs] = 1.0;
        simd_incr::<N1>(&mut self.s1[SideToMove ], &move_rgn_params.w1[rel_to_move][ofs]);
        simd_incr::<N1>(&mut self.s1[SideWaiting], &wait_rgn_params.w1[rel_to_wait][ofs]);
      }
    }
    for vdx in 0..2 {
      let (piece, posn) = mini.variable[vdx];
      if posn < 0 { continue; }
      let color = piece.color();
      let kind = piece.kind();
      let src = posn as usize;
      let ofs = (kind as usize)*64
              + match color { White => src, Black => vmirror(src) };
      let side        = if color == mini.turn() { SideToMove } else { SideWaiting };
      let rel_to_move = if color == mini.turn() {  SameSide  } else {  OppoSide   };
      let rel_to_wait = rel_to_move ^ 1;
      self.inputs[side][ofs] = 1.0;
      simd_incr::<N1>(&mut self.s1[SideToMove ], &move_rgn_params.w1[rel_to_move][ofs]);
      simd_incr::<N1>(&mut self.s1[SideWaiting], &wait_rgn_params.w1[rel_to_wait][ofs]);
    }

    let mut a1 : [[MaybeUninit<f32>; N1]; 2] =
      [[MaybeUninit::uninit(); N1], [MaybeUninit::uninit(); N1]];
    // Alternatively,
    //   let mut a1 : [MaybeUninit<f32>; N1] = unsafe { MaybeUninit::uninit().assume_init() };
    // which follows examples given in the documentation, or something like
    //   let mut a1 = [MaybeUninit::uninit(); N1];
    // which seems to work in practice but is probably incorrect:
    //   > In a future Rust version the uninit_array method may become unnecessary when Rust
    //   > allows inline const expressions. Then
    //   >   let mut buf: [MaybeUninit<u8>; 32] = MaybeUninit::uninit_array();
    //   > in the example could use
    //   >   let mut buf = [const { MaybeUninit::<u8>::uninit() }; 32];
    for side in 0..2 { for n in 0..N1 { a1[side][n].write(relu(self.s1[side][n])); } }
    // Alternatively,
    //   let a1 = unsafe { MaybeUninit::array_assume_init(a1) };
    // or maybe slice_assume_init_mut or slice_assume_init_ref.
    let a1 = unsafe { std::mem::transmute::<_,&mut [[f32; N1]; 2]>(&mut a1) };

    let hd_params = &self.params.hd[hdx];

    for n in 0..N2 {
      self.s2[n] = hd_params.b2[n];
      for x in 0..N1 { self.s2[n] += a1[SideToMove ][x] * hd_params.w2[n][SideToMove ][x]; }
      for x in 0..N1 { self.s2[n] += a1[SideWaiting][x] * hd_params.w2[n][SideWaiting][x]; }
    }

    let mut a2 : [MaybeUninit<f32>; N2] = [MaybeUninit::uninit(); N2];
    for n in 0..N2 { a2[n].write(relu(self.s2[n])); }
    let a2 = unsafe { std::mem::transmute::<_,&mut [f32; N2]>(&mut a2) };

    self.s3 = hd_params.b3;
    for x in 0..N2 { self.s3 += a2[x] * hd_params.w3[x]; }

    return self.s3;
  }

  #[allow(non_snake_case)]
  fn backward(
    &mut self,
    move_rdx : usize,
    wait_rdx : usize,
    hdx      : usize,
    dE_ds3   : f32
  )
  {
    // let (leak, ofs) = load_relu_params();

    let hd_params = &self.params.hd[hdx];
    let head = &mut self.hd[hdx];

    self.dE_ds3 = dE_ds3;

    head.g.b3 += self.dE_ds3;
    for x in 0..N2 {
      let s = self.s2[x];
      head.g.w3[x]  += self.dE_ds3 *   relu(s);
      self.dE_ds2[x] = self.dE_ds3 * d_relu(s) * hd_params.w3[x];
    }

    self.dE_ds1 = [[0.0; N1]; 2];
    for n in 0..N2 {
      head.g.b2[n] += self.dE_ds2[n];
      for side in 0..2 {
        for x in 0..N1 {
          let s = self.s1[side][x];
          head.g.w2[n][side][x] += self.dE_ds2[n] *   relu(s);
          self.dE_ds1[side][x]  += self.dE_ds2[n] * d_relu(s) * hd_params.w2[n][side][x];
        }
      }
    }

    assert_lanes!(N1);

    let move_region = &mut self.rn[move_rdx];
    for n in 0..(N1/8) {
      simd_incr!(move_region.g.b1, n*8, simd_load!(self.dE_ds1[SideToMove], n*8));
    }
    for side in 0..2 {
      for x in 0..Np {
        let inp = Simd::<f32, 8>::splat(self.inputs[side][x]);
        for n in 0..(N1/8) {
          simd_incr!(move_region.g.w1[side][x], n*8,
            simd_load!(self.dE_ds1[SideToMove], n*8) * inp);
        }
      }
    }

    let wait_region = &mut self.rn[wait_rdx];
    for n in 0..(N1/8) {
      simd_incr!(wait_region.g.b1, n*8, simd_load!(self.dE_ds1[SideWaiting], n*8));
    }
    for side in 0..2 {
      for x in 0..Np {
        let inp = Simd::<f32, 8>::splat(self.inputs[side^1][x]);
        for n in 0..(N1/8) {
          simd_incr!(wait_region.g.w1[side][x], n*8,
            simd_load!(self.dE_ds1[SideWaiting], n*8) * inp);
        }
      }
    }
  }

  fn train(&mut self, mini : &MiniState, target : f32) -> f32
  {
    #![allow(non_snake_case)]
    let turn = mini.turn();
    let wk_idx =         mini.positions[White][0] as usize ;
    let bk_idx = vmirror(mini.positions[Black][0] as usize);
    let move_rdx = king_region(match turn { White => wk_idx, Black => bk_idx });
    let wait_rdx = king_region(match turn { White => bk_idx, Black => wk_idx });
    let hdx = mini.head_index();

    let prediction = self.forward(mini, move_rdx, wait_rdx, hdx);
    // ↓↓↓ DEBUG ↓↓↓
    // /* disable leakiness to run this test */
    // let state = crate::state::State::from(mini);
    // let cmp = self.params.evaluate(&state, hdx);
    // if (prediction - cmp).abs() > 0.001 {
    //   eprintln!("{}", state.to_fen());
    //   eprintln!("{} {} {}", move_rdx, wait_rdx, hdx);
    //   eprintln!("{} ~ {}", prediction, cmp);
    //   panic!("mismatch");
    // }
    // ↑↑↑ DEBUG ↑↑↑

    let error = (expectation(prediction) - target) * 4.0;
    let dE_ds3 = error * 4.0 * d_expectation(prediction);
    self.backward(move_rdx, wait_rdx, hdx, dE_ds3);
    return error;
  }

  fn reset_gradient(&mut self)
  {
    for r in 0..REGIONS {
      let region = &mut self.rn[r];
      region.g.w1 = [[[0.0; N1]; Np]; 2];
      region.g.b1 =   [0.0; N1];
    }
    for h in 0..HEADS {
      let head = &mut self.hd[h];
      head.g.w2 = [[[0.0; N1]; 2]; N2];
      head.g.w3 =   [0.0; N2];
      head.g.b2 =   [0.0; N2];
      head.g.b3 =    0.0;
    }
  }

  fn add_gradient(&mut self, other : &Self)
  {
    for r in 0..REGIONS { self.rn[r].g += &other.rn[r].g; }
    for h in 0..HEADS   { self.hd[h].g += &other.hd[h].g; }
  }

  fn scale_gradient(&mut self, batch_size : usize)
  {
    // Although I have some theories, for reasons I don't understand, the
    //   trainer performs worse with proportional gradient normalization.
    let scale = 1.0 / batch_size as f32;
    for r in 0..REGIONS { self.rn[r].g *= scale; }
    for h in 0..HEADS   { self.hd[h].g *= scale; }
  }

  fn initialize_common_moments(&mut self)
  {
    for h in 0..HEADS {
      let head = &mut self.hd[h];
      for n2 in 0..N2 {
        let b2 = head.g.b2[n2];
        head.m.b2[n2] = b2;
        head.v.b2[n2] = b2 * b2;
        let w3 = head.g.w3[n2];
        head.m.w3[n2] = w3;
        head.v.w3[n2] = w3 * w3;
      }
      let b3 = head.g.b3;
      head.m.b3 = b3;
      head.v.b3 = b3 * b3;
    }
  }

  fn initialize_moments(&mut self, n1 : usize)
  {
    for r in 0..REGIONS {
      let region = &mut self.rn[r];
      for side in 0..2 {
        for x in 0..Np {
          let w1 = region.g.w1[side][x][n1];
          region.m.w1[side][x][n1] = w1;
          region.v.w1[side][x][n1] = w1 * w1;
        }
      }
      let b1 = region.g.b1[n1];
      region.m.b1[n1] = b1;
      region.v.b1[n1] = b1 * b1;
    }
    for h in 0..HEADS {
      let head = &mut self.hd[h];
      for side in 0..2 {
        for n2 in 0..N2 {
          let w2 = head.g.w2[n2][side][n1];
          head.m.w2[n2][side][n1] = w2;
          head.v.w2[n2][side][n1] = w2 * w2;
        }
      }
    }
  }

  fn update(&mut self, alpha : f32, beta : f32, gamma : f32, epsilon : f32)
  {
    unsafe {
      use std::mem::transmute;
      for r in 0..REGIONS {
        self.rn[r].m *= beta;
        let cmpl_beta = 1.0 - beta;
        let means = transmute::<_, *mut f32>(std::ptr::addr_of_mut!(self.rn[r].m));
        let  grad = transmute::<_, *const f32>(std::ptr::addr_of!(self.rn[r].g));
        for idx in 0..BODY_SIZE {
          let upd_m = *means.add(idx) + (*grad.add(idx)) * cmpl_beta;
          std::ptr::write(means.add(idx), upd_m);
        }

        self.rn[r].v *= gamma;
        let cmpl_gamma = 1.0 - gamma;
        let vars = transmute::<_, *mut f32>(std::ptr::addr_of_mut!(self.rn[r].v));
        let grad = transmute::<_, *const f32>(std::ptr::addr_of!(self.rn[r].g));
        for idx in 0..BODY_SIZE {
          let g = *grad.add(idx);
          let upd_v = *vars.add(idx) + g * g * cmpl_gamma;
          std::ptr::write(vars.add(idx), upd_v);
        }

        let params = transmute::<_, *mut f32>(std::ptr::addr_of_mut!(self.params.rn[r]));
        let  means = transmute::<_, *const f32>(std::ptr::addr_of!(self.rn[r].m));
        let   vars = transmute::<_, *const f32>(std::ptr::addr_of!(self.rn[r].v));
        for idx in 0..BODY_SIZE {
          let upd_w = (*params.add(idx))
                    - alpha * (*means.add(idx)) / ((*vars.add(idx)).sqrt() + epsilon);
          std::ptr::write(params.add(idx), upd_w);
        }
      }
      for h in 0..HEADS {
        self.hd[h].m *= beta;
        let cmpl_beta = 1.0 - beta;
        let means = transmute::<_, *mut f32>(std::ptr::addr_of_mut!(self.hd[h].m));
        let  grad = transmute::<_, *const f32>(std::ptr::addr_of!(self.hd[h].g));
        for idx in 0..HEAD_SIZE {
          let upd_m = *means.add(idx) + (*grad.add(idx)) * cmpl_beta;
          std::ptr::write(means.add(idx), upd_m);
        }

        self.hd[h].v *= gamma;
        let cmpl_gamma = 1.0 - gamma;
        let vars = transmute::<_, *mut f32>(std::ptr::addr_of_mut!(self.hd[h].v));
        let grad = transmute::<_, *const f32>(std::ptr::addr_of!(self.hd[h].g));
        for idx in 0..HEAD_SIZE {
          let g = *grad.add(idx);
          let upd_v = *vars.add(idx) + g * g * cmpl_gamma;
          std::ptr::write(vars.add(idx), upd_v);
        }

        let params = transmute::<_, *mut f32>(std::ptr::addr_of_mut!(self.params.hd[h]));
        let  means = transmute::<_, *const f32>(std::ptr::addr_of!(self.hd[h].m));
        let   vars = transmute::<_, *const f32>(std::ptr::addr_of!(self.hd[h].v));
        for idx in 0..HEAD_SIZE {
          let upd_w = (*params.add(idx))
                    - alpha * (*means.add(idx)) / ((*vars.add(idx)).sqrt() + epsilon);
          std::ptr::write(params.add(idx), upd_w);
        }
      }
    }
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

static mut ACTIVE_WORKERS  : AtomicUsize                 = AtomicUsize::new(0);
static mut WORKER_NETWORKS : Vec<TrainingNetwork>        = Vec::new();
static mut TRAINER         : Option<std::thread::Thread> = None;
static mut TRAINING_SET    : Vec<MiniState>              = Vec::new();

pub fn train_nnue(
  finetune      : bool,
  inp_path      : &str,
  out_prefix    : &str,
  alpha         : f32,
  beta          : f32,
  gamma         : f32,
  recip_epsilon : f32,
  batch_size    : usize,
  num_epochs    : usize,
  num_threads   : usize,
  rng_seed      : u64,
) -> std::io::Result<TrainingNetwork>
{
  assert!(num_epochs > 2, "insufficient number of epochs");

  const MAX_EPOCH_SIZE : usize = 134_217_728;

  let batch_size_per_thread = batch_size / num_threads;

  // The batch_size needs to be a multiple of num_threads
  assert!(
    batch_size_per_thread * num_threads == batch_size,
    "batch size must be a multiple of the thread count"
  );

  // eprint!("  Preparing...\x1B[K\r");
  // let num_lines = BufReader::with_capacity(2_097_152, File::open(inp_path)?).lines().count();
  // let epoch_size = std::cmp::min(num_lines, MAX_EPOCH_SIZE);
  let epoch_size = MAX_EPOCH_SIZE;

  // The size of the training set needs to be a multiple of the batch_size
  let batches_per_epoch = epoch_size / batch_size;
  let training_size = batches_per_epoch * batch_size;

  let epsilon = recip_epsilon.recip();

  let mut history = BufWriter::new(
    OpenOptions::new().append(true).create(true).open(format!("{}history", out_prefix))?
  );
  writeln!(history, "N1×N2:           {}×{}", N1, N2)?;
  writeln!(history, "Input path:      {}", inp_path)?;
//writeln!(history, "Input size:      {}", num_lines)?;
  writeln!(history, "Epoch size:      {}", epoch_size)?;
  writeln!(history, "Training size:   {}", training_size)?;
  writeln!(history, "Batch size:      {}", batch_size)?;
  writeln!(history, "Alpha:           {}", alpha)?;
  writeln!(history, "Beta:            {}", beta)?;
  writeln!(history, "Gamma:           {}", gamma)?;
  writeln!(history, "1/Epsilon:       {}", recip_epsilon)?;
  writeln!(history, "Thread count:    {}", num_threads)?;
  writeln!(history, "RNG seed:        {}", rng_seed)?;
  writeln!(history, "Epochs:          {}", num_epochs)?;
  history.flush()?;

  const ZERO_BODY : NetworkBody =
    NetworkBody {
      w1: [[[0.0; N1]; Np]; 2],
      b1:   [0.0; N1],
    };

  init_rand(rng_seed);
  let mut reader = BufReader::new(File::open(inp_path)?);
  let mut network = TrainingNetwork::zero();
  let mut prev = [ZERO_BODY; REGIONS];
  if finetune {
    network.params = unsafe { NETWORK.clone() };
  }
  else {
    network.params.perturb_thd();
    network.params.perturb_fst_snd(0);
  }

  network.params.save_image("/tmp/inter-00", true, -1)?;

  eprintln!("---------------------------");
//eprintln!("Input size:      {:10}", num_lines);
  eprintln!("Epoch size:      {:10}", epoch_size);
  eprintln!("Training size:   {:10}", training_size);
  eprintln!("Batch per epoch: {:10}", batches_per_epoch);

  eprintln!("---------------------------");
  eprintln!("Epoch  Training"            );
  eprintln!("---------------------------");

  writeln!(history, "-------------------------")?;
  writeln!(history, " Ep  Training"            )?;
  history.flush()?;

  let mut batch_num  : usize = 0;
  // let mut ewma_flips : f64 = 0.0;

  for epoch in 0..num_epochs {
    if epoch == 0 /* || epoch_size != num_lines */ {
      // Read in an epoch's worth of positions
      eprint!("  Loading...\x1B[K\r");
      unsafe { TRAINING_SET.clear(); }
      let mut buffer = String::new();
      for _ in 0..epoch_size {
        if reader.read_line(&mut buffer)? == 0 {
          // If we reach EOF, start from the beginning
          reader.rewind()?;
          if reader.read_line(&mut buffer)? == 0 {
            panic!("training file appears to be empty");
          }
        }
        if let Ok(ary) = <&[u8; 40]>::try_from(buffer.trim().as_bytes()) {
          unsafe { TRAINING_SET.push(MiniState::from_quick(ary)); }
        }
        else {
          panic!("incorrect length: {}", buffer);
        }
        buffer.clear();
      }
      // Shuffle the positions
      eprint!("  Permuting...\x1B[K\r");
      for x in 0..epoch_size {
        let k = u32::rand() as usize % (epoch_size - x);
        unsafe { TRAINING_SET.swap(x, x+k); }
      }
    }
    else {
      eprint!("  Permuting...\x1B[K\r");
      for x in 0..training_size {
        let k = u32::rand() as usize % (training_size - x);
        unsafe { TRAINING_SET.swap(x, x+k); }
      }
    }

    // Create a copy of the training network for each worker
    unsafe {
      WORKER_NETWORKS.clear();
      for _ in 0..num_threads { WORKER_NETWORKS.push(network.clone()); }
    }

    // Set the initial ReLU leakage
    // set_leak(batch_num, batches_per_epoch, num_epochs);

    // Spawn the workers
    unsafe {
      TRAINER = Some(std::thread::current());
      ACTIVE_WORKERS.store(num_threads, Ordering::Release);
    }
    let mut workers = Vec::new();
    for id in 0..num_threads {
      workers.push(
        std::thread::Builder::new()
        .name(format!("{id}"))
        .stack_size(2_097_152)
        .spawn(move || -> f64 {
          // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          let mut total_error = 0.0;
          let mut index = id;
          let network = unsafe { &mut WORKER_NETWORKS[id] };
          for _ in 0..batches_per_epoch {
            network.reset_gradient();
            for _ in 0..batch_size_per_thread {
              let mini = unsafe { &TRAINING_SET[index] };
              let score16 = mini.score.from(mini.turn()).as_i16();
              // assert!(!(score16 < -100_00), "{} {} {} {}", crate::state::State::from(mini).to_fen(), mini.to_quick(), mini.score, mini.outcome());
              // assert!(!(score16 >  100_00), "{} {} {} {}", crate::state::State::from(mini).to_fen(), mini.to_quick(), mini.score, mini.outcome());
              let score16 = std::cmp::max(score16, -16_00);
              let score16 = std::cmp::min(score16,  16_00);
              let score = score16 as f32 / 100.0;
              let mut target = expectation(score);

              use crate::score::{CoOutcome, PovOutcome};
              if mini.outcome() != CoOutcome::Unknown {
                let antiphase = mini.antiphase();
                let phase = 96 - std::cmp::min(96, antiphase);
                let x = 0.0625 + (phase as f32 / 96.0) * 0.25;  // 0.0625 to 0.3125
                let result = match mini.outcome().from(mini.turn()) {
                  PovOutcome::Unknown => panic!(),
                  PovOutcome::Loss    => 0.0,
                  PovOutcome::Draw    => 0.5,
                  PovOutcome::Win     => 1.0,
                };
                target = target * (1.0 - x) + result * x;
              }

              let error = network.train(mini, target) as f64;
              total_error += error * error;
              index += num_threads;
            }
            unsafe {
              let num_active = ACTIVE_WORKERS.fetch_sub(1, Ordering::AcqRel);
              if num_active == 1 { TRAINER.as_ref().unwrap().unpark(); }
            }
            loop {
              std::thread::park();
              unsafe { if ACTIVE_WORKERS.load(Ordering::Relaxed) > 0 { break; } }
            }
          }
          total_error
          // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        })
        .unwrap()
      );
    }

    // Manage each batch
    let mut num_processed : usize = 0;
    eprint!("\r{:3}     0.00%\x1B[K\r", epoch);
    for _ in 0..batches_per_epoch {
      // Wait for the workers to finish a batch
      loop {
        std::thread::park();
        unsafe { if ACTIVE_WORKERS.load(Ordering::Acquire) == 0 { break; } }
      }

      // Sum the gradients from the worker networks
      network.reset_gradient();
      unsafe { for n in 0..num_threads { network.add_gradient(&WORKER_NETWORKS[n]); } }
      network.scale_gradient(batch_size);

      // Update the weights
      if finetune {
        if batch_num == 0 { network.initialize_common_moments(); }
        if batch_num == 0 { for n in 0..N1 { network.initialize_moments(n); } }
      }
      else {
        if batch_num == 0 { network.initialize_common_moments(); }
        if batch_num < N1 { network.initialize_moments(batch_num); }
      }
      network.update(alpha, beta, gamma, epsilon);
      num_processed += batch_size;
      batch_num += 1;

      // Introduce the next neuron
      if !finetune {
        if batch_num < N1 { network.params.perturb_fst_snd(batch_num); }
      }

      // Copy the weights back to the worker networks
      unsafe { for n in 0..num_threads { WORKER_NETWORKS[n].params = network.params.clone(); } }

      // Set the ReLU leakage
      // set_leak(batch_num, batches_per_epoch, num_epochs);

      // Wake up the workers
      unsafe {
        ACTIVE_WORKERS.store(num_threads, Ordering::Release);
        for handle in &workers { handle.thread().unpark(); }
      }

      // Calculate the number of gradient sign flips
      // let num_flips = num_flips(&network, &prev);
      // const SZ : usize = std::mem::size_of::<NetworkBody>() / std::mem::size_of::<f32>();
      // let frac_flips = num_flips as f64 / (SZ * REGIONS) as f64;
      // ewma_flips = ewma_flips*0.875 + frac_flips*0.125;
      for r in 0..REGIONS { prev[r] = network.rn[r].g.clone(); }

      // Update the progress indication
      // eprint!(
      //   "{:3}    {:5.2}%   {:5.3} {:5.2}\x1B[K\r",
      //   epoch, num_processed as f32 * 100.0 / (training_size as f32), frac_flips, -alpha.log2()
      // );
      eprint!(
        "{:3}    {:5.2}%   {:5.2}\x1B[K\r",
        epoch, num_processed as f32 * 100.0 / (training_size as f32), -alpha.log2()
      );

      // Save interim visualizations
      let interim = batch_num / 64;
      if batch_num % 64 == 0 && interim <= 32 {
        network.params.save_image(&format!("/tmp/inter-{:02}", interim), true, -1)?;
      }
    }

    // We've finished an epoch!
    let mut total_error = 0.0;
    for handle in workers { total_error += handle.join().unwrap(); }

    // Save the network
    network.params.save(&format!("{}epoch-{:03}.nnue", out_prefix, epoch))?;

    // Save epoch visualizations
    network.params.save_image(&format!("{}epoch-{:03}", out_prefix, epoch), true, -1)?;

    // Calculate training error
    let training_error = total_error / training_size as f64;
    // eprintln!("{:3}    {:8.6} {:5.3}\x1B[K", epoch, training_error, ewma_flips);
    eprintln!("{:3}    {:8.6}\x1B[K", epoch, training_error);

    // Append this epoch to the log file
    let mut history = BufWriter::new(
      OpenOptions::new().append(true).create(true).open(&format!("{}history", out_prefix))?
    );
    // writeln!(history, "{:3}, {:8.6}, {:8.6}", epoch, training_error, ewma_flips)?;
    writeln!(history, "{:3}, {:8.6}", epoch, training_error)?;
    history.flush()?;
  }
  return Ok(network);
}
