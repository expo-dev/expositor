use crate::color::{WB, Color::*};
use crate::misc::vmirror;
use crate::nnue::*;
use crate::rand::{set_rand, rand};
use crate::state::MiniState;
use crate::util::{harsh_compress, d_harsh_compress};

use std::convert::TryFrom;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufRead, BufWriter, Write, Seek};
use std::mem::MaybeUninit;
use std::ops::{AddAssign, MulAssign};
use std::sync::atomic::{AtomicUsize, Ordering};

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

static mut LEAK : f32 = 0.03125;

fn   adj_relu(x : f32) -> f32 { return x.max(x * unsafe{LEAK}); }
fn d_adj_relu(x : f32) -> f32 { return if x < 0.0 { unsafe{LEAK} } else { 1.0 }; }

impl AddAssign<&NetworkHead> for NetworkHead {
  fn add_assign(&mut self, rhs : &NetworkHead)
  {
    unsafe {
      let  self_ary = std::mem::transmute::<_, *mut f32>(self);
      let other_ary = std::mem::transmute::<_, *const f32>(rhs);
      for idx in 0..HEAD_SIZE {
        let idx = idx as isize;
        let sum = *self_ary.offset(idx) + *other_ary.offset(idx);
        std::ptr::write(self_ary.offset(idx), sum);
      }
    }
  }
}

impl MulAssign<f32> for NetworkHead {
  fn mul_assign(&mut self, rhs : f32)
  {
    unsafe {
      let self_ary = std::mem::transmute::<_, *mut f32>(self);
      for idx in 0..HEAD_SIZE {
        let idx = idx as isize;
        let product = *self_ary.offset(idx) * rhs;
        std::ptr::write(self_ary.offset(idx), product);
      }
    }
  }
}

impl AddAssign<&NetworkBody> for NetworkBody {
  fn add_assign(&mut self, rhs : &NetworkBody)
  {
    unsafe {
      let  self_ary = std::mem::transmute::<_, *mut f32>(self);
      let other_ary = std::mem::transmute::<_, *const f32>(rhs);
      for idx in 0..BODY_SIZE {
        let idx = idx as isize;
        let sum = *self_ary.offset(idx) + *other_ary.offset(idx);
        std::ptr::write(self_ary.offset(idx), sum);
      }
    }
  }
}

impl MulAssign<f32> for NetworkBody {
  fn mul_assign(&mut self, rhs : f32)
  {
    unsafe {
      let self_ary = std::mem::transmute::<_, *mut f32>(self);
      for idx in 0..BODY_SIZE {
        let idx = idx as isize;
        let product = *self_ary.offset(idx) * rhs;
        std::ptr::write(self_ary.offset(idx), product);
      }
    }
  }
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
pub struct TrainingNetworkBody {
  g : NetworkBody,  // gradient
  m : NetworkBody,  // exponentially-weighted mean
  v : NetworkBody,  // exponentially-weighted variance
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
        for n in 0..N1 { self.s1[SideToMove ][n] += move_rgn_params.w1[rel_to_move][ofs][n]; }
        for n in 0..N1 { self.s1[SideWaiting][n] += wait_rgn_params.w1[rel_to_wait][ofs][n]; }
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
      for n in 0..N1 { self.s1[SideToMove ][n] += move_rgn_params.w1[rel_to_move][ofs][n]; }
      for n in 0..N1 { self.s1[SideWaiting][n] += wait_rgn_params.w1[rel_to_wait][ofs][n]; }
    }

    let mut a1 : [[MaybeUninit<f32>; N1]; 2] =
      [MaybeUninit::uninit_array(), MaybeUninit::uninit_array()];
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
    for side in 0..2 { for n in 0..N1 { a1[side][n].write(adj_relu(self.s1[side][n])); } }
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

    let mut a2 : [MaybeUninit<f32>; N2] = MaybeUninit::uninit_array();
    for n in 0..N2 { a2[n].write(adj_relu(self.s2[n])); }
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
    let hd_params = &self.params.hd[hdx];
    let head = &mut self.hd[hdx];

    self.dE_ds3 = dE_ds3;

    head.g.b3 += self.dE_ds3;
    for x in 0..N2 {
      let s = self.s2[x];
      head.g.w3[x]  += self.dE_ds3 *   adj_relu(s);
      self.dE_ds2[x] = self.dE_ds3 * d_adj_relu(s) * hd_params.w3[x];
    }

    self.dE_ds1 = [[0.0; N1]; 2];
    for n in 0..N2 {
      head.g.b2[n] += self.dE_ds2[n];
      for side in 0..2 {
        for x in 0..N1 {
          let s = self.s1[side][x];
          head.g.w2[n][side][x] += self.dE_ds2[n] *   adj_relu(s);
          self.dE_ds1[side][x]  += self.dE_ds2[n] * d_adj_relu(s) * hd_params.w2[n][side][x];
        }
      }
    }

    let move_region = &mut self.rn[move_rdx];
    for n in 0..N1 { move_region.g.b1[n] += self.dE_ds1[SideToMove][n]; }
    for side in 0..2 {
      for x in 0..Np {
        for n in 0..N1 {
          move_region.g.w1[side][x][n] += self.dE_ds1[SideToMove][n] * self.inputs[side][x];
        }
      }
    }

    let wait_region = &mut self.rn[wait_rdx];
    for n in 0..N1 { wait_region.g.b1[n] += self.dE_ds1[SideWaiting][n]; }
    for side in 0..2 {
      for x in 0..Np {
        for n in 0..N1 {
          wait_region.g.w1[side][x][n] += self.dE_ds1[SideWaiting][n] * self.inputs[side^1][x];
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
    let error = harsh_compress(prediction) - harsh_compress(target);
    let dE_ds3 = error * d_harsh_compress(prediction);
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

  fn update(&mut self, alpha : f32, beta : f32, gamma : f32)
  {
    let epsilon = 1.0 / 4096.0;
    unsafe {
      use std::mem::transmute;
      for r in 0..REGIONS {
        self.rn[r].m *= beta;
        let cmpl_beta = 1.0 - beta;
        let means = transmute::<_, *mut f32>(std::ptr::addr_of_mut!(self.rn[r].m));
        let  grad = transmute::<_, *const f32>(std::ptr::addr_of!(self.rn[r].g));
        for idx in 0..BODY_SIZE {
          let idx = idx as isize;
          let upd_m = *means.offset(idx) + (*grad.offset(idx)) * cmpl_beta;
          std::ptr::write(means.offset(idx), upd_m);
        }

        self.rn[r].v *= gamma;
        let cmpl_gamma = 1.0 - gamma;
        let vars = transmute::<_, *mut f32>(std::ptr::addr_of_mut!(self.rn[r].v));
        let grad = transmute::<_, *const f32>(std::ptr::addr_of!(self.rn[r].g));
        for idx in 0..BODY_SIZE {
          let idx = idx as isize;
          let g = *grad.offset(idx);
          let upd_v = *vars.offset(idx) + g * g * cmpl_gamma;
          std::ptr::write(vars.offset(idx), upd_v);
        }

        let params = transmute::<_, *mut f32>(std::ptr::addr_of_mut!(self.params.rn[r]));
        let  means = transmute::<_, *const f32>(std::ptr::addr_of!(self.rn[r].m));
        let   vars = transmute::<_, *const f32>(std::ptr::addr_of!(self.rn[r].v));
        for idx in 0..BODY_SIZE {
          let idx = idx as isize;
          let upd_w = (*params.offset(idx))
                    - alpha * (*means.offset(idx)) / ((*vars.offset(idx)).sqrt() + epsilon);
          std::ptr::write(params.offset(idx), upd_w);
        }
      }
      for h in 0..HEADS {
        self.hd[h].m *= beta;
        let cmpl_beta = 1.0 - beta;
        let means = transmute::<_, *mut f32>(std::ptr::addr_of_mut!(self.hd[h].m));
        let  grad = transmute::<_, *const f32>(std::ptr::addr_of!(self.hd[h].g));
        for idx in 0..HEAD_SIZE {
          let idx = idx as isize;
          let upd_m = *means.offset(idx) + (*grad.offset(idx)) * cmpl_beta;
          std::ptr::write(means.offset(idx), upd_m);
        }

        self.hd[h].v *= gamma;
        let cmpl_gamma = 1.0 - gamma;
        let vars = transmute::<_, *mut f32>(std::ptr::addr_of_mut!(self.hd[h].v));
        let grad = transmute::<_, *const f32>(std::ptr::addr_of!(self.hd[h].g));
        for idx in 0..HEAD_SIZE {
          let idx = idx as isize;
          let g = *grad.offset(idx);
          let upd_v = *vars.offset(idx) + g * g * cmpl_gamma;
          std::ptr::write(vars.offset(idx), upd_v);
        }

        let params = transmute::<_, *mut f32>(std::ptr::addr_of_mut!(self.params.hd[h]));
        let  means = transmute::<_, *const f32>(std::ptr::addr_of!(self.hd[h].m));
        let   vars = transmute::<_, *const f32>(std::ptr::addr_of!(self.hd[h].v));
        for idx in 0..HEAD_SIZE {
          let idx = idx as isize;
          let upd_w = (*params.offset(idx))
                    - alpha * (*means.offset(idx)) / ((*vars.offset(idx)).sqrt() + epsilon);
          std::ptr::write(params.offset(idx), upd_w);
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
  finetune    : bool,
  inp_path    : &str,
  out_prefix  : &str,
  alpha       : f32,
  beta        : f32,
  gamma       : f32,
  batch_size  : usize,
  num_epochs  : usize,
  num_threads : usize,
  rng_seed    : u64,
) -> std::io::Result<TrainingNetwork>
{
  assert!(num_epochs > 2, "insufficient number of epochs");

  const MAX_EPOCH_SIZE : usize = 1_073_741_824;

  let batch_size_per_thread = batch_size / num_threads;

  // The batch_size needs to be a multiple of num_threads
  assert!(
    batch_size_per_thread * num_threads == batch_size,
    "batch size must be a multiple of the thread count"
  );

  eprint!("  Preparing...\x1B[K\r");
  let num_lines = BufReader::new(File::open(inp_path)?).lines().count();
  let epoch_size = std::cmp::min(num_lines, MAX_EPOCH_SIZE);

  // The size of the training set needs to be a multiple of the batch_size
  let batches_per_epoch = (epoch_size / 32 * 31) / batch_size;
  let training_size = batches_per_epoch * batch_size;
  let testing_size = epoch_size - training_size;

  // Adjust alpha depending on the batch size
  let nominal_alpha = alpha;
  let alpha = alpha * (batch_size as f32).sqrt() / 128.0;

  let mut history = BufWriter::new(
    OpenOptions::new().append(true).create(true).open(&format!("{}history", out_prefix))?
  );
  writeln!(history, "N1×N2:           {}×{}", N1, N2)?;
  writeln!(history, "Input path:      {}", inp_path)?;
  writeln!(history, "Input size:      {}", num_lines)?;
  writeln!(history, "Epoch size:      {}", epoch_size)?;
  writeln!(history, "Training size:   {}", training_size)?;
  writeln!(history, "Validation size: {}", testing_size)?;
  writeln!(history, "Batch size:      {}", batch_size)?;
  writeln!(history, "Nominal alpha:   {}", nominal_alpha)?;
  writeln!(history, "Alpha:           {}", alpha)?;
  writeln!(history, "Beta:            {}", beta)?;
  writeln!(history, "Gamma:           {}", gamma)?;
  writeln!(history, "Thread count:    {}", num_threads)?;
  writeln!(history, "RNG seed:        {}", rng_seed)?;
  writeln!(history, "Epochs:          {}", num_epochs)?;
  history.flush()?;

  set_rand(rng_seed);
  let mut reader = BufReader::new(File::open(inp_path)?);
  let mut network = TrainingNetwork::zero();
  if finetune {
    network.params = unsafe { NETWORK.clone() };
  }
  else {
    network.params.perturb_thd();
    network.params.perturb_fst_snd(0);
  }

  network.params.save_image(&format!("{}inter-000", out_prefix), Image::FstUnif)?;

  eprintln!("---------------------------");
  eprintln!("Input size:      {:10}", num_lines);
  eprintln!("Epoch size:      {:10}", epoch_size);
  eprintln!("Training size:   {:10}", training_size);
  eprintln!("Testing size:    {:10}", testing_size);
  eprintln!("Batch per epoch: {:10}", batches_per_epoch);

  eprintln!("---------------------------");
  eprintln!("Epoch  Training  Validation");
  eprintln!("---------------------------");

  writeln!(history, "-------------------------")?;
  writeln!(history, " Ep  Training  Validation")?;
  history.flush()?;

  let mut i_counter : usize = 0;
  let mut interim   : usize = 0;

  let mut batch_num : usize = 0;

  let mut testing_set = Vec::new();

  for epoch in 0..num_epochs {
    let mut num_processed : usize = 0;

    if epoch == 0 || epoch_size != num_lines {
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
        let k = rand() as usize % (epoch_size - x);
        unsafe { TRAINING_SET.swap(x, x+k); }
      }

      // Partition the positions into training and validation sets
      testing_set = unsafe { TRAINING_SET.split_off(training_size) };
      assert!(testing_size == testing_set.len(), "incoherent size of testing set");
    }
    else {
      eprint!("  Permuting...\x1B[K\r");
      for x in 0..training_size {
        let k = rand() as usize % (training_size - x);
        unsafe { TRAINING_SET.swap(x, x+k); }
      }
    }

    // Create a copy of the training network for each worker
    unsafe {
      WORKER_NETWORKS.clear();
      for _ in 0..num_threads { WORKER_NETWORKS.push(network.clone()); }
    }

    // Set the ReLU leakage
    //   We start at 1/16 on epoch 0 and decrease it to 1/4096 on num_epochs-2;
    //   on num_epochs-1 we finally set the leakage to zero for one last round.
    //   If we are fine-tuning, we start at 1/64 on epoch 0, instead.
    if epoch == num_epochs - 1 {
      unsafe { LEAK = 0.0; }
    }
    else {
      let progress = (epoch as f32) / ((num_epochs - 2) as f32);
      let x = if finetune { 6.0 + 6.0 * progress } else { 4.0 + 8.0 * progress };
      unsafe { LEAK = (-x).exp2(); }
    }

    // Spawn the workers
    unsafe {
      TRAINER = Some(std::thread::current());
      ACTIVE_WORKERS.store(num_threads, Ordering::Release);
    }
    let mut workers = Vec::new();
    for id in 0..num_threads {
      workers.push(
        std::thread::Builder::new()
        .stack_size(2_097_152)
        .spawn(move || -> f64 {
          let mut total_error = 0.0;
          let mut index = id;
          let network = unsafe { &mut WORKER_NETWORKS[id] };
          for _ in 0..batches_per_epoch {
            network.reset_gradient();
            for _ in 0..batch_size_per_thread {
              let mini = unsafe { &TRAINING_SET[index] };
              let score = mini.score as f32 / 100.0;
              let score = match mini.turn() { White => score, Black => -score };
              let error = network.train(mini, score) as f64;
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
        })
        .unwrap()
      );
    }

    // Manage each batch
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
      network.update(alpha, beta, gamma);
      num_processed += batch_size;
      batch_num += 1;

      // Introduce the next neuron
      if !finetune {
        if batch_num < N1 { network.params.perturb_fst_snd(batch_num); }
      }

      // Copy the weights back to the worker networks
      unsafe { for n in 0..num_threads { WORKER_NETWORKS[n].params = network.params.clone(); } }

      // Wake up the workers
      unsafe {
        ACTIVE_WORKERS.store(num_threads, Ordering::Release);
        for handle in &workers { handle.thread().unpark(); }
      }

      // Update the progress indication
      eprint!("{:3}    {:5.2}%\r", epoch, num_processed as f32 * 100.0 / (training_size as f32));

      // Save interim visualizations
      i_counter += batch_size;
      if i_counter >= 1_000_000 {
        interim += 1;
        if interim < 32 {
          network.params.save_image(
            &format!("{}inter-{:03}", out_prefix, interim), Image::FstUnif
          )?;
        }
        while i_counter >= 1_000_000 { i_counter -= 1_000_000; }
      }
    }

    // We've finished an epoch!
    let mut total_error = 0.0;
    for handle in workers { total_error += handle.join().unwrap(); }

    // Save the network
    network.params.save(&format!("{}epoch-{:03}.nnue", out_prefix, epoch))?;

    // Save epoch visualizations
    network.params.save_image(&format!("{}epoch-{:03}", out_prefix, epoch), Image::FstUnif)?;

    // Calculate training and validation error
    let training_error = total_error / training_size as f64;
    eprint!("{:3}    {:8.6}", epoch, training_error);

    let mut total_error = 0.0;
    for mini in testing_set.iter() {
      let turn = mini.turn();
      let wk_idx =         mini.positions[White][0] as usize ;
      let bk_idx = vmirror(mini.positions[Black][0] as usize);
      let move_rdx = king_region(match turn { White => wk_idx, Black => bk_idx });
      let wait_rdx = king_region(match turn { White => bk_idx, Black => wk_idx });
      let hdx = mini.head_index();

      let prediction = network.forward(&mini, move_rdx, wait_rdx, hdx);
      let score = mini.score as f32 / 100.0;
      let score = match mini.turn() { White => score, Black => -score };
      let error = (harsh_compress(prediction) - harsh_compress(score)) as f64;
      total_error += error * error;
    }
    let testing_error = total_error / testing_size as f64;
    eprintln!("    {:8.6}", testing_error);

    // Append this epoch to the log file
    let mut history = BufWriter::new(
      OpenOptions::new().append(true).create(true).open(&format!("{}history", out_prefix))?
    );
    writeln!(history, "{:3}, {:8.6}, {:8.6}", epoch, training_error, testing_error)?;
    history.flush()?;
  }
  return Ok(network);
}
