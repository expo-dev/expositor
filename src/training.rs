use crate::color::*;
use crate::misc::*;
use crate::nnue::*;
use crate::rand::*;
use crate::state::*;
use crate::util::*;

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufRead, BufWriter, Write, Seek};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[derive(Clone)]
pub struct MiniState {
  pub boards : [u64; 16],
  pub turn   : Color,
}

impl MiniState {
  pub fn from(state : &State) -> Self
  {
    return MiniState { boards: state.boards, turn: state.turn };
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// It appears that LLVM is now able to autovectorize large portions of this code (perhaps b/c
//   of the alignment of TrainingNetwork structs forced by the alignment of the Network struct).
//   I'd be quite surprised if LLVM parallelized the ReLUs, ReLU derivatives, and reductions
//   (horizontal sums); I expect not, but I'm satisified with the output as-is for the time
//   being (since this isn't part of the engine proper) and so I'll leave it be.
// Besides those operations, I vaguely wonder – with me and my attempts to cajole rustc out of
//   the way – whether LLVM is doing a better job of it than I would.

#[derive(Clone, PartialEq)]
#[allow(non_snake_case)]
pub struct TrainingNetwork {
  pub params : Network,
  turn   : Color,
  inputs :  [f32; Np],
  s1     :  [f32; N1],  //
  s2     :  [f32; N2],  // input sums
  s3     :   f32,       //
  dE_ds1 :  [f32; N1],    //
  dE_ds2 :  [f32; N2],    // ∂E∕∂s
  dE_ds3 :   f32,         //
  dE_dw1 : [[f32; Np]; B1],  //
  dE_dw2 : [[f32; N1]; N2],  // ∂E∕∂w
  dE_dw3 :  [f32; N2],       //
  dE_db1 :  [f32; B1],    //
  dE_db2 :  [f32; N2],    // ∂E∕∂b
  dE_db3 :   f32,         //
  m_gw1  : [[f32; Np]; B1],  //
  m_gw2  : [[f32; N1]; N2],  // ∂E∕∂w exponentially-weighted mean
  m_gw3  :  [f32; N2],       //
  m_gb1  :  [f32; B1],    //
  m_gb2  :  [f32; N2],    // ∂E∕∂b exponentially-weighted mean
  m_gb3  :   f32,         //
  v_gw1  : [[f32; Np]; B1],  //
  v_gw2  : [[f32; N1]; N2],  // ∂E∕∂w exponentially-weighted variance
  v_gw3  :  [f32; N2],       //
  v_gb1  :  [f32; B1],    //
  v_gb2  :  [f32; N2],    // ∂E∕∂b exponentially-weighted variance
  v_gb3  :   f32,         //
}

impl TrainingNetwork {
  pub const fn zero() -> Self
  {
    return Self {
      params: Network::zero(),
      turn:   Color::White,
      inputs:  [0.0; Np],
      s1:      [0.0; N1],
      s2:      [0.0; N2],
      s3:       0.0,
      dE_ds1:  [0.0; N1],
      dE_ds2:  [0.0; N2],
      dE_ds3:   0.0,
      dE_dw1: [[0.0; Np]; B1],
      dE_dw2: [[0.0; N1]; N2],
      dE_dw3:  [0.0; N2],
      dE_db1:  [0.0; B1],
      dE_db2:  [0.0; N2],
      dE_db3:   0.0,
      m_gw1:  [[0.0; Np]; B1],
      m_gw2:  [[0.0; N1]; N2],
      m_gw3:   [0.0; N2],
      m_gb1:   [0.0; B1],
      m_gb2:   [0.0; N2],
      m_gb3:    0.0,
      v_gw1:  [[0.0; Np]; B1],
      v_gw2:  [[0.0; N1]; N2],
      v_gw3:   [0.0; N2],
      v_gb1:   [0.0; B1],
      v_gb2:   [0.0; N2],
      v_gb3:    0.0,
    };
  }

  pub fn new() -> Self
  {
    let mut training_network = Self::zero();
    training_network.params.perturb(1.0);
    return training_network;
  }

  pub fn forward(&mut self, state : &MiniState) -> f32
  {
    // Since we're unable to benefit from incremental updates, we don't need to actually keep
    //   white and black in a fixed order. So, to make things simpler, we swap the inputs rather
    //   than first layer banks. Now inputs #0 to #383 are for the side to move and inputs #384
    //   to #767 are for the waiting side.
    self.turn = state.turn;

    for n in 0..B1 { self.s1[  n ] = self.params.b1[n]; }
    for n in 0..B1 { self.s1[B1+n] = self.params.b1[n]; }

    self.inputs = [0.0; Np];
    for black in [false, true] {
      for piece in 0..6 {
        let mut sources = state.boards[(black as usize)*8 + piece];
        while sources != 0 {
          let src = sources.trailing_zeros() as usize;
          let ofs = piece*64 + if black { vmirror(src) } else { src };
          let upper = self.turn.as_bool() ^ black;
          let x        = if upper { Bp } else {  0 } + ofs;
          let x_mirror = if upper {  0 } else { Bp } + ofs;
          self.inputs[x] = 1.0;
          for n in 0..B1 { self.s1[  n ] += self.params.w1[x       ][n]; }
          for n in 0..B1 { self.s1[B1+n] += self.params.w1[x_mirror][n]; }
          sources &= sources - 1;
        }
      }
    }

    let mut a1 : [MaybeUninit<f32>; N1] = MaybeUninit::uninit_array();
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
    for n in 0..N1 { a1[n].write(relu(self.s1[n])); }
    let a1 = unsafe { std::mem::transmute::<_,&mut [f32; N1]>(&mut a1) };
    // Alternatively,
    //   let a1 = unsafe { MaybeUninit::array_assume_init(a1) };
    // or maybe slice_assume_init_mut or slice_assume_init_ref.

    for n in 0..N2 { self.s2[n] = self.params.b2[n]; }
    for n in 0..N2 { for x in 0..N1 { self.s2[n] += a1[x] * self.params.w2[n][x]; } }

    let mut a2 : [MaybeUninit<f32>; N2] = MaybeUninit::uninit_array();
    for n in 0..N2 { a2[n].write(relu(self.s2[n])); }
    let a2 = unsafe { std::mem::transmute::<_,&mut [f32; N2]>(&mut a2) };

    self.s3 = self.params.b3;
    for x in 0..N2 { self.s3 += a2[x] * self.params.w3[x]; }

    return self.s3;
  }

  #[allow(non_snake_case)]
  fn backward(&mut self, dE_ds3 : f32)
  {
    self.dE_ds3 = dE_ds3;

    self.dE_db3 += self.dE_ds3;
    for x in 0..N2 {
      self.dE_dw3[x] += self.dE_ds3 *   relu(self.s2[x]);
      self.dE_ds2[x]  = self.dE_ds3 * d_relu(self.s2[x]) * self.params.w3[x];
    }

    self.dE_ds1 = [0.0; N1];
    for n in 0..N2 {
      self.dE_db2[n] += self.dE_ds2[n];
      for x in 0..N1 {
        self.dE_dw2[n][x] += self.dE_ds2[n] *   relu(self.s1[x]);
        self.dE_ds1[x]    += self.dE_ds2[n] * d_relu(self.s1[x]) * self.params.w2[n][x];
      }
    }

    for n in  0..B1 { self.dE_db1[n]    += self.dE_ds1[n]; }
    for n in B1..N1 { self.dE_db1[n-B1] += self.dE_ds1[n]; }
    for n in 0..B1 {
      for x in 0..Np {
        self.dE_dw1[n][x] += self.dE_ds1[n] * self.inputs[x];
      }
    }
    for n in 0..B1 {
      for x in 0..Bp { self.dE_dw1[n][Bp+x] += self.dE_ds1[B1+n] * self.inputs[  x ]; }
      for x in 0..Bp { self.dE_dw1[n][  x ] += self.dE_ds1[B1+n] * self.inputs[Bp+x]; }
    }
  }

  pub fn reset_gradient(&mut self)
  {
    self.dE_dw1 = [[0.0; Np]; B1];
    self.dE_dw2 = [[0.0; N1]; N2];
    self.dE_dw3 =  [0.0; N2];
    self.dE_db1 =  [0.0; B1];
    self.dE_db2 =  [0.0; N2];
    self.dE_db3 =   0.0;
  }

  pub fn add_gradient(&mut self, other : &Self)
  {
    for n in 0..B1 { for x in 0..Np { self.dE_dw1[n][x] += other.dE_dw1[n][x]; } }
    for n in 0..N2 { for x in 0..N1 { self.dE_dw2[n][x] += other.dE_dw2[n][x]; } }
    for x in 0..N2 { self.dE_dw3[x] += other.dE_dw3[x]; }
    for n in 0..B1 { self.dE_db1[n] += other.dE_db1[n]; }
    for n in 0..N2 { self.dE_db2[n] += other.dE_db2[n]; }
    self.dE_db3 += other.dE_db3;
  }

  pub fn scale_gradient(&mut self, scale : f32)
  {
    for n in 0..B1 { for x in 0..Np { self.dE_dw1[n][x] *= scale; } }
    for n in 0..N2 { for x in 0..N1 { self.dE_dw2[n][x] *= scale; } }
    for x in 0..N2 { self.dE_dw3[x] *= scale; }
    for n in 0..B1 { self.dE_db1[n] *= scale; }
    for n in 0..N2 { self.dE_db2[n] *= scale; }
    self.dE_db3 *= scale;
  }

  pub fn train(&mut self, state : &MiniState, target : f32) -> f32
  {
    #![allow(non_snake_case)]
    let prediction = self.forward(state);
    let dE_ds3 =
      (harsh_compress(prediction) - harsh_compress(target)) * d_harsh_compress(prediction);
    self.backward(dE_ds3);
    return prediction;
  }

  pub fn update(&mut self, alpha : f32, beta : f32, gamma : f32)
  {
    let epsilon = 1.0e-6;

    // Update means
    self.m_gb3 = self.m_gb3*beta + self.dE_db3*(1.0-beta);
    for x in 0..N2 {
      self.m_gw3[x] = self.m_gw3[x]*beta + self.dE_dw3[x]*(1.0-beta);
    }
    for n in 0..N2 {
      self.m_gb2[n] = self.m_gb2[n]*beta + self.dE_db2[n]*(1.0-beta);
      for x in 0..N1 {
        self.m_gw2[n][x] = self.m_gw2[n][x]*beta + self.dE_dw2[n][x]*(1.0-beta);
      }
    }
    for n in 0..B1 {
      self.m_gb1[n] = self.m_gb1[n]*beta + self.dE_db1[n]*(1.0-beta);
      for x in 0..Np {
        self.m_gw1[n][x] = self.m_gw1[n][x]*beta + self.dE_dw1[n][x]*(1.0-beta);
      }
    }

    // Update variances
    self.v_gb3 = self.v_gb3*gamma + self.dE_db3*self.dE_db3*(1.0-gamma);
    for x in 0..N2 {
      self.v_gw3[x] = self.v_gw3[x]*gamma + self.dE_dw3[x]*self.dE_dw3[x]*(1.0-gamma);
    }
    for n in 0..N2 {
      self.v_gb2[n] = self.v_gb2[n]*gamma + self.dE_db2[n]*self.dE_db2[n]*(1.0-gamma);
      for x in 0..N1 {
        self.v_gw2[n][x] = self.v_gw2[n][x]*gamma + self.dE_dw2[n][x]*self.dE_dw2[n][x]*(1.0-gamma);
      }
    }
    for n in 0..B1 {
      self.v_gb1[n] = self.v_gb1[n]*gamma + self.dE_db1[n]*self.dE_db1[n]*(1.0-gamma);
      for x in 0..Np {
        self.v_gw1[n][x] = self.v_gw1[n][x]*gamma + self.dE_dw1[n][x]*self.dE_dw1[n][x]*(1.0-gamma);
      }
    }

    // Update weights
    self.params.b3 -= alpha * self.m_gb3
      //            * (0.01 + self.m_gb3.abs()).sqrt()
                    / (self.v_gb3 + epsilon).sqrt();
    for x in 0..N2 {
      self.params.w3[x] -= alpha * self.m_gw3[x]
      //                 * (0.01 + self.m_gw3[x].abs()).sqrt()
                         / (self.v_gw3[x] + epsilon).sqrt();
    }
    for n in 0..N2 {
      self.params.b2[n] -= alpha * self.m_gb2[n]
      //                 * (0.01 + self.m_gb2[n].abs()).sqrt()
                         / (self.v_gb2[n] + epsilon).sqrt();
      for x in 0..N1 {
        self.params.w2[n][x] -= alpha * self.m_gw2[n][x]
      //                      * (0.01 + self.m_gw2[n][x].abs()).sqrt()
                              / (self.v_gw2[n][x] + epsilon).sqrt();
      }
    }
    for n in 0..B1 {
      self.params.b1[n] -= alpha * self.m_gb1[n]
      //                 * (0.01 + self.m_gb1[n].abs()).sqrt()
                         / (self.v_gb1[n] + epsilon).sqrt();
      for x in 0..Np {
        self.params.w1[x][n] -= alpha * self.m_gw1[n][x]
      //                      * (0.01 + self.m_gw1[n][x].abs()).sqrt()
                              / (self.v_gw1[n][x] + epsilon).sqrt();
      }
    }
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

static mut ACTIVE_WORKERS  : AtomicUsize                 = AtomicUsize::new(0);
static mut WORKER_NETWORKS : Vec<TrainingNetwork>        = Vec::new();
static mut TRAINER         : Option<std::thread::Thread> = None;
static mut TRAINING_SET    : Vec<(f32, MiniState)>       = Vec::new();

pub fn train_nnue(
  inp_path    : &str,
  out_prefix  : &str,
  alpha       : f32,
  beta        : f32,
  gamma       : f32,
  batch_size  : usize,
  num_threads : usize,
  rng_seed    : u64,
) -> std::io::Result<TrainingNetwork>
{
  const MAX_EPOCH_SIZE : usize = 67108864;
  // The batch_size needs to be a multiple of num_threads
  assert!(batch_size % num_threads == 0, "batch size must be a multiple of the thread count");
  let batch_size_per_thread = batch_size / num_threads;

  eprint!("  Preparing...\x1B[K\r");
  let num_lines = BufReader::new(File::open(inp_path)?).lines().count();
  let epoch_size = std::cmp::min(num_lines, MAX_EPOCH_SIZE);
  // The size of the training set needs to be a multiple of the batch_size
  let batches_per_epoch = (epoch_size / 16 * 15) / batch_size;
  let training_size = batches_per_epoch * batch_size;
  let testing_size = epoch_size - training_size;

  let mut history = BufWriter::new(
    OpenOptions::new().append(true).create(true).open(&format!("{}history", out_prefix))?
  );
  writeln!(history, "Input path:      {}", inp_path)?;
  writeln!(history, "Input size:      {}", num_lines)?;
  writeln!(history, "Epoch size:      {}", epoch_size)?;
  writeln!(history, "Training size:   {}", training_size)?;
  writeln!(history, "Validation size: {}", testing_size)?;
  writeln!(history, "Alpha:           {}", alpha)?;
  writeln!(history, "Beta:            {}", beta)?;
  writeln!(history, "Gamma:           {}", gamma)?;
  writeln!(history, "Batch size:      {}", batch_size)?;
  writeln!(history, "Thread count:    {}", num_threads)?;
  writeln!(history, "RNG seed:        {}", rng_seed)?;
  writeln!(history, "N1×N2:           {}×{}", N1, N2)?;
  history.flush()?;

  set_rand(rng_seed);
  let mut reader = BufReader::new(File::open(inp_path)?);
  let mut network = TrainingNetwork::new();

  network.params.save_fst_image(&format!("{}inter-fst-000", out_prefix))?;
  network.params.save_snd_image(&format!("{}inter-snd-000", out_prefix))?;

  eprintln!("---------------------------");
  eprintln!("Epoch  Training  Validation");
  eprintln!("---------------------------");

  writeln!(history, "-------------------------")?;
  writeln!(history, " Ep  Training  Validation")?;
  history.flush()?;

  let mut i_counter : usize = 0;
  let mut interim   : usize = 0;

  for epoch in 0.. {
    let mut num_processed : usize = 0;
    let mut e_counter     : usize = 0;

    // Read in an epoch's worth of positions
    eprint!("  Loading...\x1B[K\r");
    unsafe { TRAINING_SET.clear(); }
    for _ in 0..epoch_size {
      let mut buffer = String::new();
      if reader.read_line(&mut buffer)? == 0 {
        // If we reach EOF, start from the beginning
        reader.rewind()?;
        if reader.read_line(&mut buffer)? == 0 {
          panic!("training file appears to be empty");
        }
      }
      let mut fields = buffer.split_ascii_whitespace();
      let score = fields.next().unwrap().parse::<f32>().unwrap();
      let state = MiniState::from(&State::from_fen_fields(&mut fields).unwrap());
      let score = match state.turn { Color::White => score, Color::Black => -score };
      unsafe { TRAINING_SET.push((score, state)); }
      buffer.clear();
    }

    // Shuffle the positions
    eprint!("  Permuting...\x1B[K\r");
    for x in 0..epoch_size {
      let k = rand() as usize % (epoch_size - x);
      unsafe { TRAINING_SET.swap(x, x+k); }
    }

    // Partition the positions into training and validation sets
    let testing_set = unsafe { TRAINING_SET.split_off(training_size) };
    assert!(testing_size == testing_set.len(), "incoherent size of testing set");

    // Create a copy of the training network for each worker
    unsafe {
      WORKER_NETWORKS.clear();
      for _ in 0..num_threads { WORKER_NETWORKS.push(network.clone()); }
    }

    // Spawn the workers
    unsafe {
      TRAINER = Some(std::thread::current());
      ACTIVE_WORKERS.store(num_threads, Ordering::Release);
    }
    let mut workers = Vec::new();
    for id in 0..num_threads {
      workers.push(std::thread::spawn(move || -> f32 {
        let mut total_error = 0.0;
        let mut index = id;
        let network = unsafe { &mut WORKER_NETWORKS[id] };
        for _ in 0..batches_per_epoch {
          network.reset_gradient();
          for _ in 0..batch_size_per_thread {
            let (score, state) = unsafe { &TRAINING_SET[index] };
            let prediction = network.train(state, *score);
            let error = harsh_compress(prediction) - harsh_compress(*score);
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
      }));
    }

    // Manage each batch
    eprint!("\r{:3}     0.0%\x1B[K\r", epoch);
    for _ in 0..batches_per_epoch {
      // Wait for the workers to finish a batch
      loop {
        std::thread::park();
        unsafe { if ACTIVE_WORKERS.load(Ordering::Acquire) == 0 { break; } }
      }

      // Sum the gradients from the worker networks
      network.reset_gradient();
      unsafe { for n in 0..num_threads { network.add_gradient(&WORKER_NETWORKS[n]); } }
      network.scale_gradient(1.0 / batch_size as f32);

      // Update the weights
      network.update(alpha, beta, gamma);
      num_processed += batch_size;

      // Copy the weights back to the worker networks
      unsafe { for n in 0..num_threads { WORKER_NETWORKS[n].params = network.params.clone(); } }

      // Wake up the workers
      unsafe {
        ACTIVE_WORKERS.store(num_threads, Ordering::Release);
        for handle in &workers { handle.thread().unpark(); }
      }

      // Update the progress indication
      e_counter += batch_size;
      if e_counter >= 62500 {
        eprint!("{:3}    {:4.1}%\r", epoch, num_processed as f32 * 100.0 / (epoch_size as f32));
        while e_counter >= 62500 { e_counter -= 62500; }
      }

      // Save interim visualizations
      i_counter += batch_size;
      if i_counter >= 250000 {
        interim += 1;
        if interim < 200 {
          network.params.save_fst_image(&format!("{}inter-fst-{:03}", out_prefix, interim))?;
          network.params.save_snd_image(&format!("{}inter-snd-{:03}", out_prefix, interim))?;
        }
        while i_counter >= 250000 { i_counter -= 250000; }
      }
    }

    // We've finished an epoch!
    let mut total_error = 0.0;
    for handle in workers { total_error += handle.join().unwrap(); }

    // Save the network
    network.params.save(&format!("{}epoch-{:03}.nnue", out_prefix, epoch))?;

    // Save epoch visualizations
    network.params.save_fst_image(&format!("{}epoch-fst-{:03}", out_prefix, epoch))?;
    network.params.save_snd_image(&format!("{}epoch-snd-{:03}", out_prefix, epoch))?;

    // Calculate training and validation error
    let training_error = total_error / training_size as f32;
    eprint!("{:3}    {:8.6}", epoch, training_error);

    let mut total_error = 0.0;
    for (score, state) in testing_set {
      let prediction = network.forward(&state);
      let error = harsh_compress(prediction) - harsh_compress(score);
      total_error += error * error;
    }
    let testing_error = total_error / testing_size as f32;
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
