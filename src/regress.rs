use crate::color::*;
use crate::import::*;
use crate::piece::*;
use crate::rand::*;
use crate::state::*;

fn load_dataset(path : &str) -> std::io::Result<Vec<[f32; 3]>>
{
  eprint!("Loading...\r");
  let mut game_count = 0usize;
  let mut dataset = Vec::new();
  for movelist in PGNReader::open(path)? {
    let movelist = movelist?;
    let game_length = movelist.len() as u16;
    if !crate::util::isatty(crate::util::STDOUT) {
      print!(" {},", game_length); if game_count % 16 == 15 { print!("\n"); }
    }
    let mut working = State::new();
    working.initialize_nnue();
    for mv in movelist {
      working.apply(&mv);
      let wq = working.boards[WHITE+QUEEN ].count_ones() as i16;
      let wr = working.boards[WHITE+ROOK  ].count_ones() as i16;
      let wb = working.boards[WHITE+BISHOP].count_ones() as i16;
      let wn = working.boards[WHITE+KNIGHT].count_ones() as i16;
      let wp = working.boards[WHITE+PAWN  ].count_ones() as i16;
      let bq = working.boards[BLACK+QUEEN ].count_ones() as i16;
      let br = working.boards[BLACK+ROOK  ].count_ones() as i16;
      let bb = working.boards[BLACK+BISHOP].count_ones() as i16;
      let bn = working.boards[BLACK+KNIGHT].count_ones() as i16;
      let bp = working.boards[BLACK+PAWN  ].count_ones() as i16;

      let white_pieces = wq + wr + wb + wn + wp;
      let black_pieces = bq + br + bb + bn + bp;
      let total_pieces = white_pieces + black_pieces;

      dataset.push([
        (game_length - working.ply) as f32,
        working.ply as f32,
        total_pieces as f32,
      ]);
    }
    game_count += 1;
  }
  eprintln!("{:9} games", game_count);
  eprintln!("{:9} positions", dataset.len());
  return Ok(dataset);
}

// fn lognorm_model(ply_played : u16) -> f32
// {
//   // Fit from human games
//   let p = ply_played as f32;
//   return 75.0 + (34777.0 - p*5116.0) / (p*p + p*22.0 + 4881.0);
// }

// fn lognorm_model(ply_played : u16) -> f32
// {
//   // Fit from a log-normal distribution with mean 100 and median 80
//   let p = ply_played as f32;
//   return 167.5 - (1148000.0 + p*42120.0) / (p*p + p*282.9 + 17010.0);
// }

// fn lognorm_model(ply_played : u16) -> f32
// {
//   // Fit from a log-normal distribution with mean 120 and median 100
//   let p = ply_played as f32;
//   return 122.1 - (22980.0 + p*13700.0) / (p*p + p*75.81 + 11440.0);
// }

// fn lognorm_model(ply_played : u16) -> f32
// {
//   // Fit from a log-normal distribution with mean 140 and median 120
//   let p = ply_played as f32;
//   return 107.4 + (297500.0 - p*8729.0) / (p*p + p*22.76 + 9085.0);
// }

fn error(
  dataset : &Vec<[f32; 3]>,
  coeffs  : &[f32; 3],
) -> (f32, usize)
{
  let mut err = 0.0;
  let mut neg = 0;
  for point in dataset {
    let prediction = coeffs[0]
                   + coeffs[1] * point[1] // ply
                   + coeffs[2] * point[2];// pieces
    let d = prediction - point[0];
    err += d*d;
    if prediction < 0.0 { neg += 1; }
  }
  return (err, neg);
}

fn mean(dataset : &Vec<[f32; 3]>) -> f32
{
  let mut sum = 0.0;
  for point in dataset { sum += point[0]; }
  return sum / dataset.len() as f32;
}

fn deter(dataset : &Vec<[f32; 3]>, err : f32) -> f32
{
  let m = mean(dataset);
  let mut sum = 0.0;
  for point in dataset {
    let d = point[0] - m;
    sum += d*d;
  }
  return 1.0 - err / sum;
}

fn neighbor(
  coeffs : &[f32; 3],
  scale  : f32
) -> [f32; 3]
{
  let mut next = coeffs.clone();
  let axes = [1, 1, 1, 1, 1, 2, 2, 3][rand() as usize % 8];
  for _ in 0..axes {
    let idx = rand() as usize % 3;
    let a = next[idx] + (symmetric_uniform() as f32) * scale;
    if 150.0 >= a && a >= -150.0 { next[idx] = a; }
  }
  return next;
}

pub fn fit_dataset(path : &str) -> std::io::Result<()>
{
  let dataset = load_dataset(path)?.into_iter()
                                   .filter(|p| 240.0 >= p[0]+p[1] && p[0]+p[1] >= 60.0)
                                   .collect();

  let mut best_coeffs = [0.0; 3];
  best_coeffs[0] = mean(&dataset);

  let mut best_error = error(&dataset, &best_coeffs).0;

  let mut reach : f32 = 0.0;
  let mut counter = 0;
  loop {
    if counter > 8192 {
      counter = 0;
      reach += 0.25;
      if reach > 1.0 { break; }
    }
    let next_coeffs = neighbor(&best_coeffs, 0.25 - reach.sqrt()*0.1875);
    let (next_error, neg_pred) = error(&dataset, &next_coeffs);
    if next_error < best_error {
      if reach > 0.0 { reach -= 0.0625; }
      counter = 0;
      best_coeffs = next_coeffs;
      best_error = next_error;
      let start_estimate = best_coeffs[0] + best_coeffs[2] * 30.0;
      eprint!(
        "{:9.3} {:+8.3} {:+.3} {:+.3} \x1B[2m{:6.2}\x1B[22m \x1B[94m{:.3}\x1B[39m",
        best_error / dataset.len() as f32,
        best_coeffs[0],
        best_coeffs[1],
        best_coeffs[2],
        start_estimate,
        deter(&dataset, best_error)
      );
      if neg_pred > 0 {
        eprint!(" \x1B[91m{:3.1}%\x1B[39m", neg_pred as f32 * 100.0 / dataset.len() as f32);
      }
      eprint!("\n");
    }
    counter += 1;
  }
  eprintln!("Done.");
  return Ok(());
}
