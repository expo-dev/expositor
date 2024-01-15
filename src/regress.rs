use crate::import::{PgnReader, Termination};
use crate::piece::Piece::*;
use crate::state::State;

// These routines are used to find the constants for the linear model used in
//   limits.rs to estimate the number of ply remaining in a game.

fn load_dataset(path : &str) -> std::io::Result<Vec<[f32; 3]>>
{
  eprint!("Loading...\r");
  let mut game_count : usize = 0;
  let mut skip_count : usize = 0;
  let mut dataset = Vec::new();
  for game in PgnReader::open(path)? {
    let game = game?;
    if game.termination == Termination::Flag {
      skip_count += 1;
      continue;
    }
    let movelist = &game.movelist;
    let game_length = movelist.len() as u16;
    let mut working = State::new();
    for mv in movelist {
      working.apply(&mv);
      let wq = working.boards[WhiteQueen ].count_ones() as i16;
      let wr = working.boards[WhiteRook  ].count_ones() as i16;
      let wb = working.boards[WhiteBishop].count_ones() as i16;
      let wn = working.boards[WhiteKnight].count_ones() as i16;
      let wp = working.boards[WhitePawn  ].count_ones() as i16;
      let bq = working.boards[BlackQueen ].count_ones() as i16;
      let br = working.boards[BlackRook  ].count_ones() as i16;
      let bb = working.boards[BlackBishop].count_ones() as i16;
      let bn = working.boards[BlackKnight].count_ones() as i16;
      let bp = working.boards[BlackPawn  ].count_ones() as i16;

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
  eprintln!("{:9} skipped", skip_count);
  eprintln!("{:9} positions", dataset.len());
  return Ok(dataset);
}

// Returns (sum of squared error, count of negative predictions, derivatives w/r/t coeffs)
fn error(dataset : &Vec<[f32; 3]>, coeffs : &[f32; 3]) -> (f32, usize, [f32; 3])
{
  let mut err = 0.0;
  let mut neg = 0;
  #[allow(non_snake_case)]
  let mut dE = [0.0, 0.0, 0.0];
  for point in dataset.iter() {
    let prediction = coeffs[0]              // constant
                   + coeffs[1] * point[1]   // ply
                   + coeffs[2] * point[2];  // pieces
    let delta = prediction - point[0];
    err += delta * delta;
    if prediction < 0.0 { neg += 1; }
    dE[0] += delta;
    dE[1] += point[1] * delta;
    dE[2] += point[2] * delta;
  }
  dE[0] *= 2.0;
  dE[1] *= 2.0;
  dE[2] *= 2.0;
  return (err, neg, dE);
}

// Returns the coefficient of determination
fn deter(dataset : &Vec<[f32; 3]>, err : f32) -> f32
{
  let mut sum = 0.0;
  for point in dataset.iter() { sum += point[0]; }
  let mean = sum / dataset.len() as f32;

  let mut sum = 0.0;
  for point in dataset.iter() {
    let d = point[0] - mean;
    sum += d*d;
  }
  return 1.0 - err / sum;
}

pub fn fit_dataset(path : &str) -> std::io::Result<()>
{
  let dataset =
    load_dataset(path)?.into_iter()
                       .filter(|p| { let ply = p[0]+p[1]; 240.0 >= ply && ply >= 60.0 })
                       .collect::<Vec<[f32; 3]>>();

  eprintln!("{:9} positions in range", dataset.len());

  //~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

  eprint!("\n");
  eprintln!("step   opt rate    error     const   ply    men    pgl    cd    neg");

  let mut coeffs = [40.0, -0.25, 2.5];
  let mut err = error(&dataset, &coeffs);

  let mut rate = 0.1;
  let mut stray = 0.0;
  for step in 0..16384 {
    for x in 0..3 {
      let denom = (err.0 + err.2[x].abs()).copysign(err.2[x]);
      coeffs[x] -= err.0 / denom * rate;
    }
    let prev = err.2;
    err = error(&dataset, &coeffs);

    let mut flips = 0.0;
    for x in 0..3 {
      flips += if prev[x] * err.2[x] < 0.0 { 1.0 } else { 0.0 };
    }
    stray = stray * 0.75 + flips;
    if stray >= 11.0 { rate *= 0.5; stray = 0.0; }

    if step % 64 == 63 {
      let start_estimate = coeffs[0] + coeffs[2] * 30.0;
      eprint!(
        "\r{:6} \x1B[2m{:.9}\x1B[22m {:9.6} {:+7.3} {:+6.3} {:+6.3} \x1B[2m{:6.2}\x1B[22m \x1B[94m{:.3}\x1B[39m",
        step+1, rate, (err.0 / (dataset.len() as f32)).sqrt(),
        coeffs[0], coeffs[1], coeffs[2],
        start_estimate,
        deter(&dataset, err.0)
      );
      if err.1 > 0 {
        eprint!(" \x1B[91m{:3.1}%\x1B[39m", err.1 as f32 * 100.0 / dataset.len() as f32);
      }
      else {
        eprint!("     ");
      }
    }
  }
  eprint!("\n");

  //~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

  eprint!("\n");
  eprintln!("avg abs dev");

  let mut dev = 0.0;
  for point in dataset.iter() {
    let prediction = coeffs[0] + coeffs[1]*point[1] + coeffs[2]*point[2];
    dev += (prediction - point[0]).abs();
  }

  eprintln!("{:.3}", dev / (dataset.len() as f32));

  //~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

  eprint!("\n");
  return Ok(());
}
