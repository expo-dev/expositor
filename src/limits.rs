use crate::color::*;
use crate::state::*;

#[derive(Clone)]
pub struct SearchParams {
  pub overhead  : usize,          //
  pub movetime  : Option<usize>,  //
  pub wtime     : Option<usize>,  // These integer times
  pub btime     : Option<usize>,  //   are in milliseconds
  pub winc      : Option<usize>,  //
  pub binc      : Option<usize>,  //
  pub movestogo : Option<usize>,
  pub depth     : Option<u8>   ,
}

#[derive(Clone)]
pub struct Limits {
  pub depth     : u8,
  pub time_soft : Option<f64>,
  pub time_hard : Option<f64>,
}

impl SearchParams {
  pub const fn new() -> Self
  {
    return SearchParams {
      overhead:  0,
      movetime:  None,
      wtime:     None,
      btime:     None,
      winc:      None,
      binc:      None,
      movestogo: None,
      depth:     None,
    };
  }

  pub fn limits(&self, state : &State) -> Limits
  {
    let mut lim = Limits {
      depth:     64,
      time_soft: None,
      time_hard: None,
    };

    let overhead = self.overhead as f64 * 0.001;
    let mut default = true;

    let mut clock     : f64 = -1.0;
    let mut increment : f64 =  0.0;

    match state.turn {
      Color::White => {
        if let Some(clk) = self.wtime { clock     = clk as f64 * 0.001; }
        if let Some(inc) = self.winc  { increment = inc as f64 * 0.001; }
      }
      Color::Black => {
        if let Some(clk) = self.btime { clock     = clk as f64 * 0.001; }
        if let Some(inc) = self.binc  { increment = inc as f64 * 0.001; }
      }
    }

    if !(clock < 0.0) {
      let do_not_exceed = (clock - 1.0).max(0.0);
      let max_time = ((clock - increment) * 0.125 + increment).min(do_not_exceed);
      let base =
        if let Some(movestogo) = self.movestogo {
          (clock - increment) / std::cmp::max(movestogo, 1) as f64 + increment
        }
        else {
          ideal_time(state, clock, increment)
        };
      let soft = base.min(max_time);
      let hard = (base * 2.0).min(max_time);
      lim.time_soft = Some((soft - overhead).max(0.0));
      lim.time_hard = Some((hard - overhead).max(0.0));
      default = false;
    }

    // Overrides other time limits
    if let Some(movetime) = self.movetime {
      let secs = movetime as f64 * 0.001;
      lim.time_soft = None;
      lim.time_hard = Some((secs - overhead).max(0.0));
      default = false;
    }

    if let Some(depth) = self.depth { lim.depth = depth; default = false; }
    if lim.depth == 0 { lim.depth =  1; }
    if lim.depth > 64 { lim.depth = 64; }

    if default {
      lim.time_soft = None;
      lim.time_hard = Some(10.0);
    }
    return lim;
  }
}

fn ideal_time(state : &State, seconds_remaining : f64, increment : f64) -> f64
{
  let time_control = seconds_remaining + 240.0*increment;
  let remaining_ply;
  // Bullet time control
  if time_control < 60.0 {
    remaining_ply = bullet_model(state);
  }
  // Transitional time control
  else if time_control < 120.0 {
    let blend = (time_control - 60.0) / 60.0;
    remaining_ply = blend*standard_model(state) + (1.0-blend)*bullet_model(state);
  }
  // Standard time control
  else {
    remaining_ply = standard_model(state);
  }
  let remaining_moves = (remaining_ply * 0.5).max(1.0);
  return (seconds_remaining - increment) / remaining_moves + increment;
}

fn standard_model(state : &State) -> f64
{
  // Fit from TCEC games that lasted 60 to 240 ply
  // The coefficient of determination was about 0.45
  //   and the average squared error was about 1200
  // Adding dfz, breaking down piece count by pieces vs
  //   pawns, and the static evaluation to the linear model
  //   did not improve upon this in any meaningful way
  let ply = state.ply as f64;
  let men = (state.sides[0] | state.sides[1]).count_ones() as f64 - 2.0;
  let rem = 35.0 - ply*0.24 + men*2.8;
  return rem;
}

fn bullet_model(state : &State) -> f64
{
  // Fit from a small collection of games that Expositor
  //   played on Lichess that lasted 60 ply or more
  // The coefficient of determination was about 0.46
  //   and the average squared error was about 1250
  // This should be revisited in the future
  let ply = state.ply as f64;
  let men = (state.sides[0] | state.sides[1]).count_ones() as f64 - 2.0;
  let rem = 29.0 - ply*0.125 + men*3.2;
  return rem;
}
