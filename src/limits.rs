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
  pub depth  : u8,
  pub target : Option<f64>,
  pub cutoff : Option<f64>,
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

  //============================================================================================
  pub fn calculate_limits(&self, state : &State) -> Limits
  {
    let mut limits = Limits {
      depth:  64,
      target: None,
      cutoff: None,
    };

    let overhead = self.overhead as f64 * 0.001;
    if let Some(depth) = self.depth {
      limits.depth = match depth { 0 => 1, 64..=255 => 64, _ => depth };
    }

    // Mode 1. Fixed time  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    if let Some(movetime) = self.movetime {
      let secs = movetime as f64 * 0.001;
      limits.cutoff = Some((secs - overhead).max(0.0));
      return limits;
    }

    // Mode 2. Time control  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    //   See https://www.desmos.com/calculator/olv4i9y3rj, which incidental is not so different
    //   from Stockfish's time control: https://www.desmos.com/calculator/jm26qglpzc.

    let mut clock : Option<f64> = None;
    let mut increment : f64 = 0.0;

    match state.turn {
      Color::White => {
        if let Some(clk) = self.wtime { clock     = Some(clk as f64 * 0.001); }
        if let Some(inc) = self.winc  { increment =      inc as f64 * 0.001 ; }
      }
      Color::Black => {
        if let Some(clk) = self.btime { clock     = Some(clk as f64 * 0.001); }
        if let Some(inc) = self.binc  { increment =      inc as f64 * 0.001 ; }
      }
    }

    if let Some(seconds_remaining) = clock {
      // In search, we impose a minimum depth that must be reached before we're allowed to break
      //   due to time limits. This usually takes 0.010 seconds or less.
      let min_time = 0.010;
      let no_increment = increment <= min_time + overhead;
      let usable = seconds_remaining - overhead;

      let moves_remaining;
      if let Some(movestogo) = self.movestogo {
        // We first handle a special case.
        if movestogo == 1 {
          // Use all the remaining time, but try not to cut it too close.
          let search_time = if usable >= 10.0 { usable - 1.0 } else { usable * 0.9 };
          // We leave target = None to communicate fixed time.
          limits.cutoff = Some(search_time.max(0.0));
          return limits;
        }
        // According to the informal UCI specification, "[movestogo x] will
        //   only be sent if x is greater than zero", but we want to be robust.
        moves_remaining = movestogo.max(2) as f64;
      }
      else {
        let ply_remaining = linear_model(state);
        moves_remaining = (ply_remaining * 0.5).max(if no_increment { 24.0 } else { 8.0 });
      }

      const one_third : f64 = 0.333_333_333_333_333_333;
      const two_third : f64 = 0.666_666_666_666_666_667;

      let base;
      if no_increment {
        base = seconds_remaining / moves_remaining;
      }
      else {
        base = (seconds_remaining - increment) / moves_remaining + increment;
      }
      let target = (   base   ).min(seconds_remaining * one_third);
      let cutoff = (base * 3.0).min(seconds_remaining * two_third);
      limits.target = Some((target - overhead).max(0.0));
      limits.cutoff = Some((cutoff - overhead).max(0.0));
      return limits;
    }

    // Modes 3 and 4. Default and Fixed depth  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    if self.depth.is_none() { limits.cutoff = Some(10.0); }
    return limits;
  }
  //============================================================================================
}

fn linear_model(state : &State) -> f64
{
  // Fit from TCEC games and games that Expositor played on Lichess
  //   that were all at least 30 moves long and less than 120 moves long.
  // The coefficient of determination for the regression is about 0??47
  //   and the mean absolute deviation was about 25??5 ply.
  let ply = state.ply as f64;
  let men = (state.sides[0] | state.sides[1]).count_ones() as f64 - 2.0;
  let rem = 39.0 - ply*0.25 + men*2.6;
  return rem;
}
