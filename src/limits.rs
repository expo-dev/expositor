use crate::color::Color::*;
use crate::search::MAX_DEPTH;
use crate::state::State;
use crate::misc::{ONE_THIRD, TWO_THIRD};

#[derive(Clone)]
pub struct SearchParams {
  pub overhead  : usize,          //
  pub movetime  : Option<usize>,  //
  pub wtime     : Option<usize>,  // These integer times
  pub btime     : Option<usize>,  //   are in milliseconds
  pub winc      : Option<usize>,  //
  pub binc      : Option<usize>,  //
  pub movestogo : Option<usize>,
  pub nodes     : Option<usize>,
  pub depth     : Option<u8>   ,
}

#[derive(Clone)]
pub struct Limits {
  pub cutoff : Option<f64>,
  pub target : Option<f64>,
  pub nodes  : usize,
  pub depth  : u8,
}

impl Limits {
  pub const fn new() -> Self
  {
    return Self {
      cutoff: None,
      target: None,
      nodes:  0,
      depth:  0,
    };
  }
}

impl SearchParams {
  pub const fn new() -> Self
  {
    return Self {
      overhead:  0,
      movetime:  None,
      wtime:     None,
      btime:     None,
      winc:      None,
      binc:      None,
      movestogo: None,
      nodes:     None,
      depth:     None,
    };
  }

  //============================================================================================
  pub fn calculate_limits(&self, state : &State, default : f64) -> Limits
  {
    const MAX_DEPTH_U8 : u8 = MAX_DEPTH as u8;

    let mut limits = Limits {
      cutoff: None,
      target: None,
      nodes:  0,
      depth:  MAX_DEPTH_U8,
    };

    if let Some(nodes) = self.nodes {
      limits.nodes = std::cmp::max(nodes, 1);
    }

    if let Some(depth) = self.depth {
      limits.depth = match depth { 0 => 1, MAX_DEPTH_U8..=255 => MAX_DEPTH_U8, _ => depth };
    }

    // Mode 1. Fixed time  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    let overhead = self.overhead as f64 * 0.001;

    if let Some(movetime) = self.movetime {
      let secs = movetime as f64 * 0.001;
      limits.cutoff = Some((secs - overhead).max(0.0));
      return limits;
    }

    // Mode 2. Time control  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    //   See https://www.desmos.com/calculator/olv4i9y3rj, which incidentally is not so
    //   different from Stockfish's time control: https://www.desmos.com/calculator/jm26qglpzc.

    let mut clock : Option<f64> = None;
    let mut increment : f64 = 0.0;

    match state.turn {
      White => {
        if let Some(clk) = self.wtime { clock     = Some(clk as f64 * 0.001); }
        if let Some(inc) = self.winc  { increment =      inc as f64 * 0.001 ; }
      }
      Black => {
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
        moves_remaining = (ply_remaining * 0.5).max(if no_increment { 24.0 } else { 12.0 });
      }

      let base;
      if no_increment {
        base = seconds_remaining / moves_remaining;
      }
      else {
        base = (seconds_remaining - increment) / moves_remaining + increment;
      }
      // If we wanted to spend the same time per move, we'd write
      //   base = seconds_remaining / moves_remaining;
      // but we usually want to spend more time in the opening
      //   and midgame, so we reshape a bit.
      let base =
        if state.ply < 80 { base * (1.5 - 0.00625 * state.ply as f64) }
        else              { base                                      };
      let target = (   base   ).min(seconds_remaining * ONE_THIRD);
      let cutoff = (base * 3.0).min(seconds_remaining * TWO_THIRD);
      limits.target = Some((target - overhead).max(0.0));
      limits.cutoff = Some((cutoff - overhead).max(0.0));
      return limits;
    }

    // Modes 3 and 4. Default and Fixed depth  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    if self.depth.is_none() { limits.cutoff = Some(default); }
    return limits;
  }
  //============================================================================================
}

fn linear_model(state : &State) -> f64
{
  // Fit from TCEC games from Season 19 onward that were all at least 30 moves
  //   long and at most 120 moves long.
  // The coefficient of determination for the regression was about 0·48 and the
  //   mean absolute deviation was about 27 ply.
  let ply = state.ply as f64;
  let men = (state.sides[0] | state.sides[1]).count_ones() as f64 - 2.0;
  let rem = 29.0 - ply*0.2 + men*3.0;
  return rem;
}
