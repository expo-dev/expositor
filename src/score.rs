use crate::color::Color;
use crate::term::Lch;

use std::fmt::Write;

use std::ops::{Not, Neg, Add, Sub, Mul, Div};

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// Absolute scores less than 80.00 are non-checkmate evaluations, absolute scores between
//   80.00 and 90.00 indicate checkmates that are unproven but inevitable with reasonable play,
//   absolute scores between 90.00 and 100.00 indicate inevitable tablebase entry (leading to
//   checkmate), and absolute scores between 100.00 to 250.00 indicate proven checkmates.
//
// An absolute score of 100.00 indicates the position is within the tablebase (and leads to
//   checkmate). A score of −250.00 means the game is over.
//
// The range for inevitable unproven mates is somewhat small (10 pawn as opposed to, say,
//   100 pawn) because we want e.g. to use the same widths for aspiration windows and have
//   them be sensible.

const  REALIZED_WIN : i16 = 250_00; // 100 − 250  winning, proven, realized
const TABLEBASE_WIN : i16 = 100_00; //  90 − 100  winning, proven
const  UNPROVEN_WIN : i16 =  90_00; //  80 −  90  winning
const WIN_THRESHOLD : i16 =  80_00;

const  REALIZED_LOSS : i16 =  -REALIZED_WIN;
const TABLEBASE_LOSS : i16 = -TABLEBASE_WIN;
const  UNPROVEN_LOSS : i16 =  -UNPROVEN_WIN;
const LOSS_THRESHOLD : i16 = -WIN_THRESHOLD;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[derive(Copy, Clone, PartialEq, Eq)] pub struct PovScore { inner : i16 } // point of view
#[derive(Copy, Clone, PartialEq, Eq)] pub struct  CoScore { inner : i16 } // color

impl PovScore {
  pub const ZERO : Self = Self { inner: 0 };
  pub const LOST : Self = Self { inner: REALIZED_LOSS };

  pub const SENTINEL : Self = Self { inner:  i16::MIN };
  pub const MAX      : Self = Self { inner:  i16::MAX };
  pub const MIN      : Self = Self { inner: -i16::MAX };  // i16::MIN + 1

  pub const fn new(s : i16) -> Self { return Self { inner: s }; }

  pub const fn from(&self, pov : Color) -> CoScore
  {
    return CoScore::new(match pov { Color::White => self.inner, Color::Black => -self.inner });
  }

  pub const fn as_i16      (&self) ->  i16 { return  self.inner; }
  pub const fn as_i16_ref  (&self) -> &i16 { return &self.inner; }
  pub const fn abs         (&self) ->  i16 { return  self.inner.abs(); }
  pub const fn unsigned_abs(&self) ->  u16 { return  self.inner.unsigned_abs(); }

  pub const fn is_positive(&self) -> bool { return self.inner > 0; }
  pub const fn is_negative(&self) -> bool { return self.inner < 0; }

  pub const fn is_zero(&self) -> bool { return self.inner == 0; }

  pub const fn is_invalid(&self) -> bool { return self.abs() > REALIZED_WIN; }
  pub const fn is_valid  (&self) -> bool { return !self.is_invalid(); }

  pub const fn is_sentinel(&self) -> bool { return self.inner ==  i16::MIN; }
  pub const fn is_max     (&self) -> bool { return self.inner ==  i16::MAX; }
  pub const fn is_min     (&self) -> bool { return self.inner == -i16::MAX; }

  pub const fn is_checkmate(&self) -> bool { return self.inner == REALIZED_LOSS; }

  pub const fn is_realized_winloss(&self) -> bool { return self.inner.abs() > TABLEBASE_WIN; }
  pub const fn is_proven_winloss  (&self) -> bool { return self.inner.abs() > UNPROVEN_WIN ; }
  pub const fn is_winloss         (&self) -> bool { return self.inner.abs() > WIN_THRESHOLD; }

  pub const fn is_realized_win (&self) -> bool { return self.inner > TABLEBASE_WIN ; }
  pub const fn is_realized_loss(&self) -> bool { return self.inner < TABLEBASE_LOSS; }
  pub const fn is_proven_win   (&self) -> bool { return self.inner > UNPROVEN_WIN  ; }
  pub const fn is_proven_loss  (&self) -> bool { return self.inner < UNPROVEN_LOSS ; }
  pub const fn is_win          (&self) -> bool { return self.inner > WIN_THRESHOLD ; }
  pub const fn is_loss         (&self) -> bool { return self.inner < LOSS_THRESHOLD; }

  pub const fn  realized_win (distance :  u8) -> Self { return Self { inner:  REALIZED_WIN  - distance as i16 }; }
  pub const fn  realized_loss(distance :  u8) -> Self { return Self { inner:  REALIZED_LOSS + distance as i16 }; }
  pub const fn tablebase_win (distance :  u8) -> Self { return Self { inner: TABLEBASE_WIN  - distance as i16 }; }
  pub const fn tablebase_loss(distance :  u8) -> Self { return Self { inner: TABLEBASE_LOSS + distance as i16 }; }
  pub const fn  unproven_win (distance : u16) -> Self { return Self { inner:  UNPROVEN_WIN  - distance as i16 }; }
  pub const fn  unproven_loss(distance : u16) -> Self { return Self { inner:  UNPROVEN_LOSS + distance as i16 }; }

  pub const fn checkmate_moves(&self) -> i16
  {
    return if self.inner >= 0
        {   (REALIZED_WIN+1 - self.inner) / 2  }
      else
        { -((REALIZED_WIN+1 + self.inner) / 2) };
  }

  pub const fn tablebase_moves(&self) -> i16
  {
    return if self.inner >= 0
        {   (TABLEBASE_WIN+1 - self.inner) / 2  }
      else
        { -((TABLEBASE_WIN+1 + self.inner) / 2) };
  }
}

impl CoScore {
  pub const ZERO : Self = Self { inner: 0 };

  pub const SENTINEL : Self = Self { inner:  i16::MIN };
  pub const MAX      : Self = Self { inner:  i16::MAX };
  pub const MIN      : Self = Self { inner: -i16::MAX };  // i16::MIN + 1

  pub const fn new(s : i16) -> Self { return Self { inner: s }; }

  pub const fn from(&self, pov : Color) -> PovScore
  {
    return PovScore::new(match pov { Color::White => self.inner, Color::Black => -self.inner });
  }

  pub const fn as_i16      (&self) ->  i16 { return  self.inner; }
  pub const fn as_i16_ref  (&self) -> &i16 { return &self.inner; }
  pub const fn abs         (&self) ->  i16 { return  self.inner.abs(); }
  pub const fn unsigned_abs(&self) ->  u16 { return  self.inner.unsigned_abs(); }

  pub const fn is_positive(&self) -> bool { return self.inner > 0; }
  pub const fn is_negative(&self) -> bool { return self.inner < 0; }

  pub const fn is_zero(&self) -> bool { return self.inner == 0; }

  pub const fn is_invalid(&self) -> bool { return self.abs() > REALIZED_WIN; }
  pub const fn is_valid  (&self) -> bool { return !self.is_invalid(); }

  pub const fn is_sentinel(&self) -> bool { return self.inner ==  i16::MIN; }
  pub const fn is_max     (&self) -> bool { return self.inner ==  i16::MAX; }
  pub const fn is_min     (&self) -> bool { return self.inner == -i16::MAX; }

  pub const fn is_checkmate(&self) -> bool { return self.inner.abs() == REALIZED_WIN; }

  pub const fn is_realized_winloss(&self) -> bool { return self.inner.abs() > TABLEBASE_WIN; }
  pub const fn is_proven_winloss  (&self) -> bool { return self.inner.abs() > UNPROVEN_WIN ; }
  pub const fn is_winloss         (&self) -> bool { return self.inner.abs() > WIN_THRESHOLD; }

  pub const fn checkmate_moves(&self) -> i16
  {
    return if self.inner >= 0
        {   (REALIZED_WIN+1 - self.inner) / 2  }
      else
        { -((REALIZED_WIN+1 + self.inner) / 2) };
  }

  pub const fn tablebase_moves(&self) -> i16
  {
    return if self.inner >= 0
        {   (TABLEBASE_WIN+1 - self.inner) / 2  }
      else
        { -((TABLEBASE_WIN+1 + self.inner) / 2) };
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn blend(
  c0 : &(f64, f64, f64),
  c1 : &(f64, f64, f64),
  x  : f64
) -> (f64, f64, f64)
{
  return (
    c0.0 * (1.0 - x) + c1.0 * x,
    c0.1 * (1.0 - x) + c1.1 * x,
    c0.2 * (1.0 - x) + c1.2 * x,
  );
}

fn scale(c : &(f64, f64, f64), s : f64) -> (f64, f64, f64)
{
  // scales lightness and chroma (keeping saturation approximately constant)
  return (c.0 * s, c.1 * s, c.2);
}

fn rjust(
  disp_text  : &str,
  disp_width : usize,
  formatter  : &mut std::fmt::Formatter<'_>
) -> std::fmt::Result
{
  if let Some(reqd_width) = formatter.width() {
    if disp_width < reqd_width {
      for _ in 0..(reqd_width - disp_width) {
        formatter.write_char(' ')?;
      }
    }
  }
  return formatter.write_str(disp_text);
}

fn scorefmt(
  score     : i16,
  palette   : &[(f64, f64, f64); 9],
  formatter : &mut std::fmt::Formatter<'_>
) -> std::fmt::Result
{
  if !formatter.alternate() {
    // TODO
  }

  let post = "\x1B[39m";

  if score == 0 {
    let c = &palette[4];
    let pre = Lch::new(c.0, c.1, c.2).to_rgb().fg();
    return rjust(&format!("{pre}0·00{post}"), 4, formatter);
  }

  let abs = score.abs();
  let sign = if score < 0 { "−" } else { "+" };

  if abs > TABLEBASE_WIN {
    let moves = (REALIZED_WIN+1 - abs) / 2;

    let c0 = if score < 0 { &palette[0] } else { &palette[8] };
    let c1 = if score < 0 { &palette[1] } else { &palette[7] };

    if moves == 0 {
      let pre = Lch::new(c0.0, c0.1, c0.2).to_rgb().fg();
      return rjust(&format!("{pre}#0{post}"), 2, formatter);
    }

    let k = 2.0_f64.ln() / -8.0;
    let x = (std::cmp::max(REALIZED_WIN - abs, 1) - 1) as f64;
    let x = 1.0 - (x * k).exp();

    let ci = blend(c0, c1, x);
    let cs = blend(&scale(c0, 0.75), &scale(c1, 0.75), x);

    let pre = Lch::new(ci.0, ci.1, ci.2).to_rgb().fg();
    let dim = Lch::new(cs.0, cs.1, cs.2).to_rgb().fg();

    let width = if moves > 9 { 4 } else { 3 };
    return rjust(
      &format!("{pre}#{dim}{sign}{pre}{moves}{post}"),
      width, formatter
    );
  }

  let int = abs / 100;
  let frac = abs  - (int * 100);

  let c0 = if score < 0 { &palette[3] } else { &palette[5] };
  let c1 = if score < 0 { &palette[2] } else { &palette[6] };

  let k = 3.0_f64.ln() / -100.0;
  let x = (1.0 + (abs as f64 * k).exp()).recip();
  let x = x * 2.0 - 1.0;

  let ci = blend(c0, c1, x);
  // let cs = c1;

  let pre = Lch::new(ci.0, ci.1, ci.2).to_rgb().fg();
  // let ext = Lch::new(cs.0, cs.1, cs.2).to_rgb().fg();

  let width = if int > 99 { 7 } else if int > 9 { 6 } else { 5 };
  return rjust(
    // &format!("{ext}{sign}{pre}{int}{ext}·{pre}{frac:02}{post}"),
    &format!("{pre}{sign}{int}·{frac:02}{post}"),
    width, formatter
  );
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

impl std::fmt::Octal for PovScore {
  fn fmt(&self, f : &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    // We use the o flag for UCI output
    if self.is_checkmate() {
      return f.write_str("mate 0");
    }
    if self.is_realized_winloss() && self.is_valid() {
      return f.write_fmt(format_args!("mate {}", self.checkmate_moves()));
    }
    return f.write_fmt(format_args!("cp {}", self.as_i16()));
  }
}

impl std::fmt::Display for PovScore {
  fn fmt(&self, f : &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    // Anomalous scores
    if self.is_sentinel() { return f.write_str("PovScore::SENTINEL"); }
    if self.is_max()      { return f.write_str("PovScore::MAX"     ); }
    if self.is_min()      { return f.write_str("PovScore::MIN"     ); }
    if self.is_invalid() {
      return f.write_fmt(format_args!("PovScore {{ {:+} }}", self.inner));
    }
    let palette = [
      (0.6667, 0.175,  30.0 / 360.0),
      (0.75  , 0.135,  45.0 / 360.0),
      (0.75  , 0.135,  45.0 / 360.0),
      (0.9583, 0.0  ,  60.0 / 360.0),
      (0.9583, 0.0  ,   0.0        ),
      (0.9583, 0.0  , 315.0 / 360.0),
      (0.7083, 0.195, 315.0 / 360.0),
      (0.7083, 0.195, 315.0 / 360.0),
      (0.625 , 0.215, 315.0 / 360.0),
    ];
    return scorefmt(self.as_i16(), &palette, f);
  }
}

impl std::fmt::Display for CoScore {
  fn fmt(&self, f : &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    // Anomalous scores
    if self.is_sentinel() { return f.write_str("CoScore::SENTINEL"); }
    if self.is_max()      { return f.write_str("CoScore::MAX"     ); }
    if self.is_min()      { return f.write_str("CoScore::MIN"     ); }
    if self.is_invalid() {
      return f.write_fmt(format_args!("CoScore {{ {:+} }}", self.inner));
    }
    let palette = [
      (0.6   , 0.215,  15.0 / 360.0),
      (0.6667, 0.175,  22.5 / 360.0),
      (0.6667, 0.175,  30.0 / 360.0),
      (0.9583, 0.0  ,  30.0 / 360.0),
      (0.75  , 0.175, 157.5 / 360.0),
      (0.9583, 0.0  , 270.0 / 360.0),
      (0.6667, 0.175, 270.0 / 360.0),
      (0.6667, 0.175, 285.0 / 360.0),
      (0.6   , 0.235, 300.0 / 360.0),
    ];
    return scorefmt(self.as_i16(), &palette, f);
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(i8)]
pub enum PovOutcome {
  Unknown = i8::MIN,
  Loss    = -1,
  Draw    =  0,
  Win     =  1,
}

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(i8)]
pub enum CoOutcome {
  Unknown = i8::MIN,
  Black   = -1,
  Draw    =  0,
  White   =  1,
}

impl PovOutcome {
  pub fn from(&self, pov : Color) -> CoOutcome
  {
    return match pov {
      Color::White => unsafe { std::mem::transmute(*self) },
      Color::Black => unsafe { std::mem::transmute((*self as i8).wrapping_neg()) }
    };
  }
}

impl CoOutcome {
  pub fn from(&self, pov : Color) -> PovOutcome
  {
    return match pov {
      Color::White => unsafe { std::mem::transmute(*self) },
      Color::Black => unsafe { std::mem::transmute((*self as i8).wrapping_neg()) }
    };
  }
}

impl std::fmt::Display for PovOutcome {
  fn fmt(&self, f : &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    if f.alternate() {
      let s = match self {
          Self::Unknown => "Unknown",
          Self::Loss    => "PovLoss",
          Self::Draw    => "Draw",
          Self::Win     => "PovWin",
      };
      if let Some(reqd_width) = f.width() {
        return write!(f, "{s:0$}", reqd_width);
      }
      return f.write_str(s);
    }
    return f.write_char(match self {
        Self::Unknown => '*',
        Self::Loss    => 'L',
        Self::Draw    => 'D',
        Self::Win     => 'W',
    });
  }
}

impl std::fmt::Display for CoOutcome {
  fn fmt(&self, f : &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    if f.alternate() {
      let s = match self {
          Self::Unknown => "Unknown",
          Self::Black   => "BlackWin",
          Self::Draw    => "Draw",
          Self::White   => "WhiteWin",
      };
      if let Some(reqd_width) = f.width() {
        return write!(f, "{s:0$}", reqd_width);
      }
      return f.write_str(s);
    }
    return f.write_char(match self {
        Self::Unknown => '*',
        Self::Black   => 'b',
        Self::Draw    => 'd',
        Self::White   => 'w',
    });
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

impl Neg for PovScore { type Output = PovScore; fn neg(self) -> Self::Output { return Self::Output { inner: -self.inner }; } }
impl Not for PovScore { type Output = PovScore; fn not(self) -> Self::Output { return Self::Output { inner: -self.inner }; } }

impl PartialOrd for PovScore { fn partial_cmp(&self, other : &Self) -> Option<std::cmp::Ordering> { return Some(self.inner.cmp(&other.inner)); } }
impl        Ord for PovScore { fn         cmp(&self, other : &Self) ->        std::cmp::Ordering  { return      self.inner.cmp(&other.inner) ; } }

impl Add< PovScore> for  PovScore { type Output = PovScore; fn add(self, other :  Self::Output) -> Self::Output { return Self::Output { inner: self.inner + other.inner }; } }
impl Add<&PovScore> for  PovScore { type Output = PovScore; fn add(self, other : &Self::Output) -> Self::Output { return Self::Output { inner: self.inner + other.inner }; } }
impl Add< PovScore> for &PovScore { type Output = PovScore; fn add(self, other :  Self::Output) -> Self::Output { return Self::Output { inner: self.inner + other.inner }; } }
impl Add<&PovScore> for &PovScore { type Output = PovScore; fn add(self, other : &Self::Output) -> Self::Output { return Self::Output { inner: self.inner + other.inner }; } }

impl Sub< PovScore> for  PovScore { type Output = PovScore; fn sub(self, other :  Self::Output) -> Self::Output { return Self::Output { inner: self.inner - other.inner }; } }
impl Sub<&PovScore> for  PovScore { type Output = PovScore; fn sub(self, other : &Self::Output) -> Self::Output { return Self::Output { inner: self.inner - other.inner }; } }
impl Sub< PovScore> for &PovScore { type Output = PovScore; fn sub(self, other :  Self::Output) -> Self::Output { return Self::Output { inner: self.inner - other.inner }; } }
impl Sub<&PovScore> for &PovScore { type Output = PovScore; fn sub(self, other : &Self::Output) -> Self::Output { return Self::Output { inner: self.inner - other.inner }; } }

impl Mul< i16     > for  PovScore { type Output = PovScore; fn mul(self, other :  i16         ) -> Self::Output { return Self::Output { inner: self.inner * other       }; } }
impl Mul<&i16     > for  PovScore { type Output = PovScore; fn mul(self, other : &i16         ) -> Self::Output { return Self::Output { inner: self.inner * other       }; } }
impl Mul< i16     > for &PovScore { type Output = PovScore; fn mul(self, other :  i16         ) -> Self::Output { return Self::Output { inner: self.inner * other       }; } }
impl Mul<&i16     > for &PovScore { type Output = PovScore; fn mul(self, other : &i16         ) -> Self::Output { return Self::Output { inner: self.inner * other       }; } }

impl Div< i16     > for  PovScore { type Output = PovScore; fn div(self, other :  i16         ) -> Self::Output { return Self::Output { inner: self.inner / other       }; } }
impl Div<&i16     > for  PovScore { type Output = PovScore; fn div(self, other : &i16         ) -> Self::Output { return Self::Output { inner: self.inner / other       }; } }
impl Div< i16     > for &PovScore { type Output = PovScore; fn div(self, other :  i16         ) -> Self::Output { return Self::Output { inner: self.inner / other       }; } }
impl Div<&i16     > for &PovScore { type Output = PovScore; fn div(self, other : &i16         ) -> Self::Output { return Self::Output { inner: self.inner / other       }; } }

impl Add< CoScore> for  CoScore { type Output = CoScore; fn add(self, other :  Self::Output) -> Self::Output { return Self::Output { inner: self.inner + other.inner }; } }
impl Add<&CoScore> for  CoScore { type Output = CoScore; fn add(self, other : &Self::Output) -> Self::Output { return Self::Output { inner: self.inner + other.inner }; } }
impl Add< CoScore> for &CoScore { type Output = CoScore; fn add(self, other :  Self::Output) -> Self::Output { return Self::Output { inner: self.inner + other.inner }; } }
impl Add<&CoScore> for &CoScore { type Output = CoScore; fn add(self, other : &Self::Output) -> Self::Output { return Self::Output { inner: self.inner + other.inner }; } }

impl Sub< CoScore> for  CoScore { type Output = CoScore; fn sub(self, other :  Self::Output) -> Self::Output { return Self::Output { inner: self.inner - other.inner }; } }
impl Sub<&CoScore> for  CoScore { type Output = CoScore; fn sub(self, other : &Self::Output) -> Self::Output { return Self::Output { inner: self.inner - other.inner }; } }
impl Sub< CoScore> for &CoScore { type Output = CoScore; fn sub(self, other :  Self::Output) -> Self::Output { return Self::Output { inner: self.inner - other.inner }; } }
impl Sub<&CoScore> for &CoScore { type Output = CoScore; fn sub(self, other : &Self::Output) -> Self::Output { return Self::Output { inner: self.inner - other.inner }; } }

impl Mul< i16    > for  CoScore { type Output = CoScore; fn mul(self, other :  i16         ) -> Self::Output { return Self::Output { inner: self.inner * other       }; } }
impl Mul<&i16    > for  CoScore { type Output = CoScore; fn mul(self, other : &i16         ) -> Self::Output { return Self::Output { inner: self.inner * other       }; } }
impl Mul< i16    > for &CoScore { type Output = CoScore; fn mul(self, other :  i16         ) -> Self::Output { return Self::Output { inner: self.inner * other       }; } }
impl Mul<&i16    > for &CoScore { type Output = CoScore; fn mul(self, other : &i16         ) -> Self::Output { return Self::Output { inner: self.inner * other       }; } }

impl Div< i16    > for  CoScore { type Output = CoScore; fn div(self, other :  i16         ) -> Self::Output { return Self::Output { inner: self.inner / other       }; } }
impl Div<&i16    > for  CoScore { type Output = CoScore; fn div(self, other : &i16         ) -> Self::Output { return Self::Output { inner: self.inner / other       }; } }
impl Div< i16    > for &CoScore { type Output = CoScore; fn div(self, other :  i16         ) -> Self::Output { return Self::Output { inner: self.inner / other       }; } }
impl Div<&i16    > for &CoScore { type Output = CoScore; fn div(self, other : &i16         ) -> Self::Output { return Self::Output { inner: self.inner / other       }; } }
