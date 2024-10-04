use crate::color::*;
use crate::dest::*;
use crate::import::*;
use crate::misc::*;
use crate::piece::*;
use crate::rand::*;
use crate::score::*;
use crate::state::*;

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

The side waiting will not have any of its pieces en prise (because resolving search will explore
  the capture), but it's possible for the side to move to have its pieces en prise.

Can a piece move safely to a square? This isn't possible to answer without performing static
  exchange analysis (or more correctly, without performing a search), which is too expensive to
  do for every square. However, we can approximate the answer.

  Is it safe for a piece to move to an empty square?
  ├─ If the square is not attacked by an enemy piece, then YES.
  ├─ If the square is attacked by an enemy piece of lesser value, then NO.
  └─ If the square is attacked by an enemy piece of equal or greater value,
     ├─ If the square is not defended by a friendly piece, then NO.
     ├─ If the square is defended by a friendly piece,
     │  ├─ If the square is attacked by only one enemy piece, then YES.
     │  └─ Otherwise, UNKNOWN.
     └─ Otherwise, UNKNOWN.

  Is it safe for a piece to capture on an occupied square?
  ├─ If the piece is of equal or greater value, then YES.
  └─ If the piece is of lesser value,
     ├─ If the square is not defended by an enemy piece, then YES.
     └─ If the square is defended by an enemy piece,
        ├─ If the square is not attacked by a friendly piece, then NO.
        └─ If the square is attacked by a friendly piece,
           ├─ If the square is defended by only one enemy piece,
           │  ├─ If the combined value of the captured enemy piece and defending enemy
           │  │    piece is greater than or equal to the value of the piece, then YES.
           │  └─ Otherwise, NO.
           └─ Otherwise, UNKNOWN.

If we err on the side of classifying UNKNOWNs as NOs, and make a
  simplification in the case of an occupied square, then this becomes:

  Is it safe for a piece to move to an empty square?
  ├─ If the square is not attacked by an enemy piece, then YES.
  ├─ If the square is attacked by only one enemy piece and it is of equal or
  │    greater value, and the square is defended by a friendly piece, then YES.
  └─ Otherwise, NO.

  Is it safe for a piece to capture on an occupied square?
  ├─ If the captured piece is of equal or greater value, then YES.
  ├─ If the captured piece is not defended by an enemy piece, then YES.
  ├─ If the captured piece is defended by only one enemy piece and it is of equal
  │    or greater value, and the square is attacked by a friendly piece, then YES.
  └─ Otherwise, NO.

And so the two end up being identical except for the rule that it is always safe to capture a
  piece of greater or equal value.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

const ONETHIRD   : f64 = 0.333_333_333_333_333_333_333_333;
const ONESEVENTH : f64 = 0.142_857_142_857_142_857_142_857;

fn compress(x : f64) -> f64 {
  return (1.0 + (x.abs()*ONETHIRD - 1.0) / (x.abs()*ONETHIRD + 1.0)).copysign(x);
}

fn d_compress(x : f64) -> f64 {
  let denom = x.abs() + 3.0; return 6.0 / (denom * denom);
}

fn softclip(x : f64) -> f64 {
  return x.max(x / 16.0).min(x / 16.0 + 15.0 / 16.0);
}

fn d_softclip(x : f64) -> f64 {
  return if x < 0.0 || 1.0 < x { 1.0 / 16.0 } else { 1.0 };
}

#[inline]
fn flip(x : f64, s : usize) -> f64 {
  // Equivalent to "return if s == 0 { x } else { -x };" when s is 0 or 1.
  unsafe {
    let a = std::mem::transmute::<_,u64>(x);
    let b = (s as u64) << 63;
    return std::mem::transmute(a ^ b);
  }
}

#[inline] fn fpcount(w : u64) -> f64 { return w.count_ones() as f64; }
#[inline] fn unitize(w : u64) -> f64 { return if w != 0 { 1.0 } else { 0.0 }; }
#[inline] fn multize(w : u64) -> f64 { return if w.count_ones() > 1 { 1.0 } else { 0.0 }; }

#[inline] fn diff(a : u8, b : u8) -> u8 { return (b as i8 - a as i8).abs() as u8; }

#[allow(non_snake_case)]
fn cubic(x : f64, a : &[f64; 4], ds : &mut [f64; 4], scale : f64, color : usize) -> f64 {
  debug_assert!(x < 1.015625, "{} > 1", x);
  debug_assert!(x >= 0.0,     "{} < 0", x);
  let j   = 1.0 - x;
  let jj  = j * j;
  let jjj = jj * j;
  let k   = x;
  let kk  = k * k;
  let kkk = kk * k;
  let c0 = jjj;
  let c1 = jj * k * 3.0;
  let c2 = j * kk * 3.0;
  let c3 = kkk;
  ds[0] += flip(c0 * scale, color);
  ds[1] += flip(c1 * scale, color);
  ds[2] += flip(c2 * scale, color);
  ds[3] += flip(c3 * scale, color);
  let z = c0 * a[0] + c1 * a[1] + c2 * a[2] + c3 * a[3];
  return flip(z * scale, color);
}

#[inline]
fn norm_area(x : usize, n : usize) -> usize {
  // https://www.desmos.com/calculator/bgwuvtarui
  return match x {
    3 =>  n,
    5 => (n*3 + 2) / 5,
    8 => (n*3 + 2) / 8,
    _ => unreachable!()
  };
}

fn king_shield(
  pov_pawns : u64,
  opp_pawns : u64,
  king_idx  : usize,
  fst : &[[f64; 6]; 36],
  snd :  &[f64; 6],
  dfst : &mut [[f64; 6]; 36],
  dsnd : &mut  [f64; 6],
) -> f64
{
  let mut i = [0.0; 36];

  let rank = king_idx / 8;
  let file = king_idx % 8;

  //  -9,  -8,  -7,       0,  1,  2,       0,  1,  2,       0,  1,  2,
  //  -1,       +1,       8,  9, 10,       3,  4,  5,       3,      4,
  //  +7,  +8,  +9,  ->  16, 17, 18,  ->   6,  7,  8,  ->   5,  6,  7,
  // +15, +16, +17,      24, 25, 26,       9, 10, 11,       8,  9, 10,
  // +23, +24, +25       27, 28, 29,      12, 13, 14       11, 12, 13

  let idx = king_idx as i8;
  let column = FILE_A << file;
  let region = column | ((column & !FILE_H) << 1) | ((column & !FILE_A) >> 1);

  let side_pawns = [pov_pawns, opp_pawns];
  for side in 0..2 {
    let mut pawns = side_pawns[side] & region;
    while pawns != 0 {
      let src = pawns.trailing_zeros() as i8;
      let src = src - idx + 9;
      if src < 0 || src > 34 { pawns &= pawns - 1; continue; }
      let mut src = (src / 8) * 3 + (src % 8);
      if src > 4 { src -= 1; }
      debug_assert!(src >= 0 && src < 14);
      i[side*14 + src as usize] = 1.0;
      pawns &= pawns - 1;
    }
  }

  if file == 7 { i[28] = 1.0; }
  if file == 6 { i[29] = 1.0; }
  if file == 1 { i[30] = 1.0; }
  if file == 0 { i[31] = 1.0; }
  if rank == 7 { i[32] = 1.0; }
  if rank == 6 { i[33] = 1.0; }
  if rank == 1 { i[34] = 1.0; }
  if rank == 0 { i[35] = 1.0; }

  let mut s = [0.0; 6];

  // `dsnd` happens to be what we usually call `a`
  for x in 0..36 { for y in 0..6 { s[y] += i[x] * fst[x][y]; } }
  for y in 0..6 { dsnd[y] = softclip(s[y]); }

  for y in 0..6 {
    let d = snd[y] * d_softclip(s[y]);
    for x in 0..36 {
      dfst[x][y] = d * i[x];
    }
  }

  let mut out = 0.0;
  for y in 0..6 { out += dsnd[y] * snd[y]; }
  return out;
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[repr(align(32))]
#[derive(Clone, Debug)]
struct Parameters {
  // Material
  base  : [[f64; 2]; 4],
  pawn  : [f64; 8],
  qu_rk : f64,
  qu_bp : f64,
  qu_kt : f64,
  rk_bp : f64,
  rk_kt : f64,
  bp_kt : f64,
  bp_fs : [f64; 4],
  bp_fo : [f64; 4],
  bp_es : [f64; 4],
  bp_eo : [f64; 4],
  qu_f  : [f64; 4],
  qu_e  : [f64; 4],
  rk_f  : [f64; 4],
  rk_e  : [f64; 4],
  kt_f  : [f64; 4],
  kt_e  : [f64; 4],
  // Mobility
  scope : [[[f64; 3]; 51]; 4],
  ns1   : [[[f64; 6]; 51]; 4],
  ns2   : [[f64; 6]; 4],
  mob   : [[[f64; 4]; 3]; 4],
  fork  : [[f64; 3]; 4],
  // King Safety
  indiv    : [f64; 6],
  combo    : [[f64; 6]; 6],
  pressure : [[f64; 4]; 4],
  threat   : [[f64; 4]; 4],
  expos    : [[f64; 4]; 3],
  shield1  : [[f64; 6]; 36],
  shield2  : [f64; 6],
  // Pawn Structure
  project : [[f64; 31]; 6],
  interp  : [[[[[[[f64; 2]; 2]; 2]; 2]; 2]; 2]; 3],
  seventh : [[[[f64; 3]; 2]; 2]; 4],
  // Outposts
  r_outpost : [[f64; 3]; 3],
  b_outpost : [[f64; 3]; 3],
  n_outpost : [[f64; 3]; 3],
  // Miscellaneous
  tempo : [f64; 3],
}

fn printlist(xs : &[f64]) {
  eprint!("{:+7.3}", xs[0]);
  for x in &xs[1..] { eprint!(" {:+7.3}", x); }
}

fn printiter<'a, T>(mut xs : T) where T : Iterator<Item=&'a f64> {
  if let Some(x) = xs.next() {
    eprint!("{:+7.3}", x);
    for x in xs { eprint!(" {:+7.3}", x); }
  }
}

fn printcubic(a : &[f64; 4]) {
  let mut null = [0.0; 4];
  eprint!("{:+6.2}", cubic(0.0, a, &mut null, 1.0, 0));
  for x in 1..9 { eprint!(" {:+6.2}", cubic(x as f64 * 0.125, a, &mut null, 1.0, 0)); }
}

const NUM_PARAMS : usize = std::mem::size_of::<Parameters>() / std::mem::size_of::<f64>();

impl Parameters {
  const fn zero() -> Self {
    // We have to use this because std::mem::zeroed is not const.
    return unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
  }

  fn reset(&mut self) {
    unsafe { *self = std::mem::zeroed(); }
  }

  fn scale(&mut self, s : f64) {
    let array = unsafe { std::mem::transmute::<_, &mut [f64; NUM_PARAMS]>(self) };
    for n in 0..NUM_PARAMS { array[n] *= s; }
  }

  fn add(&mut self, other : &Self) {
    let  self_array = unsafe { std::mem::transmute::<_, &mut [f64; NUM_PARAMS]>(self)  };
    let other_array = unsafe { std::mem::transmute::<_,     &[f64; NUM_PARAMS]>(other) };
    for n in 0..NUM_PARAMS { self_array[n] += other_array[n]; }
  }

  fn add_scaled(&mut self, other : &Self, s : f64) {
    let  self_array = unsafe { std::mem::transmute::<_, &mut [f64; NUM_PARAMS]>(self)  };
    let other_array = unsafe { std::mem::transmute::<_,     &[f64; NUM_PARAMS]>(other) };
    for n in 0..NUM_PARAMS { self_array[n] += other_array[n] * s; }
  }

  fn add_square_scaled(&mut self, other : &Self, s : f64) {
    let  self_array = unsafe { std::mem::transmute::<_, &mut [f64; NUM_PARAMS]>(self)  };
    let other_array = unsafe { std::mem::transmute::<_,     &[f64; NUM_PARAMS]>(other) };
    for n in 0..NUM_PARAMS { self_array[n] += other_array[n] * other_array[n] * s; }
  }

  fn update(&mut self, means : &Self, vars : &Self, adjusted_alpha : f64) {
    let  self_array = unsafe { std::mem::transmute::<_, &mut [f64; NUM_PARAMS]>(self)  };
    let means_array = unsafe { std::mem::transmute::<_,     &[f64; NUM_PARAMS]>(means) };
    let  vars_array = unsafe { std::mem::transmute::<_,     &[f64; NUM_PARAMS]>(vars)  };
    for n in 0..NUM_PARAMS {
      self_array[n] -= means_array[n] / (epsilon + vars_array[n].sqrt()) * adjusted_alpha;
    }
  }

  fn num_flips(&self, other : &Self) -> usize {
    let  self_array = unsafe { std::mem::transmute::<_, &[f64; NUM_PARAMS]>(self)  };
    let other_array = unsafe { std::mem::transmute::<_, &[f64; NUM_PARAMS]>(other) };
    let mut flips = 0;
    for n in 0..NUM_PARAMS {
      if self_array[n].is_sign_negative() != other_array[n].is_sign_negative() { flips += 1; }
    }
    return flips;
  }

  fn show(&self) {
    eprint!("Tempo           "); printlist(&self.tempo);    eprint!("\n");
    eprint!("\n");
    eprint!("Base  Queen     "); printlist(&self.base[0]); eprint!("\n");
    eprint!("      Rook      "); printlist(&self.base[1]); eprint!("\n");
    eprint!("      Bishop    "); printlist(&self.base[2]); eprint!("\n");
    eprint!("      Knight    "); printlist(&self.base[3]); eprint!("\n");
    eprint!("      Pawn      "); printlist(&self.pawn);    eprint!("\n");
    eprint!("\n");
    eprint!("Comb  Queen     {:+7.3} {:+7.3} {:+7.3}", self.qu_rk, self.qu_bp, self.qu_kt); eprint!("\n");
    eprint!("      Rook              {:+7.3} {:+7.3}",             self.rk_bp, self.rk_kt); eprint!("\n");
    eprint!("      Bishop                    {:+7.3}",                         self.bp_kt); eprint!("\n");
    eprint!("\n");
    eprint!("Bshp  Frnd+Same "); printlist(&self.bp_fs); eprint!("  = "); printcubic(&self.bp_fs); eprint!("\n");
    eprint!("      Enem+Same "); printlist(&self.bp_es); eprint!("  = "); printcubic(&self.bp_es); eprint!("\n");
    eprint!("      Frnd+Oppo "); printlist(&self.bp_fo); eprint!("  = "); printcubic(&self.bp_fo); eprint!("\n");
    eprint!("      Enem+Oppo "); printlist(&self.bp_eo); eprint!("  = "); printcubic(&self.bp_eo); eprint!("\n");
    eprint!("\n");
    eprint!("Queen Friendly  "); printlist(&self.qu_f); eprint!("  = "); printcubic(&self.qu_f); eprint!("\n");
    eprint!("      Enemy     "); printlist(&self.qu_e); eprint!("  = "); printcubic(&self.qu_e); eprint!("\n");
    eprint!("Rook  Friendly  "); printlist(&self.rk_f); eprint!("  = "); printcubic(&self.rk_f); eprint!("\n");
    eprint!("      Enemy     "); printlist(&self.rk_e); eprint!("  = "); printcubic(&self.rk_e); eprint!("\n");
    eprint!("Kght  Friendly  "); printlist(&self.kt_f); eprint!("  = "); printcubic(&self.kt_f); eprint!("\n");
    eprint!("      Enemy     "); printlist(&self.kt_e); eprint!("  = "); printcubic(&self.kt_e); eprint!("\n");
    eprint!("\n");
    for k in 0..4 {
      for p in 0..3 {
        eprint!("Scope {k} {p}       "); printiter(self.scope[k][ 0.. 9].iter().map(|x| &x[p])); eprint!("\n");
        eprint!("                "    ); printiter(self.scope[k][ 9..18].iter().map(|x| &x[p])); eprint!("\n");
        eprint!("                "    ); printiter(self.scope[k][18..27].iter().map(|x| &x[p])); eprint!("\n");
        eprint!("                "    ); printiter(self.scope[k][27..36].iter().map(|x| &x[p])); eprint!("\n");
        eprint!("                "    ); printiter(self.scope[k][36..45].iter().map(|x| &x[p])); eprint!("\n");
        eprint!("                "    ); printiter(self.scope[k][45..51].iter().map(|x| &x[p])); eprint!("\n");
        eprint!("\n");
      }
    }
    eprint!("Queen Mob       "); printlist(&self.mob[0][0]); eprint!("\n");
    eprint!("  Legal+Safe    "); printlist(&self.mob[0][1]); eprint!("\n");
    eprint!("                "); printlist(&self.mob[0][2]); eprint!("\n");
    eprint!("Rook Mob        "); printlist(&self.mob[1][0]); eprint!("\n");
    eprint!("  Legal+Safe    "); printlist(&self.mob[1][1]); eprint!("\n");
    eprint!("                "); printlist(&self.mob[1][2]); eprint!("\n");
    eprint!("Bishop Mob      "); printlist(&self.mob[2][0]); eprint!("\n");
    eprint!("  Legal+Safe    "); printlist(&self.mob[2][1]); eprint!("\n");
    eprint!("                "); printlist(&self.mob[2][2]); eprint!("\n");
    eprint!("Knight Mob      "); printlist(&self.mob[3][0]); eprint!("\n");
    eprint!("  Legal+Safe    "); printlist(&self.mob[3][1]); eprint!("\n");
    eprint!("                "); printlist(&self.mob[3][2]); eprint!("\n");
    eprint!("\n");
    eprint!("Fork            "); printlist(&self.fork[0]); eprint!("\n");
    eprint!("                "); printlist(&self.fork[1]); eprint!("\n");
    eprint!("                "); printlist(&self.fork[2]); eprint!("\n");
    eprint!("                "); printlist(&self.fork[3]); eprint!("\n");
    eprint!("\n");
    eprint!("Indiv Threat    "); printlist(&self.indiv); eprint!("\n");
    eprint!("Combination King        {:+7.3} {:+7.3} {:+7.3} {:+7.3} {:+7.3}", self.combo[0][1], self.combo[0][2], self.combo[0][3], self.combo[0][4], self.combo[0][5]); eprint!("\n");
    eprint!("            Quen        {:+7.3} {:+7.3} {:+7.3} {:+7.3} {:+7.3}", self.combo[1][1], self.combo[1][2], self.combo[1][3], self.combo[1][4], self.combo[1][5]); eprint!("\n");
    eprint!("            Rook                {:+7.3} {:+7.3} {:+7.3} {:+7.3}",                   self.combo[2][2], self.combo[2][3], self.combo[2][4], self.combo[2][5]); eprint!("\n");
    eprint!("            Bshp                        {:+7.3} {:+7.3} {:+7.3}",                                     self.combo[3][3], self.combo[3][4], self.combo[3][5]); eprint!("\n");
    eprint!("            Kght                                {:+7.3} {:+7.3}",                                                       self.combo[4][4], self.combo[4][5]); eprint!("\n");
    eprint!("Pressure        "); printlist(&self.pressure[0]); eprint!("\n");
    eprint!("                "); printlist(&self.pressure[1]); eprint!("\n");
    eprint!("                "); printlist(&self.pressure[2]); eprint!("\n");
    eprint!("                "); printlist(&self.pressure[3]); eprint!("\n");
    eprint!("\n");
    eprint!("Threat          "); printlist(&self.threat[0]); eprint!("\n");
    eprint!("                "); printlist(&self.threat[1]); eprint!("\n");
    eprint!("                "); printlist(&self.threat[2]); eprint!("\n");
    eprint!("                "); printlist(&self.threat[3]); eprint!("\n");
    eprint!("\n");
    eprint!("King Exposure   "); printlist(&self.expos[0]); eprint!("\n");
    eprint!("                "); printlist(&self.expos[1]); eprint!("\n");
    eprint!("                "); printlist(&self.expos[2]); eprint!("\n");
    eprint!("\n");
    for f in 0..6 {
      eprint!("King Shield 1:{f} "); printiter(self.shield1[ 0.. 9].iter().map(|s| &s[f])); eprint!("\n");
      eprint!("                "  ); printiter(self.shield1[ 9..18].iter().map(|s| &s[f])); eprint!("\n");
      eprint!("                "  ); printiter(self.shield1[18..27].iter().map(|s| &s[f])); eprint!("\n");
      eprint!("                "  ); printiter(self.shield1[27..36].iter().map(|s| &s[f])); eprint!("\n");
      eprint!("\n");
    }
    eprint!("King Shield 2   "); printlist(&self.shield2); eprint!("\n");
    eprint!("\n");
    for f in 0..6 {
      eprint!("Pawn Feature {f}  "); printlist(&self.project[f][ 0.. 8]); eprint!("\n");
      eprint!("                "  ); printlist(&self.project[f][ 8..16]); eprint!("\n");
      eprint!("                "  ); printlist(&self.project[f][16..24]); eprint!("\n");
      eprint!("                "  ); printlist(&self.project[f][24..31]); eprint!("\n");
      eprint!("\n");
    }
    for b in 0..2 {
      for c in 0..2 {
      eprint!("Seventh {} {} ", ["NoB", " B "][b], ["NoB", " B "][c]);
        for f in 0..4 {
          printlist(&self.seventh[f][b][c]);
          if f < 3 { eprint!(","); }
        }
        eprint!("\n");
      }
    }
    eprint!("\n");
    eprint!("Rook Outpost    "); printlist(&self.r_outpost[0]); eprint!("\n");
    eprint!("                "); printlist(&self.r_outpost[1]); eprint!("\n");
    eprint!("                "); printlist(&self.r_outpost[2]); eprint!("\n");
    eprint!("Bishop Outpost  "); printlist(&self.b_outpost[0]); eprint!("\n");
    eprint!("                "); printlist(&self.b_outpost[1]); eprint!("\n");
    eprint!("                "); printlist(&self.b_outpost[2]); eprint!("\n");
    eprint!("Knight Outpost  "); printlist(&self.n_outpost[0]); eprint!("\n");
    eprint!("                "); printlist(&self.n_outpost[1]); eprint!("\n");
    eprint!("                "); printlist(&self.n_outpost[2]); eprint!("\n");
    eprint!("\n");
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

impl State {
  fn train(
    &self,
    ps : &Parameters,
    ds : &mut Parameters,
    _debug : bool
  ) -> f64
  {
    let mut score : f64 = 0.0;

    // Setup - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    let composite = self.sides[W] | self.sides[B];

    // let frac_piece;
    let phase_piece;
    {
      let white = self.boards[WHITE+QUEEN ].count_ones() as usize * 10
                + self.boards[WHITE+ROOK  ].count_ones() as usize * 5
                + self.boards[WHITE+BISHOP].count_ones() as usize * 3
                + self.boards[WHITE+KNIGHT].count_ones() as usize * 3;
      let black = self.boards[BLACK+QUEEN ].count_ones() as usize * 10
                + self.boards[BLACK+ROOK  ].count_ones() as usize * 5
                + self.boards[BLACK+BISHOP].count_ones() as usize * 3
                + self.boards[BLACK+KNIGHT].count_ones() as usize * 3;
      // frac_piece = [
      //   (white as f64 * 0.03125).min(1.0),
      //   (black as f64 * 0.03125).min(1.0)
      // ];
      phase_piece = ((white + black) as f64 * 0.015625).min(1.0);
    }
    let phase_pawn = (
      (self.boards[WHITE+PAWN] | self.boards[BLACK+PAWN]).count_ones() as f64 * 0.0625
    ).min(1.0);

    let phase = [1.0, phase_piece, phase_pawn];

    let mut piece_attacks : [u64; 2] = [0; 2];

    let mut geq_attacks : [[u64; 4]; 2] = [[0; 4]; 2];
    let mut thd_attacks : [u64; 2] = [0; 2];
    let mut snd_attacks : [u64; 2] = [0; 2];
    let mut fst_attacks : [u64; 2] = [0; 2];

    let mut num_king_attackers :  [u16; 2]     =  [0; 2];
    let mut king_attacker_kind : [[u16; 6]; 2] = [[0; 6]; 2];

    { // begin scope guard

      for color in 0..2 {
        let ofs = color * 8;
        let opp = ofs ^ 8;
        let no_friendly_q  =     composite & !self.boards[ofs+QUEEN];
        let no_friendly_qr = no_friendly_q & !self.boards[ofs+ROOK];
        let no_friendly_qb = no_friendly_q & !self.boards[ofs+BISHOP];
        let opp_king_area = king_destinations(self.boards[opp+KING].trailing_zeros() as usize);
        // Pieces
        for kind in 0..5 {
          let piece = ofs + kind;
          let mut sources = self.boards[piece];
          while sources != 0 {
            let src = sources.trailing_zeros() as usize;

            let attacks = piece_destinations(kind, src, composite);
            let leq = [0, 0, 1, 2, 2][kind];
            for k in leq..4 { geq_attacks[color][k] |= attacks; }

            let attacks = match kind {
              KING   =>   king_destinations(                src),
              QUEEN  =>   rook_destinations(no_friendly_qr, src)
              |         bishop_destinations(no_friendly_qb, src),
              ROOK   =>   rook_destinations(no_friendly_qr, src),
              BISHOP => bishop_destinations(no_friendly_qb, src),
              KNIGHT => knight_destinations(                src),
              _      => unreachable!()
            };
            piece_attacks[color] |= attacks;
            thd_attacks[color] |= snd_attacks[color] & attacks;
            snd_attacks[color] |= fst_attacks[color] & attacks;
            fst_attacks[color] |= attacks;
            if attacks & opp_king_area != 0 {
              king_attacker_kind[color][kind] += 1;
              num_king_attackers[color] += 1;
            }
            sources &= sources - 1;
          }
        }
        // Pawns
        let half_attacks =
          if color == 0 {
            [shift_nw(self.boards[WHITE+PAWN]), shift_ne(self.boards[WHITE+PAWN])]
          }
          else {
            [shift_sw(self.boards[BLACK+PAWN]), shift_se(self.boards[BLACK+PAWN])]
          };
        for attacks in half_attacks {
          thd_attacks[color] |= snd_attacks[color] & attacks;
          snd_attacks[color] |= fst_attacks[color] & attacks;
          fst_attacks[color] |= attacks;
        }
        if (half_attacks[0] | half_attacks[1]) & opp_king_area != 0 {
          // Ideally, we'd like to count multiplicity,
          //   but we'll accept this simplification.
          king_attacker_kind[color][PAWN] = 1;
          // Question: since pawns aren't very mobile,
          //   should they be counted in num_king_attackers?
          num_king_attackers[color] += 1;
        }
      }

    } // end scope guard

    // rebind to be immutable
    let piece_attacks = piece_attacks;
    let geq_attacks = geq_attacks;
    let thd_attacks = thd_attacks;
    let snd_attacks = snd_attacks;
    let fst_attacks = fst_attacks;
    let two_attack  = [snd_attacks[W] & !thd_attacks[W], snd_attacks[B] & !thd_attacks[B]];
    let one_attack  = [fst_attacks[W] & !snd_attacks[W], fst_attacks[B] & !snd_attacks[B]];
    let num_king_attackers = num_king_attackers;
    let king_attacker_kind = king_attacker_kind;

    // geq_attacks[color][0] := squares attacked by a king or queen
    // geq_attacks[color][1] := squares attacked by a king, queen, or rook
    // geq_attacks[color][2] := squares attacked by a king, queen, rook, bishop, or knight
    // geq_attacks[color][3] := squares attacked by a king, queen, rook, bishop, or knight
    //
    // thd_attacks[color] := squares attacked three or more times
    // snd_attacks[color] := squares attacked two or more times
    // fst_attacks[color] := squares attacked one or more times
    //
    // two_attack[color]  := squares attacked exactly twice
    // one_attack[color]  := squares attacked exactly once

    // Material  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    for color in 0..2 {

      // Base
      for kind in 0..4 {
        let board = self.boards[color*8+kind+1];
        if board == 0 { continue; }
        let n = board.count_ones() as usize;
        if n > 2 {
          let a = ps.base[kind][0];
          let b = ps.base[kind][1];
          let x = (n-1) as f64;
          score += flip(a + (b-a)*x, color);
          ds.base[kind][0] += flip(1.0-x, color);
          ds.base[kind][1] += flip(  x,   color);
        }
        else {
          score += flip(ps.base[kind][n-1], color);
          ds.base[kind][n-1] += flip(1.0, color);
        }
      }
      let board = self.boards[color*8+PAWN];
      if board != 0 {
        let n = board.count_ones() as usize;
        score += flip(ps.pawn[n-1], color);
        ds.pawn[n-1] += flip(1.0, color);
      }

      // Combination
      let ofs = color * 8;
      let opp = ofs ^ 8;

      let num_queen  = self.boards[ofs+QUEEN ].count_ones() as f64;
      let num_rook   = self.boards[ofs+ROOK  ].count_ones() as f64;
      let num_bishop = self.boards[ofs+BISHOP].count_ones() as f64;
      let num_knight = self.boards[ofs+KNIGHT].count_ones() as f64;

      score += flip(num_queen  * num_rook   * ps.qu_rk, color);
      score += flip(num_queen  * num_bishop * ps.qu_bp, color);
      score += flip(num_queen  * num_knight * ps.qu_kt, color);
      score += flip(num_rook   * num_bishop * ps.rk_bp, color);
      score += flip(num_rook   * num_knight * ps.rk_kt, color);
      score += flip(num_bishop * num_knight * ps.bp_kt, color);

      ds.qu_rk += flip(num_queen  * num_rook  , color);
      ds.qu_bp += flip(num_queen  * num_bishop, color);
      ds.qu_kt += flip(num_queen  * num_knight, color);
      ds.rk_bp += flip(num_rook   * num_bishop, color);
      ds.rk_kt += flip(num_rook   * num_knight, color);
      ds.bp_kt += flip(num_bishop * num_knight, color);

      // Adjustment
      let lt_f = (self.boards[ofs+PAWN] & LIGHT_SQUARES).count_ones() as f64;
      let dk_f = (self.boards[ofs+PAWN] &  DARK_SQUARES).count_ones() as f64;
      let lt_e = (self.boards[opp+PAWN] & LIGHT_SQUARES).count_ones() as f64;
      let dk_e = (self.boards[opp+PAWN] &  DARK_SQUARES).count_ones() as f64;
      if self.boards[ofs+BISHOP] & LIGHT_SQUARES != 0 {
        score += cubic(lt_f*0.125, &ps.bp_fs, &mut ds.bp_fs, 1.0, color);
        score += cubic(dk_f*0.125, &ps.bp_fo, &mut ds.bp_fo, 1.0, color);
        score += cubic(lt_e*0.125, &ps.bp_es, &mut ds.bp_es, 1.0, color);
        score += cubic(dk_e*0.125, &ps.bp_eo, &mut ds.bp_eo, 1.0, color);
      }
      if self.boards[ofs+BISHOP] & DARK_SQUARES != 0 {
        score += cubic(dk_f*0.125, &ps.bp_fs, &mut ds.bp_fs, 1.0, color);
        score += cubic(lt_f*0.125, &ps.bp_fo, &mut ds.bp_fo, 1.0, color);
        score += cubic(dk_e*0.125, &ps.bp_es, &mut ds.bp_es, 1.0, color);
        score += cubic(lt_e*0.125, &ps.bp_eo, &mut ds.bp_eo, 1.0, color);
      }

      let num_f = self.boards[ofs+PAWN].count_ones() as f64;
      let num_e = self.boards[opp+PAWN].count_ones() as f64;
      if self.boards[ofs+QUEEN] != 0 {
        let n = self.boards[ofs+QUEEN].count_ones() as f64;
        score += cubic(num_f*0.125, &ps.qu_f, &mut ds.qu_f, n, color);
        score += cubic(num_e*0.125, &ps.qu_e, &mut ds.qu_e, n, color);
      }
      if self.boards[ofs+ROOK] != 0 {
        let n = self.boards[ofs+ROOK].count_ones() as f64;
        score += cubic(num_f*0.125, &ps.rk_f, &mut ds.rk_f, n, color);
        score += cubic(num_e*0.125, &ps.rk_e, &mut ds.rk_e, n, color);
      }
      if self.boards[ofs+KNIGHT] != 0 {
        let n = self.boards[ofs+KNIGHT].count_ones() as f64;
        score += cubic(num_f*0.125, &ps.kt_f, &mut ds.kt_f, n, color);
        score += cubic(num_e*0.125, &ps.kt_e, &mut ds.kt_e, n, color);
      }
    }

    // Mobility & Scope  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    for color in 0..2 {
      let ofs = color * 8;
      let opp = ofs ^ 8;

      let mut any_checks : bool = false;
      let mut fst_best_capture : u8 = 0;
      let mut snd_best_capture : u8 = 0;

      // A piece can attack hanging pieces, pieces that are more valuable, or pieces that are
      //   otherwise safe to capture (not hanging but en prise).
      // However, these threats will be coming from the side waiting, which means the side to
      //   move has a chance to respond. The side to move is only likely to lose material when
      //   more than one of these are present or when a check is being made simultaneously;
      //   these are forks. (Even when a fork is not being delivered, and the opponent can
      //   respond and avoid material loss, it can be advantageous to deliver checks and
      //   threaten viable attacks because these can severely constrain the opponent; it
      //   is perhaps also good to make unviable attacks because these apply pressure to
      //   the opponent.)
      // Forks are damaged, however, (and it is generally unfavorable,) when a piece delivering
      //   a check or viable attack is committed to a friendly piece (because it is the sole
      //   defender).
      // [Of course, if the piece delivering a fork, or an attack, can simply be taken, there's
      //   no threat. This can never be the case, however, since otherwise it would not be leaf
      //   of resolving search.]
      // A single fork (by this definition – the presence of multiple winning exchanges on the
      //   board) can involve more than one attacking piece, e.g. a pawn attacking a knight and
      //   elsewhere a bishop attacking a rook.
      // A piece can be overcommitted when it is the sole defender of more than one piece; this
      //   is a structural weakness.

      // Pieces  · - · - · - · - · - · - · - · - · - · - · - · - · - · - · - · - · - · -

      let eq  = self.boards[opp+QUEEN]  ;
      let eqr = self.boards[opp+QUEEN]  | self.boards[opp+ROOK];
      let ebn = self.boards[opp+BISHOP] | self.boards[opp+KNIGHT];

      for kind in 1..5 {
        let geq_enemy = match kind {
          QUEEN  => eq  ,
          ROOK   => eqr ,
          BISHOP => eqr | ebn ,
          KNIGHT => eqr | ebn ,
          _      => unreachable!()
        };
        let gt_enemy = match kind {
          QUEEN  => 0,
          ROOK   => eq  ,
          BISHOP => eqr ,
          KNIGHT => eqr ,
          _      => unreachable!()
        };
        let piece = ofs + kind;
        let mut sources = self.boards[piece];
        while sources != 0 {
          let src = sources.trailing_zeros() as usize;
          let dests = piece_destinations(kind, src, composite);

          let mut features = [0.0; 51];

          // Defending · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·

          let no_atk_def_once   = !fst_attacks[color^1] & one_attack [color];
          let no_atk_def_many   = !fst_attacks[color^1] & snd_attacks[color];
          let atk_once_def_once =  one_attack [color^1] & one_attack [color];
          let atk_once_def_many =  one_attack [color^1] & snd_attacks[color];
          let atk_many_def_once =  snd_attacks[color^1] & one_attack [color];
          let atk_many_def_many =  snd_attacks[color^1] & snd_attacks[color];
          for k in QUEEN..=PAWN {
            let d = dests & self.boards[ofs+k];
            let kdx = (k-QUEEN) * 6;
            features[kdx+0] = fpcount(d & no_atk_def_once  );
            features[kdx+1] = fpcount(d & no_atk_def_many  );
            features[kdx+2] = fpcount(d & atk_once_def_once);
            features[kdx+3] = fpcount(d & atk_once_def_many);
            features[kdx+4] = fpcount(d & atk_many_def_once);
            features[kdx+5] = fpcount(d & atk_many_def_many);
          }

          // Attacking · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·

          features[30] = fpcount(self.boards[opp+KING] & dests);

          let hanging_mask = !fst_attacks[color^1];

          let enprise_mask = snd_attacks[color]
                           & one_attack[color^1] & geq_attacks[color^1][kind-1];

          let other_mask = !hanging_mask & !enprise_mask;

          for k in QUEEN..=PAWN {
            let d = dests & self.boards[opp+k];
            let kdx = 31 + (k-QUEEN) * 3;
            features[kdx+0] = fpcount(d & hanging_mask);
            features[kdx+1] = fpcount(d & enprise_mask);
            features[kdx+2] = fpcount(d &   other_mask);
          }

          // Empty · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·

          let empty = dests & !composite;
          let safe_unatk = !fst_attacks[color^1];
          let safe_xchg  = enprise_mask;
          let not_safe   = !safe_unatk & !safe_xchg;

          features[46] = 0.25 * fpcount(empty & safe_unatk);
          features[47] = 0.25 * fpcount(empty & safe_xchg);
          features[48] = 0.25 * fpcount(empty & not_safe);

          // Commitment  · · · · · · · · · · · · · · · · · · · · · · · · · · · ·

          let covered = dests & self.sides[color] & !self.boards[ofs+KING];
          let precarious = covered &
                         ( (one_attack[color] & one_attack[color^1]) |
                           (two_attack[color] & two_attack[color^1]) ) ;
          let num_prec = precarious.count_ones();
          let committed = num_prec == 1;
          let overcmttd = num_prec >= 2;

          features[49] = if committed { 1.0 } else { 0.0 };
          features[50] = if overcmttd { 1.0 } else { 0.0 };

          // Phased Linear Scoring (Def, Atk, Empty, Commit) · · · · · · · · · ·

          let mut scope : [f64; 3] = [0.0; 3];
          for f in 0..51 {
            for x in 0..3 {
              scope[x] += features[f] * ps.scope[kind-1][f][x];
              ds.scope[kind-1][f][x] += flip(features[f] * phase[x], color);
            }
          }
          score += flip(
            scope[0] + scope[1]*phase_piece + scope[2]*phase_pawn, color
          );

          // Unphased Network Scoring (Def, Atk, Empty, Commit)  · · · · · · · ·

          let mut s = [0.0; 6];
          let mut a = [0.0; 6];

          for x in 0..51 {
            for y in 0..6 { s[y] += features[x] * ps.ns1[kind-1][x][y]; }
          }
          for y in 0..6 { a[y] = softclip(s[y]); }

          let mut subscore = 0.0;
          for y in 0..6 { subscore += a[y] * ps.ns2[kind-1][y]; }
          score += flip(subscore, color);

          for y in 0..6 {
            let d = ps.ns2[kind-1][y] * d_softclip(s[y]);
            for x in 0..51 { ds.ns1[kind-1][x][y] += flip(d * features[x], color); }
          }
          for y in 0..6 { ds.ns2[kind-1][y] += flip(a[y], color); }

          // Forks · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
          //   In theory, a piece should be ineligible to participate in
          //   delivering a fork if it's committed to defending a piece,
          //   but empirically this made no difference (and in fact ever
          //   so slightly increased the training error).

          any_checks = any_checks || (self.boards[opp+KING] & dests != 0);

          let mut threatened = dests
            & self.sides[color^1]
            & !self.boards[opp+KING]
            & (hanging_mask | enprise_mask);
          while threatened != 0 {
            let x = threatened.trailing_zeros() as usize;
            let mut v = self.squares[x].kind() as u8;
            if v == 0 { panic!(); }
            if v > 5 { panic!(); }
            if v > 3 { v -= 1; }
            v = 5 - v;  // PAWN: 1, KNIGHT: 2, BISHOP: 2, ROOK: 3, QUEEN: 4
            if v > fst_best_capture {
              snd_best_capture = fst_best_capture;
              fst_best_capture = v;
            }
            else if v > snd_best_capture {
              snd_best_capture = v;
            }
            threatened &= threatened - 1;
          }

          let trade_mask = gt_enemy & !hanging_mask & !enprise_mask;
          threatened = dests
            & self.sides[color^1]
            & !self.boards[opp+KING]
            & trade_mask;
          while threatened != 0 {
            debug_assert!(kind != QUEEN);
            let x = threatened.trailing_zeros() as usize;
            let v =
              if kind == ROOK { 3 }
              else if self.squares[x].kind() == QUEEN { 3 } else { 1 };
            if v > fst_best_capture {
              snd_best_capture = fst_best_capture;
              fst_best_capture = v;
            }
            else if v > snd_best_capture {
              snd_best_capture = v;
            }
            threatened &= threatened - 1;
          }

          // Legal and safe  · · · · · · · · · · · · · · · · · · · · · · · · · ·

          let legal = dests & !self.sides[color];
          let safe = safe_unatk | safe_xchg | geq_enemy;
          let legal_safe = legal & safe;
          let max_moves = [27.0, 14.0, 13.0, 8.0][kind-1];
          let frac_mob = legal_safe.count_ones() as f64 / max_moves;
          for p in 0..3 {
            score += cubic(
              frac_mob, &ps.mob[kind-1][p], &mut ds.mob[kind-1][p],
              phase[p], color
            );
          }

          // Rook-specific · · · · · · · · · · · · · · · · · · · · · · · · · · ·

          // TODO

          sources &= sources - 1;
        }
      }

      // Pawns - · - · - · - · - · - · - · - · - · - · - · - · - · - · - · - · - · - · -

      // TODO

      // Pawn Forks  · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·

      let dests = pawn_attacks(Color::from_usize(color), self.boards[ofs+PAWN]);
      any_checks = any_checks || (self.boards[opp+KING] & dests != 0);
      let hanging_mask = !fst_attacks[color^1];
      let enprise_mask = snd_attacks[color] & one_attack[color^1];
      let mut threatened = dests
        & self.sides[color^1]
        & !self.boards[opp+KING]
        & (hanging_mask | enprise_mask);
      while threatened != 0 {
        let x = threatened.trailing_zeros() as usize;
        let mut v = self.squares[x].kind() as u8;
        if v == 0 { panic!(); }
        if v > 5 { panic!(); }
        if v > 3 { v -= 1; }
        v = 5 - v;  // PAWN: 1, KNIGHT: 2, BISHOP: 2, ROOK: 3, QUEEN: 4
        if v > fst_best_capture {
          snd_best_capture = fst_best_capture;
          fst_best_capture = v;
        }
        else if v > snd_best_capture {
          snd_best_capture = v;
        }
        threatened &= threatened - 1;
      }
      let gt_enemy  = eqr | ebn ;
      let trade_mask = gt_enemy & !hanging_mask & !enprise_mask;
      threatened = dests
        & self.sides[color^1]
        & !self.boards[opp+KING]
        & trade_mask;
      while threatened != 0 {
        let x = threatened.trailing_zeros() as usize;
        let v = if self.squares[x].kind() == QUEEN { 4 } else { 2 };
        if v > fst_best_capture {
          snd_best_capture = fst_best_capture;
          fst_best_capture = v;
        }
        else if v > snd_best_capture {
          snd_best_capture = v;
        }
        threatened &= threatened - 1;
      }

      // Fork Scoring  - · - · - · - · - · - · - · - · - · - · - · - · - · - · - · - · -

      let best_capture = if any_checks { fst_best_capture } else { snd_best_capture };
      if best_capture > 0 {
        let idx = best_capture as usize - 1;
        let fork = &ps.fork[idx];
        score += flip(fork[0] + fork[1]*phase_piece + fork[2]*phase_pawn, color);
        ds.fork[idx][0] += flip(1.0,         color);
        ds.fork[idx][1] += flip(phase_piece, color);
        ds.fork[idx][2] += flip(phase_pawn,  color);
      }
    }

    // King safety - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    for color in 0..2 {
      let ofs = color * 8;
      let opp = ofs ^ 8;

      let king_sq = self.boards[opp+KING].trailing_zeros() as usize;
      let king_area = king_destinations(king_sq);
      let area_size = king_area.count_ones() as usize;

      let num_attackers =  num_king_attackers[color] as usize;
      let attacker_kind = &king_attacker_kind[color];

      if num_attackers < 2 { continue; }

      // This provides a way to moderate or highlight pressure on the king even
      //   when all the squares around the king are category 3–6 squares (for
      //   example, when the base score is high, attenuating a case where the
      //   attackers' vision overlaps and only a single square is attacked more
      //   than a case where the same collection pieces attacks a larger number
      //   of squares).

      let num_attackers = std::cmp::min(5, num_attackers) - 2;

      let num_attacked = (fst_attacks[color] & king_area).count_ones() as usize;
      let frac_attacked = norm_area(area_size, num_attacked);

      let pressure = ps.pressure[num_attackers][frac_attacked];

      // Squares around the king fall into six categories:
      //   1 attacked once by the defense and twice or more by the offense
      //   2 attacked once by the defense and once by the offense
      //   3 attacked once by the defense
      //   4 attacked twice or more by the defense
      //   5 attacked twice or more by the defense and once by the offense
      //   6 attacked twice or more by the defense as well as the offense
      //
      // Category 1 squares are especially deadly, because they are places near
      //   the king where pieces can land. Category 2 squares are escape squares
      //   that have been cut off and indicate that a mating net is being woven.
      //   Category 3, 4, and 5 squares are not especially problematic, and
      //   category 6 squares are best left to resolving search for assessment.

      let cat1      = one_attack[color^1] & snd_attacks[color] & king_area;
      let num_cat1  = cat1.count_ones() as usize;
      let frac_cat1 = norm_area(area_size, num_cat1);

      let cat2      = one_attack[color^1] & one_attack[color] & king_area;
      let num_cat2  = cat2.count_ones() as usize;
      let frac_cat2 = norm_area(area_size, num_cat2);

      let threat = ps.threat[frac_cat1][frac_cat2];

      // We do the base calculation last so that we have the pressure and threat
      //   terms handy for gradient updates.

      let g = flip(pressure * threat, color);

      let mut base = 0.0;
      for p in 0..5 {
        if attacker_kind[p] == 0 {
          continue;
        }
        let mut combo = 0.0;
        if attacker_kind[p] > 1 {
          combo += ps.combo[p][p];
          ds.combo[p][p] += g;
        }
        for q in (p+1)..6 {
          if attacker_kind[q] != 0 {
            combo += ps.combo[p][q];
            ds.combo[p][q] += g;
          }
        }
        base += ps.indiv[p] * (attacker_kind[p] as f64) + combo;
        ds.indiv[p] += attacker_kind[p] as f64 * g;
      }
      if attacker_kind[PAWN] != 0 { base += ps.indiv[PAWN]; ds.indiv[PAWN] += g; }

      score += flip(base * pressure * threat, color);

      ds.threat[frac_cat1][frac_cat2] += flip(base * pressure, color);
      ds.pressure[num_attackers][frac_attacked] += flip(base * threat, color);
    }

    // King exposure - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    // When the king has many squares around it, there are many ways to get to the king
    //   – from in front, from behind, or from the side. When the king is near an edge
    //   or corner, this attack surface is reduced. I would have expected there to also
    //   be some risk when the king is actually on the edge or in the corner, because its
    //   mobility is reduced and the king has fewer escape squares when it is being attacked,
    //   but from limited evidence this doesn't seem to be significant.
    //
    // In any case, we can perhaps quantify king exposure by considering area around the
    //   king, about a knight's distance in radius, and counting the number of squares
    //   that are on the board.
    //
    //  · · · · · · · ·    · · · · · · · ·    · · · · · · · ·    · · · · · · · ·
    //  · · · · · · · ·    · · · · · · · ·    · · · · · · · ·    · · · · · · · ·
    //  · · × × × × × ·    · · · · · · · ·    · · · · · · · ·    · · · · · · · ·
    //  · · × × × × × ·    · · · · · · · ·    · · · · · · · ·    · · · · · · · ·
    //  · · × × K × × ·    · · · × × × × ×    · · · · · · · ·    · · · · · · · ·
    //  · · × × × × × ·    · · · × × × × ×    · · · · × × × ×    · · · · · × × ×
    //  · · × × × × × ·    · · · × × K × ×    · · · · × × × ×    · · · · · × × ×
    //  · · · · · · · ·    · · · × × × × ×    · · · · × × K ×    · · · · · × × K
    //       24/24              19/24              11/24               8/24

    for color in 0..2 {
      let ofs = color * 8;
      let src = self.boards[ofs+KING].trailing_zeros() as usize;
      let rank = (src / 8) as usize;
      let file = (src % 8) as usize;
      let width  = [3, 4, 5, 5, 5, 5, 4, 3][rank];
      let height = [3, 4, 5, 5, 5, 5, 4, 3][file];
      let exposure = width * height - 9;  // 0 to 16 inclusive
      let frac_exposed = exposure as f64 / 16.0;
      for p in 0..3 {
        score += cubic(frac_exposed, &ps.expos[p], &mut ds.expos[p], phase[p], color);
      }
    }

    // King-pawn dynamics (shielding, storming, shepherding) - - - - - - - - - - - - - - - - - -

    // This is just a very, very small neural network with 36 inputs and a hidden layer with
    //   6 neurons. The inputs are binary: whether each of the 14 squares around the king is
    //   occupied by a friendly pawn or enemy pawn as well as whether an edge of the board is
    //   immediately to the left, right, front, or back of the king or one square away. Here
    //   are the inputs in the starting position:
    //
    //     · · ·    · · ·
    //     · · ·    · · ·     edge imm left:  no    edge one left:  no
    //     × × ×    · · ·     edge imm right: no    edge one right: no
    //     · K ·    · K ·     edge imm front: no    edge one front: no
    //     · · ·    · · ·     edge imm back:  yes   edge one back:  no
    //      pov      opp
    //
    // If a castling right exists, this is also computed as if the king were in its castled
    //   position, and the best score is used.

    for color in 0..2 {
      let pov_pawns;
      let opp_pawns;
      let king_idx;
      if color == 0 {
        pov_pawns = self.boards[WHITE+PAWN];
        opp_pawns = self.boards[BLACK+PAWN];
        king_idx  = self.boards[WHITE+KING].trailing_zeros() as usize;
      }
      else {
        pov_pawns =   bswap(self.boards[BLACK+PAWN]);
        opp_pawns =   bswap(self.boards[WHITE+PAWN]);
        king_idx  = vmirror(self.boards[BLACK+KING].trailing_zeros() as usize);
      }

      let mut d1 = [[0.0; 6]; 36];
      let mut d2 =  [0.0; 6];
      let mut subscore =
        king_shield(pov_pawns, opp_pawns, king_idx, &ps.shield1, &ps.shield2, &mut d1, &mut d2);

      let rights = (if color == 0 { self.rights } else { self.rights >> 2 }) & 0b0011;
      if rights & 1 != 0 {
        let mut kd1 = [[0.0; 6]; 36];
        let mut kd2 =  [0.0; 6];
        let altscore =
          king_shield(pov_pawns, opp_pawns, 6, &ps.shield1, &ps.shield2, &mut kd1, &mut kd2);
        if altscore > subscore {
          subscore = altscore;
          d1 = kd1;
          d2 = kd2;
        }
      }
      if rights & 2 != 0 {
        let mut qd1 = [[0.0; 6]; 36];
        let mut qd2 =  [0.0; 6];
        let altscore =
          king_shield(pov_pawns, opp_pawns, 2, &ps.shield1, &ps.shield2, &mut qd1, &mut qd2);
        if altscore > subscore {
          subscore = altscore;
          d1 = qd1;
          d2 = qd2;
        }
      }

      score += flip(subscore, color);

      for x in 0..36 { for y in 0..6 { ds.shield1[x][y] += flip(d1[x][y], color); } }
      for y in 0..6 { ds.shield2[y] += flip(d2[y], color); }
    }

    // Pawns - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //
    //           Isolated: no friendly pawns on adjacent files
    //   Weakly Connected: one or more friendly pawns on adjacent files
    // Strongly Connected: defended by a friendly pawn
    //
    //                     [if a penalty is assessed for isolation and a bonus awarded for
    //                     strong connexion, there's no need to award a bonus for weak
    //                     connexion, since a pawn is necessarily weakly connected when it
    //                     isn't isolated or strongly connected]
    //
    //             Sentry: controls square in front of an opposing pawn
    //             Passed: no enemy pawns on adjacent files are ahead (nor directly ahead)
    //
    //                     [there's no need to award a bonus for sentries, since a sentry
    //                     prevents an opponent from having a passed pawn]
    //
    //                     [pawns on the seventh rank are definitionally passed pawns]
    //
    //   Candidate Passed: no enemy pawns directly ahead and number of enemy pawns on adjacent
    //                     files covering the file ahead is less than or equal to the number of
    //                     friendly pawns on adjacent files that cover the stop square (or could
    //                     advance to do so)
    //
    //           Backward: Stop square controlled by enemy pawn and stop square not defended by
    //                       friendly pawn
    //                     [really this second condition should be "stop square not defensible",
    //                     "defensible" meaning that it is either defended by a friendly pawn or
    //                     that there is a friendly pawn on the adjacent file next to or behind
    //                     this pawn that is not blocked from walking up next to this pawn]
    //
    //      Semi-backward: supported by a neighbor, but the neighbor can't advance because it
    //                     is blocked by an enemy pawn; or, supported by a neighbor, but that
    //                     neighbor is itself backward
    //
    //      Over-advanced: friendly neighbor cannot walk forward to defend because it's blocked
    //                     by an enemy pawn
    //
    //                     [this is similar to the first case of semi-backwardness, but over-
    //                     advancement is about a pawn being defensible rather than its stop
    //                     square]
    //
    //               Hole: square that cannot possibly be covered by your own pawns (because
    //                     they have advanced); often covered by enemy pawns; often restricted
    //                     to the center 4×4 squares
    //
    //                     [a hole is a potential outpost square for the oppponent; holes are
    //                     bad near the king; holes are bad when the opponent has minor pieces;
    //                     holes are bad when the opponent has more pieces than you because
    //                     then your pieces cannot cover the hole]

    for color in 0..2 {

      let pov_side;
      let opp_side;
      let mut rectified = [0; 16];
      if color == 0 {
        pov_side = self.sides[W];
        opp_side = self.sides[B];
        rectified = self.boards;
      }
      else {
        pov_side = bswap(self.sides[B]);
        opp_side = bswap(self.sides[W]);
        for c in 0..2 {
          let ofs =  c  * 8;
          let opp = ofs ^ 8;
          for k in 0..8 { rectified[ofs+k] = bswap(self.boards[opp+k]); }
        }
      }
      let rectified = rectified;
      const OFS : usize = 0;
      const OPP : usize = 8;

      let pov_pawns = rectified[OFS+PAWN];
      let opp_pawns = rectified[OPP+PAWN];

      let pov_cover = pawn_attacks(Color::White, pov_pawns);
      let opp_cover = pawn_attacks(Color::Black, opp_pawns);

      let mut sources = pov_pawns;
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        let stop = (1u64 << src) << 8;

        let rank = (src / 8) as usize;
        let file = (src % 8) as usize;

        let row      = RANK_1 << (rank*8);
        let column   = FILE_A << file;

        let adjacent = ((column & !FILE_H) << 1) | ((column & !FILE_A) >> 1);

        let above    = (!RANK_1) << (rank*8);
        let farabove = above << 8;
        let wayabove = above << 16;

        let below    = !((!0) << (rank*8));
        let farbelow = below >> 8;

        let fwd_diag = adjacent & (row << 8);

        //  PP = source square
        //  ·· = stop square
        //
        //  |XX|  |XX|     |XX|  |XX|     |XX|  |XX|     |  |  |  |     |  |  |  |     |  |  |  |
        //  +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+
        //  |  |  |  |     |XX|  |XX|     |XX|  |XX|     |  |  |  |     |  |  |  |     |  |  |  |
        //  +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+
        //  |  |··|  |     |  |··|  |     |XX|··|XX|     |  |··|  |     |  |··|  |     |  |··|  |
        //  +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+
        //  |  |PP|  |     |  |PP|  |     |  |PP|  |     |XX|PP|XX|     |  |PP|  |     |  |PP|  |
        //  +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+
        //  |  |  |  |     |  |  |  |     |  |  |  |     |XX|  |XX|     |XX|  |XX|     |  |  |  |
        //  +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+     +--+--+--+
        //  |  |  |  |     |  |  |  |     |  |  |  |     |XX|  |XX|     |XX|  |XX|     |XX|  |XX|
        //  adjacent &     adjacent &     adjacent &     adjacent &     adjacent &     adjacent &
        //   wayabove       farabove         above         !above          below        farbelow

        let mut basic = [0.0; 31];

        // COLUMN  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        basic[ 0] = unitize(column & farabove & pov_pawns); // doubling a friendly pawn
        basic[ 1] = unitize(column & farabove & opp_pawns); // blocked by an enemy pawn
        basic[ 2] = unitize(stop              & pov_pawns); // rammed by a friendly pawn
        basic[ 3] = unitize(stop              & opp_pawns); // rammed by an enemy pawn
        basic[ 4] = unitize(stop & pov_side & !pov_pawns);  // rammed by a friendly piece
        basic[ 5] = unitize(stop & opp_side & !opp_pawns);  // rammed by an enemy piece

        // SOURCE SQUARE - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        // pawns on adjacent files that are currently defending this pawn
        //   and might be able to walk up to defend the stop square
        let supports = adjacent & (row >> 8) & pov_pawns;
        let obstruct = (row & opp_pawns) >> 8;
        let guard    = (row & opp_cover) >> 8;

        basic[ 6] = unitize(supports & !obstruct & !guard);
        basic[ 7] = unitize(supports & !obstruct &  guard);
        basic[ 8] = unitize(supports &  obstruct & !guard);
        basic[ 9] = unitize(supports &  obstruct &  guard);

        basic[10] = multize(supports & !obstruct & !guard);
        basic[11] = multize(supports & !obstruct &  guard);
        basic[12] = multize(supports &  obstruct & !guard);
        basic[13] = multize(supports &  obstruct &  guard);

        // pawns on adjacent files that are currently attacking this pawn
        basic[14] = unitize(fwd_diag & opp_pawns);
        basic[15] = multize(fwd_diag & opp_pawns);

        // friendly pawns on adjacent files that can walk up to defend this pawn
        let anchors  = adjacent & farbelow & pov_pawns;
        let obstruct = sweep_s(below & opp_pawns);
        let guard    = sweep_s(below & opp_cover);

        basic[16] = fpcount(anchors & !obstruct & !guard);  // able
        basic[17] = fpcount(anchors & !obstruct &  guard);  // somewhat likely
        basic[18] = fpcount(anchors &  obstruct & !guard);  // maybe
        basic[19] = fpcount(anchors &  obstruct &  guard);  // very unlikely

        // STOP SQUARE - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        // pawns on adjacent files that are currently defending or attacking the stop square
        basic[20] = fpcount(adjacent & row & pov_pawns);
        basic[21] = fpcount(adjacent & (row << 16) & opp_pawns);

        // enemy pawns on adjacent files that can walk down to attack the stop square
        let prevents = adjacent & wayabove & opp_pawns;
        let obstruct = sweep_n(farabove & pov_pawns);
        let guard    = sweep_n(farabove & pov_cover);

        basic[22] = fpcount(prevents & !obstruct & !guard); // able
        basic[23] = fpcount(prevents & !obstruct &  guard); // somewhat likely
        basic[24] = fpcount(prevents &  obstruct & !guard); // maybe
        basic[25] = fpcount(prevents &  obstruct &  guard); // very unlikely

        // BIAS, KING TROPISM, AND ADVANCEMENT - - - - - - - - - - - - - - - - -

        let pov_king = (rectified[OFS+KING].trailing_zeros() as u8) % 8;
        let opp_king = (rectified[OPP+KING].trailing_zeros() as u8) % 8;

        basic[26] = diff(file as u8, pov_king)  as f64 * ONESEVENTH;
        basic[27] = diff(file as u8, opp_king)  as f64 * ONESEVENTH;
        basic[28] = std::cmp::min(file, 7-file) as f64 * ONETHIRD;
        basic[29] = (rank - 1)                  as f64 * 0.2;

        basic[30] = 1.0;  // bias (constant offset)

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        // projection from 30+1 dimensions down to 6
        let mut complex   = [0.0; 6];
        let mut d_complex = [0.0; 6];
        for f in 0..31 {
          complex[0] += basic[f] * ps.project[0][f];
          complex[1] += basic[f] * ps.project[1][f];
          complex[2] += basic[f] * ps.project[2][f];
          complex[3] += basic[f] * ps.project[3][f];
          complex[4] += basic[f] * ps.project[4][f];
          complex[5] += basic[f] * ps.project[5][f];
        }

        // soft clamp to the unit hypercube
        for h in 0..6 {
          let dot = complex[h];
            complex[h] =   softclip(dot);
          d_complex[h] = d_softclip(dot);
        }

        // hexalinear interpolation
        let mut lerp = [0.0; 3];
        for p in 0..3 {
          let mut s0 = 0.0;
          let mut s1 = 0.0;
          let mut s2 = 0.0;
          let mut s3 = 0.0;
          for a in 0..2 {
            let x0 = if a == 0 { complex[0] } else { 1.0 - complex[0] };
            for b in 0..2 {
              let x1 = if b == 0 { complex[1] } else { 1.0 - complex[1] };
              let y0 = x0 * x1;
              for c in 0..2 {
                let x2 = if c == 0 { complex[2] } else { 1.0 - complex[2] };
                for d in 0..2 {
                  let x3 = if d == 0 { complex[3] } else { 1.0 - complex[3] };
                  let y1 = x2 * x3;
                  let z  = y0 * y1;
                  let k0 = (    complex[4]     *     complex[5]    ) * z;
                  let k1 = (    complex[4]     * (1.0 - complex[5])) * z;
                  let k2 = ((1.0 - complex[4]) *     complex[5]    ) * z;
                  let k3 = ((1.0 - complex[4]) * (1.0 - complex[5])) * z;
                  s0 += k0 * ps.interp[p][a][b][c][d][0][0];
                  s1 += k1 * ps.interp[p][a][b][c][d][0][1];
                  s2 += k2 * ps.interp[p][a][b][c][d][1][0];
                  s3 += k3 * ps.interp[p][a][b][c][d][1][1];
                  ds.interp[p][a][b][c][d][0][0] += flip(k0 * phase[p], color);
                  ds.interp[p][a][b][c][d][0][1] += flip(k1 * phase[p], color);
                  ds.interp[p][a][b][c][d][1][0] += flip(k2 * phase[p], color);
                  ds.interp[p][a][b][c][d][1][1] += flip(k3 * phase[p], color);
                }
              }
            }
          }
          lerp[p] = (s0 + s1) + (s2 + s3);
        }
        score += flip(lerp[0] + lerp[1]*phase_piece + lerp[2]*phase_pawn, color);

        // gradients of the projection
        for h in 0..6 {
          let mut w = 0.0;
          for p in 0..3 {
            let mut s = [[0.0; 2]; 2];
            for a in 0..2 {
              let x0 =
                if h == 0 { if a == 0 {     1.0    } else {       -1.0       } }
                else      { if a == 0 { complex[0] } else { 1.0 - complex[0] } };
              for b in 0..2 {
                let x1 =
                  if h == 1 { if b == 0 {     1.0    } else {       -1.0       } }
                  else      { if b == 0 { complex[1] } else { 1.0 - complex[1] } };
                let y0 = x0 * x1;
                for c in 0..2 {
                  let x2 =
                    if h == 2 { if c == 0 {     1.0    } else {       -1.0       } }
                    else      { if c == 0 { complex[2] } else { 1.0 - complex[2] } };
                  for d in 0..2 {
                    let x3 =
                      if h == 3 { if d == 0 {     1.0    } else {       -1.0       } }
                      else      { if d == 0 { complex[3] } else { 1.0 - complex[3] } };
                    let y1 = x2 * x3;
                    for e in 0..2 {
                      let x4 =
                        if h == 4 { if e == 0 {     1.0    } else {       -1.0       } }
                        else      { if e == 0 { complex[4] } else { 1.0 - complex[4] } };
                      for f in 0..2 {
                        let x5 =
                          if h == 5 { if f == 0 {     1.0    } else {       -1.0       } }
                          else      { if f == 0 { complex[5] } else { 1.0 - complex[5] } };
                        let y2 = x4 * x5;
                        s[e][f] += (y0 * y1) * (y2 * ps.interp[p][a][b][c][d][e][f]);
                      }
                    }
                  }
                }
              }
            }
            w += ((s[0][0] + s[0][1]) + (s[1][0] + s[1][1])) * phase[p];
          }
          let d = flip(w * d_complex[h], color);
          for f in 0..31 { ds.project[h][f] += basic[f] * d; }
        }

        // SEVENTH RANK  - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if rank == 6 {
          let edge_dist = std::cmp::min(file, 7-file);
          let promo_color = [LIGHT_SQUARES, DARK_SQUARES][file & 1];
          let pov_bishop = if rectified[OFS+BISHOP] & promo_color != 0 { 1 } else { 0 };
          let opp_bishop = if rectified[OPP+BISHOP] & promo_color != 0 { 1 } else { 0 };
          let bonus = &ps.seventh[edge_dist][pov_bishop][opp_bishop];
          score += flip(bonus[0] + bonus[1]*phase_piece + bonus[2]*phase_pawn, color);

          ds.seventh[edge_dist][pov_bishop][opp_bishop][0] += flip(1.0        , color);
          ds.seventh[edge_dist][pov_bishop][opp_bishop][1] += flip(phase_piece, color);
          ds.seventh[edge_dist][pov_bishop][opp_bishop][2] += flip(phase_pawn , color);
        }

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        // These aren't part of the projection and multilinear interpolation
        //   because we want to be able to cache as much of pawn evaluation
        //   as possible. See instead the mobility section.

        // defending friendly pieces
        //   basic[26] = unitize(fwd_diag & rectified[OFS+QUEEN ]);
        //   basic[27] = unitize(fwd_diag & rectified[OFS+ROOK  ]);
        //   basic[28] = unitize(fwd_diag & rectified[OFS+BISHOP]);
        //   basic[29] = unitize(fwd_diag & rectified[OFS+KNIGHT]);

        // attacking enemy pieces
        //   basic[30] = unitize(fwd_diag & rectified[OPP+QUEEN ]);
        //   basic[31] = unitize(fwd_diag & rectified[OPP+ROOK  ]);
        //   basic[32] = unitize(fwd_diag & rectified[OPP+BISHOP]);
        //   basic[33] = unitize(fwd_diag & rectified[OPP+KNIGHT]);

        // forking
        //   basic[34] = multize(fwd_diag & opp_side & !opp_pawns);

        sources &= sources - 1;
      }
    }

    // Pieces on outpost squares - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //
    // A stable outpost is a square occupied by a minor piece that is covered by a friendly
    //   pawn and that cannot be attacked by enemy pawns.
    //
    // An provocative outpost is a square occupied by a minor piece that is covered by a
    //   friendly pawn that provokes the enemy pawn to walk forward (abandoning its defense
    //   of a pawn on a neighboring semiopen file), e.g. Nd5 in
    //
    //     r4rk1/ppp2ppp/3p2n1/8/4P3/2N5/PPP2PPP/2KRR3 w - - 0 1
    //
    //   which provokes c6, which will leave d6 defended and vulnerable to the rook on d1 when
    //   the knight moves away.
    //
    //   Here are two more examples:
    //
    //     · · · · · · · ·    · · · · · · · ·
    //     · · D · y · . ·    x · D · y · . ·    M is a minor piece
    //     · · o P x . · ·    · P o · x . · ·    P is an enemy pawn
    //     · · · M · · · ·    · · · M · · · ·    D is an enemy pawn defending P
    //     · · · · · · · ·    · · · · · · · ·    x must not be an enemy pawn
    //     · · · · · · · ·    · · · · · · · ·    o must not be occupied
    //     · · · · · · · ·    · · · · · · · ·    y must not be an enemy pawn if
    //     · · · · · · · ·    · · · · · · · ·        x is not a friendly pawn
    //           ^              ^
    //        semiopen        semiopen
    //
    // Although interesting and valuable, I'd like the evaluation to contain generalizations,
    //   and this is a rather special case, so we will only consider stable outposts (and not
    //   provocative outposts).
    //
    // We might say player X's territory consists of the squares that X controls, which roughly
    //   means squares that X attacks more times than their opponent, Y. Or perhaps more accu-
    //   rately, squares that X attacks that Y cannot visit because X would win a favorable
    //   exchange, or maybe squares that X can visit safely (because if Y tried to capture,
    //   X would win a favorable exchange), which is a subtly different definition. Territory
    //   maybe also includes squares that X could easily control even if X does not control
    //   them currently.
    //
    // We're interested in squares that allow a bishop or knight to apply pressure and restrict
    //   the movement of the opponent's pieces. Roughly, then, that means attacking the oppo-
    //   nent's territory. We use a simplified definition: when determining whether the square
    //   that one of player X's pieces, pX, is currently on is an outpost, player Y's territory
    //   includes
    //   • squares where Y's pieces reside,
    //   • squares around Y's king, and
    //   • squares that Y's pieces may move to that are attacked more times by Y than X,
    //     including the attacks by the piece(s) that can move to that square but not including
    //     any attacks by pX, or equivalently, for the purpose of determining only whether the
    //     squares that pX attacks are part of Y's territory: squares that Y's pieces may move
    //     to that are attacked an equal or greater number of times by Y than X, including the
    //     attacks by the piece(s) that can move to that square and including any attacks by pX.

    let territory;
    {
      let w_king = king_destinations(self.boards[WHITE+KING].trailing_zeros() as usize);
      let b_king = king_destinations(self.boards[BLACK+KING].trailing_zeros() as usize);
      let w_pwns = (self.boards[WHITE+PAWN] << 8) & !composite;
      let b_pwns = (self.boards[BLACK+PAWN] >> 8) & !composite;
      let w_dest = (piece_attacks[W] | w_pwns) & !(snd_attacks[B] & one_attack[W]);
      let b_dest = (piece_attacks[B] | b_pwns) & !(snd_attacks[W] & one_attack[B]);
      territory = [
        self.sides[W] | w_king | w_dest,
        self.sides[B] | b_king | b_dest,
      ];
    }

    for color in 0..2 {
      let ofs = color * 8;

      let white_pawns = self.boards[WHITE+PAWN];
      let black_pawns = self.boards[BLACK+PAWN];
      let side_to_move = if color == 0 { white_pawns } else { bswap(black_pawns) };
      let side_waiting = if color == 0 { black_pawns } else { bswap(white_pawns) };

      let mid_pieces = self.boards[ofs+ROOK] | self.boards[ofs+BISHOP] | self.boards[ofs+KNIGHT];
      let pawn_covers = pawn_attacks(Color::from_usize(color), self.boards[ofs+PAWN]);

      let mut sources = mid_pieces & pawn_covers;
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;
        sources &= sources - 1;

        let kind = self.squares[src].kind();
        let dests = match kind {
          ROOK   =>   rook_destinations(composite, src),
          BISHOP => bishop_destinations(composite, src),
          KNIGHT => knight_destinations(src),
          _      => unreachable!()
        };
        let hits = dests & territory[color^1];

        if hits == 0 { continue; }

        let num_hits = std::cmp::min(hits.count_ones() as usize, 3) - 1;

        let above = FILE_A << (if color == 0 { src } else { vmirror(src) });
        let above_adj = shift_nw(above) | shift_ne(above);

        if side_waiting & above_adj & !sweep_n(side_to_move & above_adj) != 0 { continue; }

        let outpost = match kind {
          ROOK   => &ps.r_outpost[num_hits],
          BISHOP => &ps.b_outpost[num_hits],
          KNIGHT => &ps.n_outpost[num_hits],
          _      => unreachable!()
        };
        score += flip(outpost[0] + outpost[1]*phase_piece + outpost[2]*phase_pawn, color);

        let d = match kind {
          ROOK   => &mut ds.r_outpost[num_hits],
          BISHOP => &mut ds.b_outpost[num_hits],
          KNIGHT => &mut ds.n_outpost[num_hits],
          _      => unreachable!()
        };
        d[0] += flip(1.0        , color);
        d[1] += flip(phase_piece, color);
        d[2] += flip(phase_pawn , color);
      }
    }

    // Tempo - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    let tempo_score = ps.tempo[0] + ps.tempo[1]*phase_piece + ps.tempo[2]*phase_pawn;
    score += flip(tempo_score, self.turn as usize);
    ds.tempo[0] += flip(1.0,         self.turn as usize);
    ds.tempo[1] += flip(phase_piece, self.turn as usize);
    ds.tempo[2] += flip(phase_pawn,  self.turn as usize);

    return score;
  }

  pub fn evaluate_hce(&self) -> f64
  {
    let mut ds = Parameters::zero();
    return self.train(&DFLT_PARAMS, &mut ds, false);
  }

  pub fn debug_hce(&self) -> f64
  {
    let mut ds = Parameters::zero();
    return self.train(&DFLT_PARAMS, &mut ds, true);
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

const alpha   : f64 = 0.0625;   // initial learning rate
const beta    : f64 = 0.875 ;   // means
const gamma   : f64 = 0.9375;   // variances
const epsilon : f64 = 1.0e-6;

const NUM_THREADS : usize = 80;
const INIT_BATCH_SIZE : usize = 23_347_200; // must be a multiple of NUM_THREADS

// 2⁶ × 3 × 5 = 960 should cover most thread counts. Our dataset has 749 884 210 positions,
//   so if we want five doublings, the batch size needs to be 749 884 210 / (2⁵ × 960) × 960
//   = 23 433 600 or less. We round down slightly to 23 347 200, which is a bit rounder.

static mut DATASET : Vec<(State, f64)> = Vec::new();
static mut PARAMS : Parameters = Parameters::zero();

static mut WORKER_DERIVS : Vec<Parameters> = Vec::new();

pub fn train_hce(path : &str) -> std::io::Result<()>
{
  unsafe {
    PARAMS = INIT_PARAMS.clone();
    for x in 0..36 { for y in 0..6 { PARAMS.shield1[x][y] += f64::triangular() * 0.125; } }
    for y in 0..6 { PARAMS.shield2[y] += f64::triangular() * 0.125; }
    for k in 0..4 {
      for x in 0..51 { for y in 0..6 { PARAMS.ns1[k][x][y] += f64::triangular() * 0.125; } }
      for y in 0..6 { PARAMS.ns2[k][y] += f64::triangular() * 0.125; }
    }
  }

  let mut ms = Parameters::zero();
  let mut vs = Parameters::zero();

  let mut num_skipped = 0;
  let mut num_outcome_only = 0;
  let mut num_score_only = 0;
  let mut num_checks = 0;
  for triple
    in FENReader::open_scored(path, ScoreUnit::FractionalPawn, ScoreSign::LeaveUnchanged)?
  {
    let (state, score, outcome) = triple?;
    if score == INVALID_SCORE && outcome == Outcome::Unknown { num_skipped += 1; continue; }
    if score == INVALID_SCORE { num_outcome_only += 1; }
    if outcome == Outcome::Unknown { num_score_only += 1; }
    if state.incheck { num_checks += 1; }
    let antiphase =
        (state.boards[WHITE+QUEEN ] | state.boards[BLACK+QUEEN ]).count_ones() as i32 * 12
      + (state.boards[WHITE+ROOK  ] | state.boards[BLACK+ROOK  ]).count_ones() as i32 * 6
      + (state.boards[WHITE+BISHOP] | state.boards[BLACK+BISHOP]).count_ones() as i32 * 4
      + (state.boards[WHITE+KNIGHT] | state.boards[BLACK+KNIGHT]).count_ones() as i32 * 4
      + (state.boards[WHITE+PAWN  ] | state.boards[BLACK+PAWN  ]).count_ones() as i32 * 1;
    let phase = std::cmp::min(48, std::cmp::max(0, 96 - antiphase)) as f64;
    let x = 0.125 + (phase/48.0)*0.125; // 0.125 to 0.250
    /*
      I'd like to change this to
        let phase = std::cmp::min(96, std::cmp::max(0, 96 - antiphase)) as f64;
        let x = 0.125 + phase/256.0;  // 0.125 to 0.500 [x = 0.125 + (phase/96.0)*0.375]
      but that will mess up the training error, making it
      difficult to compare successive versions of the HCE.
    */
    let mixed =
      if score == INVALID_SCORE { (outcome as i16 * 2) as f64 } else {
        let c_score = compress(score as f64 / 100.0);
        match outcome {
          Outcome::Black   => c_score*(1.0-x) - 2.0*x,
          Outcome::Draw    => c_score*(1.0-x),
          Outcome::White   => c_score*(1.0-x) + 2.0*x,
          Outcome::Unknown => c_score,
        }
      };
    assert!(2.0 >= x && x >= -2.0);
    unsafe { DATASET.push((state, mixed)); }
  }
  let num_positions = unsafe { DATASET.len() };
  let num_both = num_positions - num_skipped - num_outcome_only - num_score_only;
  eprintln!(   "{num_positions:10} positions"        );
  eprintln!(     "{num_skipped:10} skipped"          );
  eprintln!(        "{num_both:10} outcome and score");
  eprintln!("{num_outcome_only:10} outcome only"     );
  eprintln!(  "{num_score_only:10} score only"       );
  eprintln!(      "{num_checks:10} checks"           );

  let mut adjusted_alpha = alpha;
  let mut average_flips = 0.0;

  let mut prev_ds = Parameters::zero();

  unsafe {
    WORKER_DERIVS.clear();
    for _ in 0..NUM_THREADS { WORKER_DERIVS.push(Parameters::zero()); }
  }

  let mut offset : usize = 0;
  let mut batch_size : usize = INIT_BATCH_SIZE;

  while batch_size > num_positions { batch_size /= 2; }

  for epoch in 0..65536 {
    println!("// Epoch {epoch}");
    println!("const DFLT_PARAMS : Parameters = {:?};", unsafe { &PARAMS });

    eprint!("\n");
    eprintln!("EPOCH {epoch:03}  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~");
    unsafe { PARAMS.show(); }

    let mut workers = Vec::new();
    for id in 0..NUM_THREADS {
      let offset = offset;
      let batch_size = batch_size;
      workers.push(std::thread::spawn(move || -> f64 {
        let mut total_error = 0.0;
        let mut index = offset + id;
        let params = unsafe { &PARAMS };
        let ds_total = unsafe { &mut WORKER_DERIVS[id] };
        ds_total.reset();
        for _ in 0..(batch_size/NUM_THREADS) {
          while index >= num_positions { index -= num_positions; }
          let position = unsafe { &DATASET[index] };

          let (state, score) = position;
          let mut ds = Parameters::zero();
          let prediction = state.train(params, &mut ds, false);
          let error = compress(prediction) - score;
          total_error += error * error;

          let s = 2.0 * error * d_compress(prediction);
          ds.scale(s);
          ds_total.add(&ds);

          index += NUM_THREADS;
        }
        total_error
      }));
    }

    let mut total_error = 0.0;
    for handle in workers { total_error += handle.join().unwrap(); }

    let mut ds_total = Parameters::zero();

    unsafe { for id in 0..NUM_THREADS { ds_total.add(&WORKER_DERIVS[id]); } }

    let avg_error = total_error / batch_size as f64;

    eprintln!("{:8.6} error", avg_error);

    if adjusted_alpha <= 0.000_244_140_625 { break; }

    ds_total.scale(1.0 / batch_size as f64);

    // Update means
    ms.scale(beta);
    ms.add_scaled(&ds_total, 1.0-beta);

    // Update variances
    vs.scale(gamma);
    vs.add_square_scaled(&ds_total, 1.0-gamma);

    // Update constants
    unsafe { PARAMS.update(&ms, &vs, adjusted_alpha); }

    // Et cetera
    offset += batch_size;
    while offset >= num_positions { offset -= num_positions; }

    let num_flips = ds_total.num_flips(&prev_ds) as f64 / NUM_PARAMS as f64;

    if epoch > 0 { average_flips = average_flips*0.5 + num_flips*0.5; }
    if average_flips >= 0.5 {
      adjusted_alpha *= 0.5;
      average_flips = 0.0;
      if batch_size < num_positions/2 { batch_size *= 2; }
    }
    eprintln!("last flips {:6.4}", average_flips*2.0);
    eprintln!("next alpha {:8.6}", adjusted_alpha);
    eprintln!("next batch {}"    , batch_size);

    prev_ds = ds_total.clone();
  }
  return Ok(());
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn print_board(board : u64)
{
  for rank in (0..8).rev() {
    eprint!("  ");
    for file in 0..8 {
      let square = rank*8 + file;
      let palettes = [["\x1B[48;2;181;135;99m", "\x1B[48;2;212;190;154m"],
                      ["\x1B[48;5;92m",         "\x1B[48;5;93m" ]];
      let bg = palettes[((board>>square)&1) as usize][((rank+file)%2) as usize];
      eprint!("{}   ", bg);
    }
    eprint!("\x1B[49m\n");
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

static INIT_PROJECTION : [[f64; 31]; 6] =
  [
    // nonpassed
    [ 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // blocked
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, // stop square attacked, future stop squares attacked
      1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0       // future stop squares attacked
    ] ,
    // connected
    [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.75,0.75, // source defended once or more
     0.75,0.75,0.25,0.25,0.25,0.25, 0.0, 0.0, // source defence once or more, source defended twice
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ] ,
    // backward
    [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.5, 0.0, // stop square can be defended
      0.0, 0.0,-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, // stop square can be defended
      0.0, 0.0, 0.0, 0.0,-1.0, 1.0, 0.5, 0.0, // stop square defended, stop square attacked, stop square can be attacked
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ] ,
    // overadvanced
    [ 0.0, 0.0, 0.0, 0.0,0.0,0.0,-0.75,-0.75, // source defended
    -0.75,-0.75,0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.75, 0.75,0.75,0.75,0.0, 0.0, 0.0, 0.0, // source can be defended, source difficult to defend
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ] ,
    // advanced
    [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
    ] ,
    // (blank)
    [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
  ] ;
static INIT_PARAMS : Parameters = Parameters {
  base:  [[12.0, 24.0], [5.0, 10.0], [3.0, 6.0], [3.0, 6.0]],
  pawn:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
  qu_rk: 0.0,
  qu_bp: 0.0,
  qu_kt: 0.0,
  rk_bp: 0.0,
  rk_kt: 0.0,
  bp_kt: 0.0,
  bp_fs: [0.0; 4],
  bp_fo: [0.0; 4],
  bp_es: [0.0; 4],
  bp_eo: [0.0; 4],
  qu_f:  [0.0; 4],
  qu_e:  [0.0; 4],
  rk_f:  [0.0; 4],
  rk_e:  [0.0; 4],
  kt_f:  [0.0; 4],
  kt_e:  [0.0; 4],
  scope: [[[0.0; 3]; 51]; 4],
  ns1:   [[[0.0; 6]; 51]; 4],
  ns2:   [[0.0; 6]; 4],
  mob:   [[[0.0; 4]; 3]; 4],
  fork:  [[0.0; 3]; 4],
  indiv:    [0.0; 6],
  combo:    [[0.0; 6]; 6],
  pressure: [[1.0; 4]; 4],
  threat:   [[1.0; 4]; 4],
  expos:    [[0.0; 4]; 3],
  shield1:  [[0.0; 6]; 36],
  shield2:  [0.0; 6],
  project: INIT_PROJECTION,
  interp:  [[[[[[[0.0; 2]; 2]; 2]; 2]; 2]; 2]; 3],
  seventh: [[[[0.0; 3]; 2]; 2]; 4],
  r_outpost: [[0.0; 3]; 3],
  b_outpost: [[0.0; 3]; 3],
  n_outpost: [[0.0; 3]; 3],
  tempo: [0.1, 0.2, 0.0],
};
static DFLT_PARAMS : Parameters = INIT_PARAMS;
