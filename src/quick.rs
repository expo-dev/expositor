use crate::piece::Piece;
use crate::score::CoScore;
use crate::state::MiniState;

// Training datasets can contain nearly a billion positions, taking up a fair
//   amount of disk space and taking a long time to deserialize.
//
// To alleviate some of these pain points, we use a bespoke storage format
//   as an alternative to Forsyth-Edwards notation, one designed to be quickly
//   deserialized. Incidentally, it's also smaller on average.

// Positions in the quick format have a fixed length of 40 bytes:
//   AAA C KQRRBBNNPPPPPPPP kqrrbbnnpppppppp VVVV
//
// Each byte is encoded as one of the characters [0-9A-Za-z&$:]. After decoding,
//   its value is in the range 0−63 or is 255.
//
// The first three bytes encode scoring information (annotations):
//   AAA = 00ss ssss  (byte 0)
//         00ss ssss  (byte 1)
//         00oo ssss  (byte 2)
//   oo = outcome (0 incomplete, 1 drawn, 2 white win, 3 black win)
//   ssss ssss ssss ssss = score as an i16
//
// The next byte encodes two fields:
//   C = 000t rrrr
//   rrrr = castling rights
//   t = 0 when white has the move and 1 when black has the move
//
// The next sixteen bytes encode the square indices of white's pieces. When a
//   piece is not present on the board, its index is 255. For example, white's
//   pieces in the position "8/8/8/8/8/8/4PPPP/4KBNR w K - 0 1" would be coded
//
//   [4, 255, 7, 255, 5, 255, 6, 255, 12, 13, 14, 15, 255, 255, 255, 255]
//
// The next sixteen bytes encode the square indices of black's pieces.
//
// The last four bytes encode two pairs. The first value of a pair is in the
//   range 0–5, 8–13, or 255 and indicates the kind of a piece. The second
//   value of a pair is in the range 0−63 or 255 and indicates a square index.
//   These two pairs are used for promotions.
//
// Note that the en passant target, halfmove clock, and fullmove number are not
//   stored in this format.
//
// TODO include the depth from zeroing

impl MiniState {
  pub fn to_quick(&self) -> String
  {
    let mut out = vec![58u8; 40];

    let s = self.score.as_i16() as u16;
    let o = match self.packed >> 1 {
      0b101 => 0, // incomplete
      0b000 => 1, // black win
      0b001 => 2, // draw
      0b010 => 3, // white win
      _ => unreachable!()
    };
    out[0] = enc(( s        & 63) as u8);
    out[1] = enc(((s >>  6) & 63) as u8);
    out[2] = enc(((s >> 12) & 63) as u8 | o << 4);
    out[3] = enc(self.rights | (self.packed & 1) << 4);

    for color in 0..2 {
      for ofs in 0..16 {
        out[4 + color*16 + ofs] = enc(self.positions[color][ofs] as u8);
      }
    }

    out[36] = enc(self.variable[0].0 as u8);
    out[37] = enc(self.variable[0].1 as u8);
    out[38] = enc(self.variable[1].0 as u8);
    out[39] = enc(self.variable[1].1 as u8);

    return unsafe { String::from_utf8_unchecked(out) };
  }

  pub fn from_quick(record : &[u8; 40]) -> MiniState
  {
    let a0 = dec(record[0]); debug_assert!(a0 != 255);
    let a1 = dec(record[1]); debug_assert!(a1 != 255);
    let a2 = dec(record[2]); debug_assert!(a2 != 255);
    let  c = dec(record[3]); debug_assert!( c != 255);
    let s = a0 as u16 | (a1 as u16) << 6 | ((a2 & 15) as u16) << 12;
    let o = match a2 >> 4 {
      0 => 0b_1010, // incomplete
      1 => 0b_0000, // black win
      2 => 0b_0010, // draw
      3 => 0b_0100, // white win
      _ => unsafe { std::hint::unreachable_unchecked() }
    };

    let mut positions = [[-1; 16]; 2];
    for color in 0..2 {
      for ofs in 0..16 {
        positions[color][ofs] = dec(record[4 + color*16 + ofs]) as i8;
      }
    }

    let p0 = Piece::from(dec(record[36]));
    let x0 = dec(record[37]) as i8;
    let p1 = Piece::from(dec(record[38]));
    let x1 = dec(record[39]) as i8;
    let variable = [(p0, x0), (p1, x1)];

    let mini = MiniState {
      positions: positions,
      variable:  variable,
      rights: c & 15,
      packed: o | c >> 4,
      score:  CoScore::new(s as i16),
    };

    // TODO check that castling rights are set properly?

    return mini;
  }
}

#[inline] pub fn enc(x : u8) -> u8 { return ENCODE[x as usize]; }
#[inline] pub fn dec(c : u8) -> u8 { return DECODE[c as usize]; }

static ENCODE : [u8; 256] = [
   48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  65,  66,  67,  68,  69,  70,
   71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,
   87,  88,  89,  90,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
  109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,  38,  36,
   58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,
   58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,
   58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,
   58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,
   58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,
   58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,
   58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,
   58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,
   58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,
   58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,
   58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,
   58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,  58,
];

static DECODE : [u8; 256] = [
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255,  63, 255,  62, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9, 255, 255, 255, 255, 255, 255,
  255,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,
   25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35, 255, 255, 255, 255, 255,
  255,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,
   51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

/*
pub fn encode(bs : &[u8]) -> Vec<u8>
{
  let n = bs.len() / 3;
  assert!(bs.len() == n*3);
  let mut out = vec![0; n*4];
  for ofs in 0..n {
    unsafe {
      let t0 = bs.get_unchecked(ofs*3+0);
      let t1 = bs.get_unchecked(ofs*3+1);
      let t2 = bs.get_unchecked(ofs*3+2);
      let q0 =   t0 & 0b0011_1111;
      let q1 = ((t0 & 0b1100_0000) >> 6) | ((t1 & 0b0000_1111) << 2);
      let q2 = ((t1 & 0b1111_0000) >> 4) | ((t2 & 0b0000_0011) << 4);
      let q3 =  (t2 & 0b1111_1100) >> 2 ;
      *out.get_unchecked_mut(ofs*4+0) = ENCODE[q0 as usize];
      *out.get_unchecked_mut(ofs*4+1) = ENCODE[q1 as usize];
      *out.get_unchecked_mut(ofs*4+2) = ENCODE[q2 as usize];
      *out.get_unchecked_mut(ofs*4+3) = ENCODE[q3 as usize];
    }
  }
  return out;
}

pub fn decode(bs : &[u8]) -> Vec<u8>
{
  let n = bs.len() / 4;
  assert!(bs.len() == n*4);
  let mut out = vec![0; n*3];
  for ofs in 0..n {
    unsafe {
      let q0 = DECODE[*bs.get_unchecked(ofs*4+0) as usize];
      let q1 = DECODE[*bs.get_unchecked(ofs*4+1) as usize];
      let q2 = DECODE[*bs.get_unchecked(ofs*4+2) as usize];
      let q3 = DECODE[*bs.get_unchecked(ofs*4+3) as usize];
      let t0 =  (q0 & 0b0011_1111)       | ((q1 & 0b0000_0011) << 6);
      let t1 = ((q1 & 0b0011_1100) >> 2) | ((q2 & 0b0000_1111) << 4);
      let t2 = ((q2 & 0b0011_0000) >> 4) | ((q3 & 0b0011_1111) << 2);
      *out.get_unchecked_mut(ofs*3+0) = t0;
      *out.get_unchecked_mut(ofs*3+1) = t1;
      *out.get_unchecked_mut(ofs*3+2) = t2;
    }
  }
  return out;
}
*/
