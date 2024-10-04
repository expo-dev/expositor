use crate::color::Color::{self, *};
use crate::misc::{
  Algebraic,
  piece_destinations,
  pawn_attacks,
  SHORT, LONG,
  CASTLE_BTWN,
  CASTLE_CHCK,
  FILE_A,
  RANK_1
};
use crate::movegen::Selectivity::Everything;
use crate::piece::Kind::{self, *};
use crate::piece::Piece;
use crate::state::State;

use std::fmt::Write;

const CAPTURE_MASK : u8 = 0b_0000_0001; // .... ...1
const  ENPASS_MASK : u8 = 0b_0000_0010; // .... ..1.
const PROMOTE_MASK : u8 = 0b_0000_0100; // .... .1..
const  CASTLE_MASK : u8 = 0b_0000_1000; // .... 1...
const    PAWN_MASK : u8 = 0b_0001_0000; // ...1 ....
const GAINFUL_MASK : u8 = 0b_0000_0101; // .... .1.1 captures + promotions
const ZEROING_MASK : u8 = 0b_0001_0001; // ...1 ...1 captures + pawn moves
const UNUSUAL_MASK : u8 = 0b_0000_0110; // .... .11. promotions + en passant

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum MoveType {
  Manoeuvre          = 0b_0000_0000,  // .... ....
  Capture            = 0b_0000_0001,  // .... ...1
  Castle             = 0b_0000_1000,  // .... 1...
  PawnManoeuvre      = 0b_0001_0000,  // ...1 ....
  PawnCapture        = 0b_0001_0001,  // ...1 ...1
  CaptureEnPassant   = 0b_0001_0011,  // ...1 ..11
  Promotion          = 0b_0001_0100,  // ...1 .1..
  PromotionByCapture = 0b_0001_0101,  // ...1 .1.1
}

impl MoveType {
  #[inline] pub fn is_manoeuvre(self) -> bool { return (self as u8) &   !PAWN_MASK == 0; }
  #[inline] pub fn is_capture(self)   -> bool { return (self as u8) & CAPTURE_MASK != 0; }
  #[inline] pub fn is_enpass(self)    -> bool { return (self as u8) &  ENPASS_MASK != 0; }
  #[inline] pub fn is_promotion(self) -> bool { return (self as u8) & PROMOTE_MASK != 0; }
  #[inline] pub fn is_castle(self)    -> bool { return (self as u8) &  CASTLE_MASK != 0; }
  #[inline] pub fn is_pawn(self)      -> bool { return (self as u8) &    PAWN_MASK != 0; }
  #[inline] pub fn is_gainful(self)   -> bool { return (self as u8) & GAINFUL_MASK != 0; }
  #[inline] pub fn is_zeroing(self)   -> bool { return (self as u8) & ZEROING_MASK != 0; }
  #[inline] pub fn is_unusual(self)   -> bool { return (self as u8) & UNUSUAL_MASK != 0; }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Move {
  pub src        : i8,
  pub dst        : i8,
  pub piece      : Piece,
  pub promotion  : Piece,

  pub captured   : Piece,
  pub movetype   : MoveType,
  pub givescheck : u8,
  pub score      : i8,
}

impl Move {
  pub const NULL : Move = Move {
    movetype:   MoveType::Manoeuvre,
    givescheck: 0,
    src:        0,
    dst:        0,
    piece:      Piece::ZERO,
    captured:   Piece::ZERO,
    promotion:  Piece::ZERO,
    score:      0
  };
}

// NOTE that this function compares all fields of a Move except givescheck and
//   score. The order of the comparisons is essentially arbitrary, but they are
//   roughly in the order of likeliness to fail.
impl PartialEq for Move {
  fn eq(&self, other : &Self) -> bool
  {
    return self.src       == other.src
        && self.dst       == other.dst
        && self.piece     == other.piece
        && self.promotion == other.promotion
        && self.captured  == other.captured
        && self.movetype  == other.movetype;
  }
}

impl Eq for Move {}

// NOTE that this function only compares the first four fields of a Move: src,
//   dst, piece, and promotion. This requires that the Move struct uses C layout
//   rules and that those four fields are first in the struct declaration.
// NOTE that this is mega jank.
pub fn fast_eq(a : &Move, b : &Move) -> bool
{
  unsafe {
    let a = std::mem::transmute::<_, &[u32; 2]>(a);
    let b = std::mem::transmute::<_, &[u32; 2]>(b);
    return a[0] == b[0];
  }
}

impl Move {
  #[inline] pub fn is_manoeuvre(&self) -> bool { return self.movetype.is_manoeuvre(); }
  #[inline] pub fn is_capture  (&self) -> bool { return self.movetype.is_capture();   }
  #[inline] pub fn is_enpass   (&self) -> bool { return self.movetype.is_enpass();    }
  #[inline] pub fn is_promotion(&self) -> bool { return self.movetype.is_promotion(); }
  #[inline] pub fn is_castle   (&self) -> bool { return self.movetype.is_castle();    }
  #[inline] pub fn is_pawn     (&self) -> bool { return self.movetype.is_pawn();      }
  #[inline] pub fn is_gainful  (&self) -> bool { return self.movetype.is_gainful();   }
  #[inline] pub fn is_zeroing  (&self) -> bool { return self.movetype.is_zeroing();   }
  #[inline] pub fn is_unusual  (&self) -> bool { return self.movetype.is_unusual();   }
  #[inline] pub fn is_null     (&self) -> bool { return self.src == self.dst;         }

  #[inline] pub fn gives_check           (&self) -> bool { return self.givescheck     != 0; }
  #[inline] pub fn gives_direct_check    (&self) -> bool { return self.givescheck & 1 != 0; }
  #[inline] pub fn gives_discovered_check(&self) -> bool { return self.givescheck & 2 != 0; }

  const LT64 : u16 = 0b_0000_0000_0011_1111;
  const MSB  : u16 = 0b_1000_0000_0000_0000;

  pub fn compress(&self) -> u16
  {
    // Format is pkkk dddd ddss ssss
    //
    //      p   non-promotion if p = 0, promotion if p = 1
    //    kkk   ignored if p = 0, promotion kind if p = 1
    // dddddd   destination square
    // ssssss   source square
    //
    let src = self.src as u16;
    let dst = self.dst as u16;
    debug_assert!(src & !Self::LT64 == 0, "unable to compress source {}", src);
    debug_assert!(dst & !Self::LT64 == 0, "unable to compress destination {}", dst);
    let prom = if self.is_promotion() { Self::MSB } else { 0 };
    let kind = self.promotion.kind() as u16;
    return prom | (kind << 12) | (dst << 6) | src;
  }

  // NOTE that this function only checks the pseudolegality of the move in the
  //   given context (i.e. the king may be left in check)!
  // NOTE that this function does not set givescheck.
  // NOTE that this function does not set the score.
  pub fn decompress(state : &State, cm : u16) -> Self
  {
    use MoveType::*;

    let src = (   cm     & Self::LT64) as i8;
    let dst = ((cm >> 6) & Self::LT64) as i8;

    // Is this a null move?
    if src == dst { return Self::NULL; }

    // Is there a piece of the right color on the source?
    let piece = state.squares[src as usize];
    if piece.is_null() || piece.color() != state.turn { return Self::NULL; }

    // Is that piece attempting to capture another piece of its own color?
    let mut captured = state.squares[dst as usize];
    if !captured.is_null() && captured.color() == state.turn { return Self::NULL; }

    let promotion : Piece;
    let movetype : MoveType;

    let composite = state.sides[White] | state.sides[Black];

    // Non-promotions
    if cm & Self::MSB == 0 {
      if piece.kind() == Pawn {
        if dst < 8 || dst >= 56 { return Self::NULL; }
        if captured.is_null() {
          let front = src + match state.turn { White => 8, Black => -8 };
          if dst == front {
            movetype = PawnManoeuvre;
          }
          else if dst == state.enpass {
            let not_diagonal = pawn_attacks(state.turn, 1 << src) & (1 << dst) == 0;
            if not_diagonal { return Self::NULL; }
            captured = (!state.turn) + Pawn;
            movetype = CaptureEnPassant;
          }
          else {
            let start = match state.turn { White => src < 16, Black => src > 47 };
            if !start { return Self::NULL; }
            if (1 << front) & composite != 0 { return Self::NULL; }
            let advance = front + match state.turn { White => 8, Black => -8 };
            if dst != advance { return Self::NULL; }
            movetype = PawnManoeuvre;
          }
        }
        else {
          let not_diagonal = pawn_attacks(state.turn, 1 << src) & (1 << dst) == 0;
          if not_diagonal { return Self::NULL; }
          movetype = PawnCapture;
        }
      }
      else if piece.kind() == King && (dst - src).abs() == 2 {
        let direction = if dst < src { LONG } else { SHORT };

        // Does the corresponding right exist?
        //   (This also ensures the king is on the correct square and that the rook is present)
        let rights_mask = match state.turn { White => [1, 2], Black => [4, 8] };
        if state.rights & rights_mask[direction] == 0 { return Self::NULL; }

        // Are the intermediate squares clear?
        let intermediate = &CASTLE_BTWN[state.turn];
        if composite & intermediate[direction] != 0 { return Self::NULL; }

        // Are the travel squares safe?
        let travel = &CASTLE_CHCK[state.turn];
        let danger = state.attacks_by(!state.turn, composite);
        if danger & travel[direction] != 0 { return Self::NULL; }

        movetype = Castle;
      }
      else {
        // Is the destination reachable?
        let possible = piece_destinations(piece.kind(), src as usize, composite);
        if (1 << dst) & possible == 0 { return Self::NULL; }
        movetype = if captured.is_null() { Manoeuvre } else { Capture };
      }
      promotion = Piece::ZERO;
    }

    // Promotions
    else {
      if piece.kind() != Pawn { return Self::NULL; }
      let endzone = match state.turn { White => dst >= 56, Black => dst < 8 };
      if !endzone { return Self::NULL; }
      if captured.is_null() {
        let step = dst - src == match state.turn { White => 8, Black => -8 };
        if !step { return Self::NULL; }
        movetype = Promotion;
      }
      else {
        let not_diagonal = pawn_attacks(state.turn, 1 << src) & (1 << dst) == 0;
        if not_diagonal { return Self::NULL; }
        movetype = PromotionByCapture;
      }
      let kind : Kind = unsafe { std::mem::transmute(((cm >> 12) & 0b_0111) as u8) };
      promotion = state.turn + kind;
    }

    if captured.is_null() { captured = Piece::ZERO };
    return Self {
      src:        src,
      dst:        dst,
      piece:      piece,
      promotion:  promotion,
      captured:   captured,
      movetype:   movetype,
      givescheck: 0,
      score:      0,
    };
  }
}

impl State {
  pub fn parse(&self, text : &str) -> Result<Move, &'static str>
  {
    let mut text = text.as_bytes();

    while let [body @ .., last] = text {
      if *last == b'+' || *last == b'#' { text = body; } else { break; }
    }

    let mut src_piece : Piece = Piece::Null;
    let mut src_file  : i8    = -1;
    let mut src_rank  : i8    = -1;
    let mut capture   : bool  = false;
    let     dst_file  : u8    ;
    let     dst_rank  : u8    ;
    let mut promotion : Piece = Piece::Null;

    //       src_file   capture  dst_rank        ignored
    //           v         v        v               v
    // [KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?(\+|#)?
    //    ^            ^       ^           ^
    // src_piece  src_rank  dst_file   promotion

    if text == b"O-O" || text == b"0-0" {
      text = match self.turn { White => b"Ke1g1", Black => b"Ke8g8" };
    }
    else if text == b"O-O-O" || text == b"0-0-0" {
      text = match self.turn { White => b"Ke1c1", Black => b"Ke8c8" };
    }

    let mut len = text.len();
    if len < 2 { return Err("insufficient length"); }

    'parse_promotion: {
      promotion = match text[len-1] {
        b'Q' => self.turn + Queen ,
        b'R' => self.turn + Rook  ,
        b'B' => self.turn + Bishop,
        b'N' => self.turn + Knight,
        b'q' => self.turn + Queen ,
        b'r' => self.turn + Rook  ,
        b'b' => self.turn + Bishop,
        b'n' => self.turn + Knight,
          _  => {
          if text[len-2] == b'=' { return Err("invalid promotion"); }
          break 'parse_promotion
        }
      };
      if text[len-2] == b'=' { len -= 2; } else { len -= 1; }
    }

    if len < 2 { return Err("missing destination"); }

    dst_file = text[len-2].wrapping_sub(b'a');
    dst_rank = text[len-1].wrapping_sub(b'1');

    if dst_file >= 8 || dst_rank >= 8 { return Err("invalid destination"); }

    let dst_square = (dst_rank*8 + dst_file) as i8;
    len -= 2;

    'parse_prefix: {
      if len == 0 { break 'parse_prefix; }

      if text[len-1] == b'x' {
        capture = true;
        len -= 1;
        if len == 0 { break 'parse_prefix; }
      }
      if len > 1 && text[len-1] == 0x97 && text[len-2] == 0xc3 {
        capture = true;
        len -= 2;
        if len == 0 { break 'parse_prefix; }
      }

      let ofs = text[len-1].wrapping_sub(b'1');
      if ofs < 8 {
        src_rank = ofs as i8;
        len -= 1;
        if len == 0 { break 'parse_prefix; }
      }

      let ofs = text[len-1].wrapping_sub(b'a');
      if ofs < 8 {
        src_file = ofs as i8;
        len -= 1;
        if len == 0 { break 'parse_prefix; }
      }

      src_piece = match text[len-1] {
        b'K' => self.turn + King,
        b'Q' => self.turn + Queen,
        b'R' => self.turn + Rook,
        b'B' => self.turn + Bishop,
        b'N' => self.turn + Knight,
        b'k' => self.turn + King,
        b'q' => self.turn + Queen,
        b'r' => self.turn + Rook,
        b'b' => self.turn + Bishop,
        b'n' => self.turn + Knight,
          _  => return Err("invalid piece")
      };
      len -= 1;
    }

    if len > 0 { return Err("excessive length"); }

    let mut matched = Move::NULL;

    for mv in self.legal_moves(Everything) {
      if mv.dst != dst_square { continue; }
      if src_piece.is_null() {
        if mv.piece != self.turn + Pawn {
          if src_file < 0 || src_rank < 0 { continue; }
        }
      }
      else {
        if mv.piece != src_piece { continue; }
      }
      if promotion.is_null() {
        if mv.is_promotion() { continue; }
      }
      else {
        if !mv.is_promotion() { continue; }
        if mv.promotion != promotion { continue; }
      }
      if capture && !mv.is_capture() { continue; }
      if src_file >= 0 { if (mv.src as u8) % 8 != src_file as u8 { continue; } }
      if src_rank >= 0 { if (mv.src as u8) / 8 != src_rank as u8 { continue; } }

      if matched.is_null() { matched = mv; }
      else { return Err("ambiguous"); }
    }
    if matched.is_null() { return Err("no matches"); }
    return Ok(matched);
  }
}

impl std::fmt::Octal for Move {
  fn fmt(&self, f : &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    // We use the o flag for UCI output
    if self.is_null() { return f.write_str("0000"); }
    let src = self.src as u8; assert!(src < 64);
    let dst = self.dst as u8; assert!(dst < 64);
    let pair = [
      b'a' + src % 8, b'1' + src / 8,
      b'a' + dst % 8, b'1' + dst / 8,
    ];
    f.write_str(unsafe { std::str::from_utf8_unchecked(&pair) })?;
    if self.is_promotion() {
      f.write_char(self.promotion.kind().lower())?;
    }
    return Ok(());
  }
}

impl std::fmt::Display for Move {
  // NOTE that this does not disambiguate moves as required of short algebraic
  //   notation and does not annotate checkmates, as these are contextual; see
  //   StateMove::fmt for a function that is context-aware.
  fn fmt(&self, f : &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    if self.is_null() { return f.write_str("null"); }

    let mut buffer = String::new();

    let kind = self.piece.kind();
    if kind == King && self.dst == self.src + 2 {
      buffer.push_str("0-0");
    }
    else if kind == King && self.dst == self.src - 2 {
      buffer.push_str("0-0-0");
    }
    else {
      if kind == Pawn {
        if self.is_capture() {
          buffer.push(self.src.file_id());
        }
      }
      else {
        buffer.push(kind.upper());
      }
      if self.is_capture() {
        buffer.push('×');
      }
      buffer.push_str(self.dst.id());
      if self.is_promotion() {
        buffer.push('=');
        buffer.push(self.promotion.kind().upper());
      }
    }

    if self.gives_check() {
      buffer.push('+');
    }

    if let Some(width) = f.width() {
      let rjust = f.align() == Some(std::fmt::Alignment::Right);
      let written = buffer.len();
      if written < width {
        if !rjust { for _ in 0..(width - written) { f.write_char(' ')?; } }
        f.write_str(&buffer)?;
        if  rjust { for _ in 0..(width - written) { f.write_char(' ')?; } }
        return Ok(());
      }
    }

    return f.write_str(&buffer);
  }
}

pub struct DisambiguatedMove {
  pub repr  : String,
  pub width : u8,
  pub turn  : Color,
  pub hi    : u8,
}

// highlight = 0  default style
// highlight = 1  first in P-V
// highlight = 2  later in P-V

impl Move {
  // TODO or state.disambiguate(mv) intead?
  pub fn disambiguate(&self, state : &State) -> DisambiguatedMove
  {
    if self.is_null() {
      return DisambiguatedMove {
        repr: String::from("null"),
        width: 4,
        turn: state.turn,
        hi: 0
      };
    }

    let mut buffer = String::new();

    let kind = self.piece.kind();
    if kind == King && self.dst == self.src + 2 {
      buffer.push_str("0-0");
    }
    else if kind == King && self.dst == self.src - 2 {
      buffer.push_str("0-0-0");
    }
    else {
      if kind == Pawn {
        if self.is_capture() {
          buffer.push(self.src.file_id());
        }
      }
      else {
        buffer.push(kind.upper());
        // If another piece of the same color and kind can also reach the
        //   destination square, emit the file to disambiguate, or if the
        //   file is shared, emit the rank. In very rare cases, we may to
        //   emit both (consider three queens and a destination arranged
        //   in an equilateral rectangle).
        // TODO technically, this should ignore pieces that can't actually
        //   reach the destination square because they are pinned.
        let composite = state.sides[White] | state.sides[Black];
        let alt = piece_destinations(kind, self.dst as usize, composite)
                & state.boards[self.piece];
        if alt.count_ones() > 1 {
          let files = (alt & (FILE_A << (self.src &  7))).count_ones();
          let ranks = (alt & (RANK_1 << (self.src & !7))).count_ones();
          if files < 2 {
            buffer.push(self.src.file_id());
          }
          else if ranks < 2 {
            buffer.push(self.src.rank_id());
          }
          else {
            buffer.push(self.src.file_id());
            buffer.push(self.src.rank_id());
          }
        }
      }
      if self.is_capture() { buffer.push('×'); }
      buffer.push_str(self.dst.id());
      if self.is_promotion() {
        buffer.push('=');
        buffer.push(self.promotion.kind().upper());
      }
    }
    if self.gives_check() {
      let mut scratch = state.clone_empty();
      scratch.apply(self);
      let legal_moves = scratch.legal_moves(Everything);
      buffer.push(
        if legal_moves.early_len == 0
         && legal_moves.late_len == 0 { '#' } else { '+' }
      );
    }

    let mut width = if buffer.len() > 255 { 255 } else { buffer.len() as u8 };
    if self.is_capture() { width -= 1; }  // "×".len() = 2

    return DisambiguatedMove {
      repr: buffer,
      width: width,
      turn: state.turn,
      hi: 0
    };
  }
}

impl std::fmt::Display for DisambiguatedMove {
  fn fmt(&self, f : &mut std::fmt::Formatter<'_>) -> std::fmt::Result
  {
    // We use the # flag for colorized output
    if f.alternate() {
      let (l, c, h) = match self.hi {
        0 => [(0.68, 0.150, 270.0), (0.68, 0.150, 25.0)],
        1 => [(0.68, 0.0  , 270.0), (0.68, 0.0  , 25.0)],
        _ => [(0.34, 0.025, 270.0), (0.34, 0.050, 25.0)],
      }[self.turn];
      let fg = crate::term::Lch::new(l, c, h / 360.0).to_rgb().fg();
      f.write_str(&fg)?;
    }

    if let Some(width) = f.width() {
      let rjust = f.align() == Some(std::fmt::Alignment::Right);
      if (self.width as usize) < width {
        let pad = width - self.width as usize;
        if !rjust { for _ in 0..pad { f.write_char(' ')?; } }
        f.write_str(&self.repr)?;
        if  rjust { for _ in 0..pad { f.write_char(' ')?; } }
      }
      else {
        f.write_str(&self.repr)?;
      }
    }
    else {
      f.write_str(&self.repr)?;
    }

    if f.alternate() {
      f.write_str("\x1B[39m")?;
    }

    return Ok(());
  }
}
