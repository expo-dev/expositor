use crate::color::Color::*;
use crate::misc::{piece_destinations, FILE_A, RANK_1};
use crate::movegen::Selectivity::Everything;
use crate::movetype::Move;
use crate::piece::Kind::*;
use crate::piece::Piece;
use crate::state::State;

pub trait Algebraic {
  fn algebraic(&self) -> String;
}

impl Algebraic for u8 {
  fn algebraic(&self) -> String
  {
    assert!(*self < 64);
    return unsafe {
      String::from_utf8_unchecked(vec![b'a' + self % 8, b'1' + self / 8])
    };
  }
}

impl Algebraic for i8 {
  fn algebraic(&self) -> String
  {
    return (*self as u8).algebraic();
  }
}

impl Algebraic for Move {
  fn algebraic(&self) -> String
  {
    if self.is_null() { return String::from("0000"); }
    let src = self.src as u8; assert!(src < 64);
    let dst = self.dst as u8; assert!(dst < 64);
    let mut s = unsafe { String::from_utf8_unchecked(vec![
      b'a' + src % 8, b'1' + src / 8,
      b'a' + dst % 8, b'1' + dst / 8,
    ]) };
    if self.is_promotion() { s.push(self.promotion.kind().lower()); }
    return s;
  }
}

pub fn parse_long(state : &State, long : &str) -> Result<Move, &'static str>
{
  #[cfg(debug_assertions)] if !long.is_ascii() { return Err("not ASCII"); }

  let bytes = long.as_bytes();
  if bytes.len() < 4 || bytes.len() > 5 { return Err("incorrect length"); }

  let src_file = bytes[0].wrapping_sub(b'a');
  let src_rank = bytes[1].wrapping_sub(b'1');
  let dst_file = bytes[2].wrapping_sub(b'a');
  let dst_rank = bytes[3].wrapping_sub(b'1');
  if src_file >= 8 || src_rank >= 8 || dst_file >= 8 || dst_rank >= 8 {
    return Err("invalid square");
  }
  let src_square = (src_rank*8 + src_file) as i8;
  let dst_square = (dst_rank*8 + dst_file) as i8;

  let promotion =
    if bytes.len() == 5 {
      match bytes[4] {
        b'q' => state.turn + Queen,
        b'r' => state.turn + Rook,
        b'b' => state.turn + Bishop,
        b'n' => state.turn + Knight,
        _ => return Err("invalid promotion")
      }
    }
    else {
      Piece::Null
    };

  let (early_moves, late_moves) = state.legal_moves(Everything);

  for mv in early_moves.into_iter().chain(late_moves.into_iter()) {
    if mv.src != src_square { continue; }
    if mv.dst != dst_square { continue; }
    if promotion.is_null() {
      if mv.is_promotion() { continue; }
    }
    else {
      if !mv.is_promotion() { continue; }
      if mv.promotion != promotion { continue; }
    }
    return Ok(mv);
  }
  return Err("no matches");
}

pub fn parse_short(state : &State, short : &str) -> Result<Move, &'static str>
{
  #[cfg(debug_assertions)] if !short.is_ascii() { return Err("not ASCII"); }

  let mut bytes = short.trim_end_matches(|x| x == '+' || x == '#').as_bytes();
  if bytes.len() < 2 { return Err("incorrect length"); }

  let mut src_piece = state.turn + Pawn;
  let mut src_file  : i8    = -1    ;
  let mut src_rank  : i8    = -1    ;
  let mut capture   : bool  = false ;
  let     dst_file  : u8    ;
  let     dst_rank  : u8    ;
  let mut promotion : Piece = Piece::Null;

  //       src_file   capture  dst_rank        ignored
  //           v         v        v               v
  // [KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?(\+|#)?
  //    ^            ^       ^           ^
  // src_piece  src_rank  dst_file   promotion

  if bytes == b"O-O-O" || bytes == b"0-0-0" {
    bytes = match state.turn { White => b"Ke1c1", Black => b"Ke8c8" };
  }
  else if bytes == b"O-O" || bytes == b"0-0" {
    bytes = match state.turn { White => b"Ke1g1", Black => b"Ke8g8" };
  }

  let mut len = bytes.len();
  if len < 2 { return Err("incorrect length"); }

  let c = bytes[len-1];
  if bytes[len-2] == b'=' {
    promotion = match c {
      b'Q' => state.turn + Queen,
      b'R' => state.turn + Rook,
      b'B' => state.turn + Bishop,
      b'N' => state.turn + Knight,
      _  => return Err("invalid promotion")
    };
    len -= 2;
  }

  if len < 2 { return Err("missing destination"); }
  dst_file = bytes[len-2].wrapping_sub(b'a');
  dst_rank = bytes[len-1].wrapping_sub(b'1');
  if dst_file >= 8 || dst_rank >= 8 { return Err("invalid destination"); }
  let dst_square = (dst_rank*8 + dst_file) as i8;
  len -= 2;

  loop {
    if len == 0 { break; }

    if bytes[len-1] == b'x' {
      capture = true;
      len -= 1;
      if len == 0 { break; }
    }

    let ofs = bytes[len-1].wrapping_sub(b'1');
    if ofs < 8 {
      src_rank = ofs as i8;
      len -= 1;
      if len == 0 { break; }
    }

    let ofs = bytes[len-1].wrapping_sub(b'a');
    if ofs < 8 {
      src_file = ofs as i8;
      len -= 1;
      if len == 0 { break; }
    }

    src_piece = match bytes[len-1] {
      b'K' => state.turn + King,
      b'Q' => state.turn + Queen,
      b'R' => state.turn + Rook,
      b'B' => state.turn + Bishop,
      b'N' => state.turn + Knight,
      _ => return Err("invalid piece")
    };
    len -= 1;
    break;
  }
  if len > 0 { return Err("incorrect length"); }

  let (early_moves, late_moves) = state.legal_moves(Everything);

  let mut matched = Move::NULL;
  for mv in early_moves.into_iter().chain(late_moves.into_iter()) {
    if mv.dst          != dst_square { continue; }
    if mv.piece        != src_piece  { continue; }
    if mv.is_capture() != capture    { continue; }
    if promotion.is_null() {
      if mv.is_promotion() { continue; }
    }
    else {
      if !mv.is_promotion() { continue; }
      if mv.promotion != promotion { continue; }
    }
    if src_file >= 0 { if (mv.src as u8) % 8 != src_file as u8 { continue; } }
    if src_rank >= 0 { if (mv.src as u8) / 8 != src_rank as u8 { continue; } }

    if matched.is_null() { matched = mv; }
    else { return Err("multiple matches"); }
  }
  if matched.is_null() { return Err("no matches"); }
  return Ok(matched);
}

impl Move {
  pub fn in_context(&self, state : &State) -> String
  {
    let algebraic_file = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    let algebraic_rank = ['1', '2', '3', '4', '5', '6', '7', '8'];

    if self.is_null() { return String::from("0000"); }
    let mut buf = String::new();
    let kind = self.piece.kind();
    if kind == King && self.dst == self.src + 2 {
      buf.push_str("0-0");
    }
    else
    if kind == King && self.dst == self.src - 2 {
      buf.push_str("0-0-0");
    }
    else {
      if kind == Pawn {
        if self.is_capture() {
          buf.push(algebraic_file[self.src as usize % 8]);
        }
      }
      else {
        buf.push(kind.upper());
        // If another piece of the same color and kind can also reach the
        //   destination square, emit the file to disambiguate, or if the
        //   file is shared, emit the rank. In very rare cases, we may to
        //   emit both (consider three queens and a destination arranged
        //   in an equilateral rectangle).
        // TODO technically, this should ignore pieces that can't actually
        //   reach the destination square because they are pinned.
        let composite = state.sides[White] | state.sides[Black];
        let reaching = piece_destinations(kind, self.dst as usize, composite)
                     & state.boards[self.piece];
        if reaching.count_ones() > 1 {
          let file = self.src as usize % 8;
          let rank = self.src as usize / 8;
          let file_count = (reaching & (FILE_A <<   file  )).count_ones();
          let rank_count = (reaching & (RANK_1 << (rank*8))).count_ones();
          if file_count < 2 {
            buf.push(algebraic_file[file]);
          }
          else if rank_count < 2 {
            buf.push(algebraic_rank[rank]);
          }
          else {
            buf.push(algebraic_file[file]);
            buf.push(algebraic_rank[rank]);
          }
        }
      }
      if self.is_capture() { buf.push('x'); }
      buf.push_str(&self.dst.algebraic());
      if self.is_promotion() {
        buf.push('=');
        buf.push(self.promotion.kind().upper());
      }
    }
    if self.gives_check() {
      let mut scratch = state.clone_empty();
      scratch.apply(self);
      let (early_moves, late_moves) = scratch.legal_moves(Everything);
      buf.push(
        if early_moves.is_empty() && late_moves.is_empty() { '#' } else { '+' }
      );
    }
    return buf;
  }
}
