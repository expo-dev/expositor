use crate::color::*;
use crate::movegen::*;
use crate::movetype::*;
use crate::piece::*;
use crate::state::*;

use std::str::SplitAsciiWhitespace;

pub trait Algebraic {
  fn algebraic(&self) -> String;
}

impl Algebraic for u8 {
  fn algebraic(&self) -> String
  {
    if *self >= 64 { return String::from("00"); }
    let mut s = Vec::new();
    s.push(('a' as u8) + (*self % 8));
    s.push(('1' as u8) + (*self / 8));
    return unsafe { String::from_utf8_unchecked(s) };
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
    let mut out = self.src.algebraic();
    out.push_str(&self.dst.algebraic());
    if self.is_promotion() { out.push(LOWER[self.promotion as usize]); }
    return out;
  }
}

pub fn parse_long(state : &State, long : &str) -> Result<Move, String>
{
  #[cfg(debug_assertions)] if !long.is_ascii() { return Err(String::from("not ASCII")); }

  let bytes = long.as_bytes();
  if bytes.len() < 4 || bytes.len() > 5 { return Err(String::from("incorrect length")); }

  let src_file = bytes[0].wrapping_sub('a' as u8);
  let src_rank = bytes[1].wrapping_sub('1' as u8);
  let dst_file = bytes[2].wrapping_sub('a' as u8);
  let dst_rank = bytes[3].wrapping_sub('1' as u8);
  if src_file >= 8 || src_rank >= 8 || dst_file >= 8 || dst_rank >= 8 {
    return Err(String::from("invalid square"));
  }
  let src_square = (src_rank*8 + src_file) as i8;
  let dst_square = (dst_rank*8 + dst_file) as i8;

  let mut promotion : Option<Piece> = None;

  if bytes.len() == 5 {
    let p = bytes[4] as char;
    promotion = match p {
      'q' => Some(Piece::new(state.turn, QUEEN  as u8)),
      'r' => Some(Piece::new(state.turn, ROOK   as u8)),
      'b' => Some(Piece::new(state.turn, BISHOP as u8)),
      'n' => Some(Piece::new(state.turn, KNIGHT as u8)),
       _  => return Err(String::from("invalid promotion"))
    };
  }

  let mut early_moves = Vec::with_capacity(16);
  let mut late_moves  = Vec::with_capacity(32);
  state.generate_legal_moves(Selectivity::Everything, &mut early_moves, &mut late_moves);

  for mv in early_moves.into_iter().chain(late_moves.into_iter()) {
    if mv.src != src_square { continue; }
    if mv.dst != dst_square { continue; }
    match promotion {
      None => if mv.is_promotion() { continue; }
      Some(p) => {
        if !mv.is_promotion() { continue; }
        if mv.promotion != p { continue; }
      }
    }
    return Ok(mv);
  }
  return Err(String::from("no matches"));
}

pub fn parse_short(state : &State, short : &str) -> Result<Move, String>
{
  #[cfg(debug_assertions)] if !short.is_ascii() { return Err(String::from("not ASCII")); }

  let mut src_piece = Piece::new(state.turn, PAWN as u8);
  let mut src_file  : Option<u8>    = None  ;
  let mut src_rank  : Option<u8>    = None  ;
  let mut capture   : bool          = false ;
  let     dst_file  : u8            ;
  let     dst_rank  : u8            ;
  let mut promotion : Option<Piece> = None  ;

  //       src_file   capture  dst_rank        ignored
  //           v         v        v               v
  // [KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?(\+|#)?
  //    ^            ^       ^           ^
  // src_piece  src_rank  dst_file   promotion

  let mut bytes = short.trim_end_matches(|x| x == '+' || x == '#').as_bytes();
  if bytes.len() < 2 { return Err(String::from("incorrect length")); }

  if bytes == "O-O-O".as_bytes() || bytes == "0-0-0".as_bytes() {
    bytes = match state.turn {
      Color::White => "Ke1c1".as_bytes(),
      Color::Black => "Ke8c8".as_bytes(),
    };
  }
  else if bytes == "O-O".as_bytes() || bytes == "0-0".as_bytes() {
    bytes = match state.turn {
      Color::White => "Ke1g1".as_bytes(),
      Color::Black => "Ke8g8".as_bytes(),
    };
  }

  let mut idx : isize = bytes.len() as isize - 1;

  let c = bytes[idx as usize] as char;
  if ['Q', 'R', 'B', 'N'].contains(&c) && bytes[idx as usize - 1] == ('=' as u8) {
    promotion = match c {
      'Q' => Some(Piece::new(state.turn, QUEEN  as u8)),
      'R' => Some(Piece::new(state.turn, ROOK   as u8)),
      'B' => Some(Piece::new(state.turn, BISHOP as u8)),
      'N' => Some(Piece::new(state.turn, KNIGHT as u8)),
       _  => None
    };
    idx = idx - 2;
    if idx < 1 { return Err(String::from("only specifies promotion")); }
  }

  dst_file = bytes[idx as usize - 1].wrapping_sub('a' as u8);
  dst_rank = bytes[  idx as usize  ].wrapping_sub('1' as u8);
  if dst_file >= 8 || dst_rank >= 8 { return Err(String::from("invalid square")); }
  let dst_square = (dst_rank*8 + dst_file) as i8;
  idx = idx - 2;

  if idx >= 0 && bytes[idx as usize] == 'x' as u8 {
    capture = true;
    idx = idx - 1;
  }
  if idx >= 0 && bytes[idx as usize].wrapping_sub('1' as u8) < 8 {
    src_rank = Some(bytes[idx as usize] - '1' as u8);
    idx = idx - 1;
  }
  if idx >= 0 && bytes[idx as usize].wrapping_sub('a' as u8) < 8 {
    src_file = Some(bytes[idx as usize] - 'a' as u8);
    idx = idx - 1;
  }
  if idx >= 0 {
    let c = bytes[idx as usize] as char;
    if ['K', 'Q', 'R', 'B', 'N'].contains(&c) {
      src_piece = match c {
        'K' => Piece::new(state.turn, KING   as u8),
        'Q' => Piece::new(state.turn, QUEEN  as u8),
        'R' => Piece::new(state.turn, ROOK   as u8),
        'B' => Piece::new(state.turn, BISHOP as u8),
        'N' => Piece::new(state.turn, KNIGHT as u8),
         _  => src_piece
      };
      idx = idx - 1;
    }
  }
  if idx != -1 { return Err(String::from("extraneous character")); }

  let mut early_moves = Vec::with_capacity(16);
  let mut late_moves  = Vec::with_capacity(32);
  state.generate_legal_moves(Selectivity::Everything, &mut early_moves, &mut late_moves);

  let mut matched = NULL_MOVE;
  for mv in early_moves.into_iter().chain(late_moves.into_iter()) {
    if mv.piece        != src_piece  { continue; }
    if mv.is_capture() != capture    { continue; }
    if mv.dst          != dst_square { continue; }
    match promotion {
      None => if mv.is_promotion() { continue; }
      Some(p) => {
        if !mv.is_promotion() { continue; }
        if  mv.promotion != p { continue; }
      }
    }
    if src_file.is_some() { if mv.src % 8 != (src_file.unwrap() as i8) { continue; } }
    if src_rank.is_some() { if mv.src / 8 != (src_rank.unwrap() as i8) { continue; } }

    if matched.is_null() { matched = mv.clone(); }
    else { return Err(format!("multiple matches {} {}", matched.algebraic(), mv.algebraic())); }
  }
  if matched.is_null() { return Err(String::from("no matches")); }
  return Ok(matched);
}

pub fn parse_universal(state : &State, either : &str) -> Result<Move, String>
{
  match parse_long(state, either) {
    Ok(mv)     => { return Ok(mv); }
    Err(msg_l) => match parse_short(state, either) {
      Ok(mv)     => { return Ok(mv); }
      Err(msg_s) => { return
        Err(format!("unable to parse as long ({}) or short ({})", msg_l, msg_s));
      }
    }
  }
}

pub fn parse_pgn(state : &State, text : &str, verify : bool) -> Result<Vec<Move>, String>
{
  #[cfg(debug_assertions)] if !text.is_ascii() { return Err(String::from("not ASCII")); }
  return parse_pgn_tokens(state, &mut text.split_ascii_whitespace(), verify);
}

pub fn parse_pgn_tokens(
  state  : &State,
  text   : &mut SplitAsciiWhitespace,
  verify : bool
) -> Result<Vec<Move>, String>
{
  let mut working = state.clone();
  let mut movelist : Vec<Move> = Vec::new();

  for token in text {
    if token.ends_with('.') {
      if verify {
        match token.trim_end_matches('.').parse::<u16>() {
          Ok(x) => if x == 0 || working.ply != (x-1)*2 {
            return Err(String::from("move number mismatch"));
          }
          Err(_) => {
            return Err(String::from("invalid move number"));
          }
        }
      }
      continue;
    }
    if token=="1-0" || token=="0-1" || token=="1/2-1/2" || token=="*" { break; }
    let mv = parse_short(&working, token)?;
    working.apply(&mv);
    movelist.push(mv);
  }
  return Ok(movelist);
}
