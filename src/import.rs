use crate::algebraic::*;
use crate::color::*;
use crate::movetype::*;
use crate::state::*;
use crate::score::*;

use std::fs::File;
use std::io::{BufReader, BufRead, Error, ErrorKind};

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ScoreUnit {
  IntegerCentipawn,
  FractionalPawn,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ScoreSign {
  WhitePositive,
  PointOfView,
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub struct       FENReader { reader : BufReader<File> }
pub struct ScoredFENReader { reader : BufReader<File>, unit : ScoreUnit, sign : ScoreSign }
pub struct       PGNReader { reader : BufReader<File>, line : usize }

impl FENReader {
  pub fn open(path : &str) -> std::io::Result<Self> {
    return Ok(Self { reader: BufReader::new(File::open(path)?) });
  }
}

impl ScoredFENReader {
  pub fn open(path : &str, unit : ScoreUnit, sign : ScoreSign) -> std::io::Result<Self> {
    return Ok(Self { reader: BufReader::new(File::open(path)?), unit: unit, sign: sign });
  }
}

impl PGNReader {
  pub fn open(path : &str) -> std::io::Result<Self> {
    return Ok(Self { reader: BufReader::new(File::open(path)?), line: 0 });
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn state_from_fen(fen : &str) -> std::io::Result<State>
{
  return match State::from_fen(fen) {
    Ok(state) => Ok(state),
    Err(msg) => Err(Error::new(ErrorKind::Other, msg))
  };
}

impl Iterator for FENReader {
  type Item = std::io::Result<State>;

  fn next(&mut self) -> Option<Self::Item> {
    let mut buf = String::new();
    return match self.reader.read_line(&mut buf) {
      Err(e) => Some(Err(e)),
      Ok(sz) => if sz == 0 { None } else { Some(state_from_fen(&buf)) }
    };
  }
}

impl FENReader {
  pub fn collect(self) -> std::io::Result<Vec<State>>
  {
    let mut positions = Vec::new();
    for state in self {
      match state {
        Ok(pos) => { positions.push(pos); }
        Err(e) => { return Err(e); }
      }
    }
    return Ok(positions);
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

fn scored_state_from_fen(
  scored_fen : &str,
  unit : ScoreUnit,
  sign : ScoreSign
) -> std::io::Result<(i16, State)>
{
  let mut fields = scored_fen.split_ascii_whitespace();
  if let Some(fst) = fields.next() {
    let score;
    match unit {
      ScoreUnit::IntegerCentipawn => {
        score = match fst.parse::<i16>() {
           Ok(x) => x,
          Err(_) => match fst {
            "+#" => PROVEN_MATE,
            "-#" => PROVEN_LOSS,
            _ => return Err(Error::new(ErrorKind::Other, String::from("unable to parse score")))
          }
        };
      }
      ScoreUnit::FractionalPawn => {
        score = match fst.parse::<f32>() {
           Ok(x) => (x * 100.0).round() as i16,
          Err(_) => match fst {
            "+#" => PROVEN_MATE,
            "-#" => PROVEN_LOSS,
            _ => return Err(Error::new(ErrorKind::Other, String::from("unable to parse score")))
          }
        };
      }
    }
    return match State::from_fen_fields(&mut fields) {
      Ok(state) => {
        let flip = sign == ScoreSign::WhitePositive && state.turn == Color::Black;
        Ok((if flip { -score } else { score }, state))
      }
      Err(msg) => Err(Error::new(ErrorKind::Other, msg))
    };
  }
  else {
    return Err(Error::new(ErrorKind::Other, String::from("nothing to parse")));
  }
}

impl Iterator for ScoredFENReader {
  type Item = std::io::Result<(i16, State)>;

  fn next(&mut self) -> Option<Self::Item> {
    let mut buf = String::new();
    return match self.reader.read_line(&mut buf) {
      Err(e) => Some(Err(e)),
      Ok(sz) => if sz == 0 { None } else {
        Some(scored_state_from_fen(&buf, self.unit, self.sign))
      }
    };
  }
}

impl ScoredFENReader {
  pub fn collect(self) -> std::io::Result<Vec<(i16, State)>>
  {
    let mut positions = Vec::new();
    for state in self {
      match state {
        Ok(pos) => { positions.push(pos); }
        Err(e) => { return Err(e); }
      }
    }
    return Ok(positions);
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
// TODO handle games from TCEC that don't have spaces after move numbers

impl Iterator for PGNReader {
  type Item = std::io::Result<Vec<Move>>;

  fn next(&mut self) -> Option<Self::Item> {
    let mut buf = String::new();
    // We skip lines containing metadata (in brackets) and empty lines
    loop {
      self.line += 1;
      let sz = match self.reader.read_line(&mut buf) {
        Ok(x) => x, Err(e) => return Some(Err(e))
      };
      if sz == 0 { return None; }
      if buf.trim_start().chars().next().unwrap_or('[') != '[' { break; }
      buf.clear();
    }
    // We then process the move list, stripping out comments
    //   (note that we don't support nested comments)
    let mut working = State::new();
    let mut movelist : Vec<Move> = Vec::new();
    loop {
      let mut text = String::new();
      let mut comment = false;
      for a in buf.chars() {
        if comment {
          if a == '}' {
            if !comment {
              let e = Error::new(ErrorKind::Other, String::from("unpaired brace"));
              return Some(Err(e));
            }
            comment = false;
          }
        }
        else {
          if a == '{' {
            if comment {
              let e = Error::new(ErrorKind::Other, String::from("nested brace"));
              return Some(Err(e));
            }
            comment = true;
            text.push(' ');
          }
          else {
            if ![';', '!', '?'].contains(&a) { text.push(a); }
          }
        }
      }
      for token in text.split_ascii_whitespace() {
        if token.ends_with('.') {
          let movenum = token.trim_end_matches('.');
          if !movenum.is_empty() {
            match movenum.parse::<u16>() {
              // Move numbers might be repeated after a comment, so we
              //   check correctness down to the move, not to the ply
              Ok(x) => if x == 0 || working.ply/2 != x-1 {
                let e = Error::new(ErrorKind::Other, String::from("move number mismatch"));
                return Some(Err(e));
              }
              Err(_) => {
                let e = Error::new(ErrorKind::Other, String::from("invalid move number"));
                return Some(Err(e));
              }
            }
          }
          continue;
        }
        if token=="1-0" || token=="0-1" || token=="1/2-1/2" || token=="*" {
          return Some(Ok(movelist));
        }
        let mv = match parse_short(&working, token) {
          Ok(m) => m, Err(msg) => return Some(Err(Error::new(ErrorKind::Other, msg)))
        };
        working.apply(&mv);
        movelist.push(mv);
      }
      buf.clear();
      self.line += 1;
      let sz = match self.reader.read_line(&mut buf) {
        Ok(x) => x, Err(e) => return Some(Err(e))
      };
      // Games are definitively completed by reading a result (1-0, 1/2-1/2, 0-1, or *)
      //   but we also accept EOF or an empty line
      if sz == 0 { return Some(Ok(movelist)); }
      if buf.trim_start().is_empty() { return Some(Ok(movelist)); }
    }
  }
}

impl PGNReader {
  pub fn collect(self) -> std::io::Result<Vec<Vec<Move>>>
  {
    let mut games = Vec::new();
    for state in self {
      match state {
        Ok(gm) => { games.push(gm); }
        Err(e) => { return Err(e); }
      }
    }
    return Ok(games);
  }
}
