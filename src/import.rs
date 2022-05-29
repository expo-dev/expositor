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

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum GameResult {
  White,      // 1-0
  Black,      // 0-1
  Draw,       // ½-½
  Incomplete, //  *
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum GameTermination {
  Normal,
  Flag,
}

pub struct PGNGame {
  pub base        : f32,
  pub increment   : f32,
  pub result      : GameResult,
  pub termination : GameTermination,
  pub expo_white  : bool,
  pub expo_black  : bool,
  pub white_elo   : u16,
  pub black_elo   : u16,
  pub movelist    : Vec<Move>,
}

impl Iterator for PGNReader {
  type Item = std::io::Result<PGNGame>;

  // TODO this is a mess and probably ought to be rewritten
  fn next(&mut self) -> Option<Self::Item> {
    let mut game = PGNGame {
      base:        0.0,
      increment:   0.0,
      result:      GameResult::Incomplete,
      termination: GameTermination::Normal,
      expo_white:  false,
      expo_black:  false,
      white_elo:   0,
      black_elo:   0,
      movelist:    Vec::new()
    };
    let mut buf = String::new();
    // We skip empty lines and parse lines containing metadata (in brackets)
    loop {
      let sz = match self.reader.read_line(&mut buf) {
        Ok(x) => x, Err(e) => return Some(Err(e))
      };
      self.line += 1;
      if sz == 0 { return None; }
      let mut tail = buf.trim_start().chars();
      if let Some(head) = tail.next() {
        if head != '[' { break; }
        // Leading whitespace and the opening bracket have been removed by this point.
        //   We consider /^ *([^ ]+) +"(.+?)"/ to be valid: there must be space after the
        //   key, quotes are not escapable in the value, the trailing bracket is ignored.
        //   In theory, the key can contain quotes (but can never contain spaces).
        let tail = tail.as_str().trim_start();
        if let Some((key, tail)) = tail.split_once(' ') {
          let mut tail = tail.trim_start().chars();
          if let Some(head) = tail.next() {
            let tail = tail.as_str();
            if head == '"' {
              if let Some((value, _)) = tail.split_once('"') {
                match key {
                  "White"  => {
                    if value.starts_with("Expo") || value.starts_with("expo") {
                      game.expo_white = true;
                    }
                  }
                  "Black"  => {
                    if value.starts_with("Expo") || value.starts_with("expo") {
                      game.expo_black = true;
                    }
                  }
                  "Result" => {
                    match value {
                      "1-0"     => { game.result = GameResult::White; }
                      "0-1"     => { game.result = GameResult::Black; }
                      "1/2-1/2" => { game.result = GameResult::Draw;  }
                      _ => {}
                    }
                  }
                  "TimeControl" => {
                    if let Some((left, right)) = value.split_once('+') {
                      if let Ok(base) =  left.parse::<f32>() { game.base      = base; }
                      if let Ok(incr) = right.parse::<f32>() { game.increment = incr; }
                    }
                  }
                  "Termination" => {
                    match value {
                      "Normal"       => { game.termination = GameTermination::Normal; }
                      "Time forfeit" => { game.termination = GameTermination::Flag;   }
                      _ => {}
                    }
                  }
                  "WhiteElo" => {
                    if let Ok(rating) = value.parse::<u16>() { game.white_elo = rating; }
                  }
                  "BlackElo" => {
                    if let Ok(rating) = value.parse::<u16>() { game.black_elo = rating; }
                  }
                  _ => {}
                }
              }
            }
          }
        }
      }
      buf.clear();
    }
    // We then process the move list, stripping out comments
    //   (note that we don't support nested comments)
    let mut working = State::new();
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
      // TODO handle games from TCEC that don't have spaces after move numbers
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
          let list_result = match token {
            "1-0"     => GameResult::White,
            "0-1"     => GameResult::Black,
            "1/2-1/2" => GameResult::Draw,
            _         => GameResult::Incomplete
          };
          if list_result != GameResult::Incomplete {
            if game.result != GameResult::Incomplete && list_result != game.result {
              return Some(Err(Error::new(ErrorKind::Other, "game result mismatch")));
            }
            game.result = list_result;
          }
          return Some(Ok(game));
        }
        let mv = match parse_short(&working, token) {
          Ok(m) => m, Err(msg) => return Some(Err(Error::new(ErrorKind::Other, msg)))
        };
        working.apply(&mv);
        game.movelist.push(mv);
      }
      buf.clear();
      self.line += 1;
      let sz = match self.reader.read_line(&mut buf) {
        Ok(x) => x, Err(e) => return Some(Err(e))
      };
      // Games are definitively completed by reading a result (1-0, 1/2-1/2, 0-1, or *)
      //   but we also accept EOF or an empty line
      if sz == 0 { return Some(Ok(game)); }
      if buf.trim_start().is_empty() { return Some(Ok(game)); }
    }
  }
}

impl PGNReader {
  pub fn collect(self) -> std::io::Result<Vec<PGNGame>>
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
