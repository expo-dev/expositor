use crate::algebraic::*;
use crate::color::*;
use crate::movetype::*;
use crate::state::*;
use crate::score::*;

use std::fs::File;
use std::io::{BufReader, BufRead, Seek, Error};

const BUFSIZE : usize = 2097152;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
// ↓↓↓ TEMPORARY ↓↓↓

pub fn canonicalize(path : &str) -> std::io::Result<()>
{
  let reader = BufReader::with_capacity(BUFSIZE, File::open(path)?);
  for line in reader.lines() {
    let line = line?;
    let mut fields = line.split_ascii_whitespace();
    if let Some(score) = fields.next() {
      for f in fields { print!("{f} "); }
      println!("= {score}");
    }
  }
  return Ok(());
}

// ↑↑↑ TEMPORARY ↑↑↑
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
// The outcome codes are "w" (1-0), "b" (0-1), "d" (1/2-1/2), and "*" (incomplete),
//   e.g. "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 = +5.0 w".

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ScoreUnit {
  IntegerCentipawn,
  FractionalPawn,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ScoreSign {
  LeaveUnchanged,
  FlipWhenBlackToMove,
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

pub struct FENReader {
  reader : BufReader<File>,
  buffer : String,
  scored : bool,
  unit   : ScoreUnit,
  sign   : ScoreSign
}

pub struct PGNReader {
  reader : BufReader<File>,
  buffer : String,
  line   : usize
}

impl FENReader {
  pub fn open(path : &str) -> std::io::Result<Self> {
    return Ok(Self {
      reader: BufReader::with_capacity(BUFSIZE, File::open(path)?),
      buffer: String::new(),
      scored: false,
      unit: ScoreUnit::IntegerCentipawn,
      sign: ScoreSign::LeaveUnchanged
    });
  }

  pub fn open_scored(path : &str, unit : ScoreUnit, sign : ScoreSign) -> std::io::Result<Self> {
    return Ok(Self {
      reader: BufReader::with_capacity(BUFSIZE, File::open(path)?),
      buffer: String::new(),
      scored: true,
      unit: unit,
      sign: sign
    });
  }
}

impl PGNReader {
  pub fn open(path : &str) -> std::io::Result<Self> {
    return Ok(Self {
      reader: BufReader::with_capacity(BUFSIZE, File::open(path)?),
      buffer: String::new(),
      line: 0
    });
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[inline]
fn parse_fen_line(
  line   : &str,
  scored : bool,
  unit   : ScoreUnit,
  sign   : ScoreSign
) -> std::io::Result<(State, i16, i16)> // (position, score, outcome)
{
  let mut fields = line.split_ascii_whitespace();
  let state =
    match State::from_fen_fields(&mut fields) {
      Ok(s) => s, Err(msg) => return Err(Error::other(msg))
    };

  if !scored { return Ok((state, i16::MIN, i16::MIN)); }  // ignore trailing fields

  match fields.next() {
    None => { return Ok((state, i16::MIN, i16::MIN)); }
    Some(delim) => { if delim != "=" { return Err(Error::other("missing delimiter")); } }
  }

  let fst = match fields.next() {
    Some(a) => a, None => return Err(Error::other("missing score or outcome"))
  };

  loop {
    let outcome;
    match fst {
      "w" => { outcome =  1; }
      "d" => { outcome =  0; }
      "b" => { outcome = -1; }
      "*" => { outcome = i16::MIN; }
       _  => { break; }
    }
    if fields.next().is_none() {
      return Ok((state, i16::MIN, outcome));
    }
    else {
      return Err(Error::other("trailing fields"));
    }
  }

  let mut score;
  match unit {
    ScoreUnit::IntegerCentipawn => {
      score = match fst.parse::<i16>() {
         Ok(x) => x,
        Err(_) => match fst {
          "+#" => PROVEN_MATE,
          "-#" => PROVEN_LOSS,
          _ => return Err(Error::other("unable to parse score"))
        }
      };
    }
    ScoreUnit::FractionalPawn => {
      score = match fst.parse::<f32>() {
         Ok(x) => (x * 100.0).round() as i16,
        Err(_) => match fst {
          "+#" => PROVEN_MATE,
          "-#" => PROVEN_LOSS,
          _ => return Err(Error::other("unable to parse score"))
        }
      };
    }
  }

  if sign == ScoreSign::FlipWhenBlackToMove && state.turn == Color::Black { score = -score; }

  let mut outcome = i16::MIN;
  if let Some(snd) = fields.next() {
    match snd {
      "w" => { outcome =  1; }
      "d" => { outcome =  0; }
      "b" => { outcome = -1; }
      "*" => { /* nothing */ }
       _  => { return Err(Error::other("invalid outcome")); }
    }
  }

  return Ok((state, score, outcome));
}

impl Iterator for FENReader {
  type Item = std::io::Result<(State, i16, i16)>;

  fn next(&mut self) -> Option<Self::Item> {
    self.buffer.clear();
    return match self.reader.read_line(&mut self.buffer) {
      Err(e) => Some(Err(e)),
      Ok(sz) => if sz == 0 { None } else {
        Some(parse_fen_line(&self.buffer, self.scored, self.unit, self.sign))
      }
    };
  }
}

impl FENReader {
  pub fn collect(self) -> std::io::Result<Vec<(State, i16, i16)>>
  {
    let mut fen = Vec::new();
    for state in self {
      match state {
        Ok(f) => { fen.push(f); }
        Err(e) => { return Err(e); }
      }
    }
    return Ok(fen);
  }

  // This loops around to the beginning of the file whenever the end is reached.
  pub fn collect_exactly(&mut self, n : usize) -> std::io::Result<Vec<(State, i16, i16)>>
  {
    let mut fen = Vec::new();
    for _ in 0..n {
      self.buffer.clear();
      if self.reader.read_line(&mut self.buffer)? == 0 {
        self.reader.rewind()?;
        if self.reader.read_line(&mut self.buffer)? == 0 {
          panic!("file appears to be empty");
        }
      }
      fen.push(parse_fen_line(&self.buffer, self.scored, self.unit, self.sign)?);
    }
    return Ok(fen);
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

// wiki: abandoned, adjudication, normal, time forfeit, unterminated
// cute: abandoned, adjudication, time forfeit, unterminated, illegal move, stalled connection
//       (no normal termination – tag is simply omitted)
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
  pub initial     : State,
  pub movelist    : Vec<Move>,
}

fn parse_metadata(tail : &mut std::str::Chars, game : &mut PGNGame) -> Option<&'static str>
{
  // Leading whitespace and the opening bracket have been removed by this point.
  //   We consider /^ *([^ ]+) +"(.+?)"/ to be valid: there must be space after
  //   the key, quotes are not escapable in the value, and the trailing bracket
  //   (and everything thereafter) is ignored. In theory, the key can contain
  //   quotes (but can never contain spaces).
  let tail = tail.as_str().trim_start();
  let (key, tail) = match tail.split_once(' ') {
    Some(pair) => pair, None => return Some("missing value")
  };
  let mut tail = tail.trim_start().chars();
  let head = match tail.next() { Some(a) => a, None => return Some("missing value") };
  if head != '"' { return Some("missing quote"); }
  let tail = tail.as_str();
  let (value, _) = match tail.split_once('"') {
    Some(pair) => pair,
    None => return Some("missing quote")
  };
  match key {
    "FEN" => {
      game.initial = match State::from_fen(value) {
        Ok(s) => s, Err(_) => return Some("unable to parse fen")
      };
    }
    "Result" => {
      match value {
        "1-0"     => { game.result = GameResult::White; }
        "0-1"     => { game.result = GameResult::Black; }
        "1/2-1/2" => { game.result = GameResult::Draw;  }
        _ => {}
      }
    }
    "Termination" => {
      match value.to_ascii_lowercase().as_str() {
        "normal"       => { game.termination = GameTermination::Normal; }
        "time forfeit" => { game.termination = GameTermination::Flag;   }
        _ => {}
      }
    }
    "TimeControl" => {
      if let Some((left, right)) = value.split_once('+') {
        if let Ok(base) = left.parse::<f32>() { game.base = base; }
          else { return Some("unable to parse base"); }
        if let Ok(incr) = right.parse::<f32>() { game.increment = incr; }
          else { return Some("unable to parse increment"); }
      }
      else {
        return Some("unable to parse time control");
      }
    }
    _ => {}
  }
  return None;
}

impl Iterator for PGNReader {
  type Item = std::io::Result<PGNGame>;

  fn next(&mut self) -> Option<Self::Item> {
    let mut game = PGNGame {
      base:        0.0,
      increment:   0.0,
      result:      GameResult::Incomplete,
      termination: GameTermination::Normal,
      initial:     State::new(),
      movelist:    Vec::new()
    };
    // We skip empty lines and parse lines containing metadata (in brackets)
    loop {
      self.buffer.clear();
      self.line += 1;
      let sz = match self.reader.read_line(&mut self.buffer) {
        Ok(x) => x, Err(e) => return Some(Err(e))
      };
      if sz == 0 { return None; }
      let mut tail = self.buffer.trim_start().chars();
      if let Some(head) = tail.next() {
        if head != '[' { break; }
        let err = parse_metadata(&mut tail, &mut game);
        if let Some(msg) = err { return Some(Err(Error::other(msg))); }
      }
    }
    // We then process the move list
    let mut working = game.initial.clone();
    loop {
      // First we strip out the comments and annotations
      //   (note that we don't support nested comments)
      let mut text = String::new();
      let mut comment = false;
      for a in self.buffer.chars() {
        if comment {
          if a == '}' {
            if !comment { return Some(Err(Error::other("unpaired brace"))); }
            comment = false;
          }
        }
        else {
          if a == '{' {
            if comment { return Some(Err(Error::other("nested brace"))); }
            comment = true;
            text.push(' ');
          }
          else if a == '.' {
            text.push_str(". ");
          }
          else if ![';', '!', '?'].contains(&a) {
            text.push(a);
          }
        }
      }
      // Then we parse the moves
      for token in text.split_ascii_whitespace() {
        if token.ends_with('.') { continue; }
        if token=="1-0" || token=="0-1" || token=="1/2-1/2" || token=="*" {
          let list_result = match token {
            "1-0"     => GameResult::White,
            "0-1"     => GameResult::Black,
            "1/2-1/2" => GameResult::Draw,
            _         => GameResult::Incomplete
          };
          if list_result != GameResult::Incomplete {
            if game.result != GameResult::Incomplete && list_result != game.result {
              return Some(Err(Error::other("game result mismatch")));
            }
            game.result = list_result;
          }
          return Some(Ok(game));
        }
        let mv = match parse_short(&working, token) {
          Ok(m) => m, Err(msg) => return Some(Err(Error::other(msg)))
        };
        working.apply(&mv);
        game.movelist.push(mv);
      }
      self.buffer.clear();
      self.line += 1;
      let sz = match self.reader.read_line(&mut self.buffer) {
        Ok(x) => x, Err(e) => return Some(Err(e))
      };
      // Games are definitively completed by reading a result (1-0, 1/2-1/2, 0-1, or *)
      //   but we also accept EOF or an empty line
      if sz == 0 { return Some(Ok(game)); }
      if self.buffer.trim_start().is_empty() { return Some(Ok(game)); }
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
