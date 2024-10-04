use crate::movetype::Move;
use crate::score::{CoOutcome, PovScore};
use crate::state::{MiniState, State};

use std::fs::File;
use std::io::{BufReader, BufRead, Seek, Error};

const BUFSIZE : usize = 2_097_152;

pub struct QuickReader {
  reader : BufReader<File>,
  buffer : String,
  line   : usize
}

pub struct FenReader {
  reader : BufReader<File>,
  buffer : String,
  line   : usize,
}

pub struct PgnReader {
  reader : BufReader<File>,
  buffer : String,
  line   : usize
}

impl QuickReader {
  pub fn open(path : &str) -> std::io::Result<Self> {
    return Ok(Self {
      reader: BufReader::with_capacity(BUFSIZE, File::open(path)?),
      buffer: String::new(),
      line: 0
    });
  }
}

impl FenReader {
  pub fn open(path : &str) -> std::io::Result<Self> {
    return Ok(Self {
      reader: BufReader::with_capacity(BUFSIZE, File::open(path)?),
      buffer: String::new(),
      line: 0
    });
  }
}

impl PgnReader {
  pub fn open(path : &str) -> std::io::Result<Self> {
    return Ok(Self {
      reader: BufReader::with_capacity(BUFSIZE, File::open(path)?),
      buffer: String::new(),
      line: 0
    });
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

impl Iterator for QuickReader {
  type Item = std::io::Result<MiniState>;

  fn next(&mut self) -> Option<Self::Item> {
    self.buffer.clear();
    self.line += 1;
    return match self.reader.read_line(&mut self.buffer) {
      Err(e) => Some(Err(e)),
      Ok(sz) =>
        if sz == 0 {
          None
        }
        else if sz != 41 {
          Some(Err(Error::other(format!("{}: incorrect length {}", self.line, sz))))
        }
        else {
          let ptr = self.buffer.as_ptr() as *const [u8; 40];
          let record = unsafe { &*ptr };
          Some(Ok(MiniState::from_quick(record)))
        }
    };
  }
}

impl QuickReader {
  pub fn collect(self) -> std::io::Result<Vec<MiniState>>
  {
    let mut ms = Vec::new();
    for state in self {
      match state {
        Ok(m) => { ms.push(m); }
        Err(e) => { return Err(e); }
      }
    }
    return Ok(ms);
  }

  // This loops around to the beginning of the file whenever the end is reached.
  pub fn take(&mut self, n : usize) -> std::io::Result<Vec<MiniState>>
  {
    let mut ms = Vec::new();
    for _ in 0..n {
      self.buffer.clear();
      let mut sz = self.reader.read_line(&mut self.buffer)?;
      if sz == 0 {
        self.reader.rewind()?;
        self.line = 0;
        sz = self.reader.read_line(&mut self.buffer)?;
        if sz == 0 { return Err(Error::other("file appears to be empty")); }
      }
      self.line += 1;
      if sz != 41 {
        return Err(Error::other(format!("{}: incorrect length {}", self.line, sz)));
      }
      let ptr = self.buffer.as_ptr() as *const [u8; 40];
      let record = unsafe { &*ptr };
      ms.push(MiniState::from_quick(record));
    }
    return Ok(ms);
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

impl Iterator for FenReader {
  type Item = std::io::Result<State>;

  fn next(&mut self) -> Option<Self::Item> {
    self.buffer.clear();
    self.line += 1;
    return match self.reader.read_line(&mut self.buffer) {
      Err(e) => Some(Err(e)),
      Ok(sz) =>
        if sz == 0 {
          None
        }
        else {
          let mut fields = self.buffer.split_ascii_whitespace();
          match State::from_fen_fields(&mut fields) {
            Ok(s) => Some(Ok(s)),
            Err(msg) => Some(Err(Error::other(format!("{}: {}", self.line, msg))))
          }
        }
    };
  }
}

impl FenReader {
  pub fn collect(self) -> std::io::Result<Vec<State>>
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
  pub fn take(&mut self, n : usize) -> std::io::Result<Vec<State>>
  {
    let mut fen = Vec::new();
    for _ in 0..n {
      self.buffer.clear();
      if self.reader.read_line(&mut self.buffer)? == 0 {
        self.reader.rewind()?;
        self.line = 0;
        if self.reader.read_line(&mut self.buffer)? == 0 {
          return Err(Error::other("file appears to be empty"));
        }
      }
      let mut fields = self.buffer.split_ascii_whitespace();
      match State::from_fen_fields(&mut fields) {
        Ok(s) => { fen.push(s); }
        Err(msg) => { return Err(Error::other(format!("{}: {}", self.line, msg))); }
      }
    }
    return Ok(fen);
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// wiki: abandoned, adjudication, normal, time forfeit, unterminated
// cute: abandoned, adjudication, time forfeit, unterminated, illegal move, stalled connection
//       (no normal termination – tag is simply omitted)
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Termination {
  Normal,
  Flag,
}

// NOTE that the scores are taken verbatim; if the game is from cutechess, that means the scores
//   are point of view (not white positive) and that the n'th score is the engine evaluation
//   before the n'th move was made (not after).

pub struct PgnGame {
  pub base        : f32,
  pub increment   : f32,
  pub result      : CoOutcome,
  pub termination : Termination,
  pub initial     : State,
  pub movelist    : Vec<Move>,
  pub scorelist   : Vec<PovScore>
}

fn parse_metadata(tail : &mut std::str::Chars, game : &mut PgnGame) -> Option<&'static str>
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
        "1-0"     => { game.result = CoOutcome::White; }
        "0-1"     => { game.result = CoOutcome::Black; }
        "1/2-1/2" => { game.result = CoOutcome::Draw;  }
        _ => {}
      }
    }
    "Termination" => {
      match value.to_ascii_lowercase().as_str() {
        "normal"       => { game.termination = Termination::Normal; }
        "time forfeit" => { game.termination = Termination::Flag;   }
        _ => {}
      }
    }
    "TimeControl" => {
      if let Some((left, right)) = value.split_once('+') {
        if let Ok(base) = left.parse::<f32>() {
          if let Ok(incr) = right.parse::<f32>() {
            game.base = base;
            game.increment = incr;
          }
        }
      }
      // if let Some((left, right)) = value.split_once('+') {
      //   if let Ok(base) = left.parse::<f32>() { game.base = base; }
      //     else { return Some("unable to parse base"); }
      //   if let Ok(incr) = right.parse::<f32>() { game.increment = incr; }
      //     else { return Some("unable to parse increment"); }
      // }
      // else {
      //   return Some("unable to parse time control");
      // }
    }
    _ => {}
  }
  return None;
}

impl Iterator for PgnReader {
  type Item = std::io::Result<PgnGame>;

  fn next(&mut self) -> Option<Self::Item> {
    let mut game = PgnGame {
      base:        0.0,
      increment:   0.0,
      result:      CoOutcome::Unknown,
      termination: Termination::Normal,
      initial:     State::new(),
      movelist:    Vec::new(),
      scorelist:   Vec::new()
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
      let mut annot = String::new();
      let mut comment = false;
      for a in self.buffer.chars() {
        if comment {
          if a == '}' {
            if !comment { return Some(Err(Error::other("unpaired brace"))); }
            comment = false;
            annot.clear();
          }
          else if a == '/' {
            // If this looks like a cutechess annotation, e.g. {+0.31/10 0.41s}
            //   or {-M12/18 0.15s}, try to parse a score
            if let Ok(x) = annot.parse::<f32>() {
              game.scorelist.push(PovScore::new((x*100.0) as i16));
            }
            else if annot.starts_with("+M") || annot.starts_with("-M") {
              if let Ok(n) = annot[2..].parse::<u8>() {
                game.scorelist.push(match annot.as_bytes()[0] as char {
                  '+' => PovScore::realized_win(n),
                  '-' => PovScore::realized_loss(n),
                   _  => unreachable!()
                });
              }
            }
          }
          else {
            annot.push(a);
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
        if token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*" {
          let list_result = match token {
            "1-0"     => CoOutcome::White,
            "0-1"     => CoOutcome::Black,
            "1/2-1/2" => CoOutcome::Draw,
            _         => CoOutcome::Unknown
          };
          if list_result != CoOutcome::Unknown {
            if game.result != CoOutcome::Unknown && list_result != game.result {
              return Some(Err(Error::other("game result mismatch")));
            }
            game.result = list_result;
          }
          // If the number of evaluations doesn't match the number of moves,
          //   this isn't a cutechess game or we missed / failed parsing an
          //   annotation (in which case the correspondence is broken)
          if game.scorelist.len() > 0 && game.scorelist.len() != game.movelist.len() {
            game.scorelist = Vec::new();
          }
          return Some(Ok(game));
        }
        let mv = match working.parse(token) {
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
      if sz == 0 || self.buffer.trim_start().is_empty() {
        if game.scorelist.len() > 0 && game.scorelist.len() != game.movelist.len() {
          game.scorelist = Vec::new();
        }
        return Some(Ok(game));
      }
    }
  }
}

impl PgnReader {
  pub fn collect(self) -> std::io::Result<Vec<PgnGame>>
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
