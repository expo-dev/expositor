use crate::constants::{SearchConstants, DEFAULT_CONSTANTS};
use crate::movetype::Move;
use crate::piece::Piece;
use crate::search::MAX_HEIGHT;

type KillerTable  = [(Move, Move); MAX_HEIGHT];       // [height]
type HistoryTable = [[[(i16, i16, i16); 64]; 2]; 16]; // [piece][direciton][to-sq] -> (scores, ...)

pub struct Context {
  // Constant
  pub tuning        : SearchConstants,
  // Dynamic                                       Read         Written       Index
  pub gainful       : [bool; MAX_HEIGHT],       //   resolving    resolving     length
  pub killer_table  : KillerTable,              //   main         main          height
  pub history_table : HistoryTable,             //   main         main          ...
  pub state_history : Vec<(u64, bool)>,         //   main         setup+main    ...
  pub null          : [bool; MAX_HEIGHT],       //   main         main          height
  pub exclude       : [Move; MAX_HEIGHT],       //   main         main          height
  pub nominal       : u8,                       //   main         setup         ...
  pub pv            : [Vec<Move>; MAX_HEIGHT],  //   main         main          height
  // Statistics
  pub m_nodes_at_height : [usize; MAX_HEIGHT],
  pub r_nodes_at_height : [usize; MAX_HEIGHT],
  pub r_nodes_at_length : [usize; MAX_HEIGHT],
  pub tb_hits : usize
}

impl Context {
  pub const fn new() -> Self
  {
    const NULL_PAIR : (Move, Move) = (Move::NULL, Move::NULL);
    return Self {
      tuning:        DEFAULT_CONSTANTS,
      gainful:       [false; MAX_HEIGHT],
      killer_table:  [NULL_PAIR; MAX_HEIGHT],
      history_table: [[[(0, 0, 0); 64]; 2]; 16],
      state_history: Vec::new(),
      null:          [false; MAX_HEIGHT],
      exclude:       [Move::NULL; MAX_HEIGHT],
      nominal:       0,
      pv: [
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), Vec::new(), Vec::new(), Vec::new(),
      ],
      m_nodes_at_height: [0; MAX_HEIGHT],
      r_nodes_at_height: [0; MAX_HEIGHT],
      r_nodes_at_length: [0; MAX_HEIGHT],
      tb_hits: 0
    };
  }

  pub const fn from(parameters : &SearchConstants) -> Self
  {
    let mut context = Self::new();
    context.tuning = *parameters;
    return context;
  }

  pub fn reset(&mut self)
  {
    self.state_history.clear();
    self.reset_search();
    self.reset_stats();
  }

  pub fn reset_search(&mut self)
  {
    // does not clear state history or statistics
    const NULL_PAIR : (Move, Move) = (Move::NULL, Move::NULL);
    self.gainful       = [false; MAX_HEIGHT];
    self.killer_table  = [NULL_PAIR; MAX_HEIGHT];
    self.history_table = [[[(0, 0, 0); 64]; 2]; 16];
    self.null          = [false; MAX_HEIGHT];
    self.exclude       = [Move::NULL; MAX_HEIGHT];
    self.nominal       =  0;
    for x in 0..MAX_HEIGHT { self.pv[x].clear(); }
  }

  pub fn reset_stats(&mut self)
  {
    // does not clear state history or search context
    self.m_nodes_at_height = [0; MAX_HEIGHT];
    self.r_nodes_at_height = [0; MAX_HEIGHT];
    self.r_nodes_at_length = [0; MAX_HEIGHT];
    self.tb_hits = 0;
  }

  pub fn nodes(&self) -> usize
  {
    return self.m_nodes_at_height.iter().sum::<usize>()
         + self.r_nodes_at_height.iter().sum::<usize>();
  }

  pub fn lookup_history(&self, piece : Piece, src : usize, dst : usize) -> i8
  {
    let dir = if src > dst { 1 } else { 0 };
    let entry = &self.history_table[piece][dir][dst];
    let score = (entry.0 as i32 + entry.1 as i32 + entry.2 as i32) / 768;
    return score as i8;
  }

  pub fn update_history(&mut self, piece : Piece, src : usize, dst : usize, cutoff : bool)
  {
    let dir = if src > dst { 1 } else { 0 };
    let entry = &mut self.history_table[piece][dir][dst];
    if cutoff {
      entry.0 = ((entry.0 as i32 *   16 - entry.0 as i32 + 32767) >>  4) as i16;
      entry.1 = ((entry.1 as i32 *  128 - entry.1 as i32 + 32767) >>  7) as i16;
      entry.2 = ((entry.2 as i32 * 1024 - entry.2 as i32 + 32767) >> 10) as i16;
    }
    else {
      entry.0 = ((entry.0 as i32 *   16 - entry.0 as i32 - 32768) >>  4) as i16;
      entry.1 = ((entry.1 as i32 *  128 - entry.1 as i32 - 32768) >>  7) as i16;
      entry.2 = ((entry.2 as i32 * 1024 - entry.2 as i32 - 32768) >> 10) as i16;
    }
  }

  // pub fn print_history(&self)
  // {
  //   const FULLNAME : [&str; 6] = ["King", "Queen", "Rook", "Bishop", "Knight", "Pawn"];
  //   for color in 0..2 {
  //     for kind in 0..6 {
  //       let piece = color*8 + kind;
  //       eprintln!("{} {}", ["White", "Black"][color], FULLNAME[kind]);
  //       for dst in 0..64 {
  //         let (deep, shallow) = self.history_table[piece][dst];
  //         let deep_color =
  //           if deep > 0 { "\x1B[92m" } else if deep < 0 { "\x1B[91m" } else { "" };
  //         let shallow_color =
  //           if shallow > 0 { "\x1B[92m" } else if shallow < 0 { "\x1B[91m" } else { "" };
  //         eprintln!(
  //           "  {} {}{:+6}\x1B[39m {}{:+6}\x1B[39m",
  //           (dst as u8).id(), deep_color, deep, shallow_color, shallow
  //         );
  //       }
  //     }
  //   }
  // }
}
