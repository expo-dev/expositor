use crate::movetype::Move;
use crate::piece::Piece;

type KillerTable  =    [(Move, Move); 128]; // [height]
type HistoryTable = [[(i16, i16); 64]; 16]; // [piece][to-sq] -> (score, score)

pub struct Context {                    // Read         Written       Index
  pub gainful       : [bool; 128],      //   resolving    resolving     length
  pub killer_table  : KillerTable,      //   main         main          height
  pub history_table : HistoryTable,     //   main         main          ...
  pub state_history : Vec<(u64, bool)>, //   main         setup+main    ...
  pub null          : [bool; 128],      //   main         main          height
  pub exclude       : [Move; 128],      //   main         main          height
  pub nominal       : u8,               //   main         setup         ...
  pub pv            : [Vec<Move>; 128], //   main         main          height
  // Statistics
  pub m_nodes_at_height : [usize; 128],
  pub r_nodes_at_height : [usize; 128],
  pub r_nodes_at_length : [usize; 128],
  pub tb_hits : usize
}

impl Context {
  pub const fn new() -> Self
  {
    const NULL_PAIR : (Move, Move) = (Move::NULL, Move::NULL);
    return Self {
      gainful:       [false; 128],
      killer_table:  [NULL_PAIR; 128],
      history_table: [[(0, 0); 64]; 16],
      state_history: Vec::new(),
      null:          [false; 128],
      exclude:       [Move::NULL; 128],
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
      ],
      m_nodes_at_height: [0; 128],
      r_nodes_at_height: [0; 128],
      r_nodes_at_length: [0; 128],
      tb_hits: 0
    };
  }

  pub fn reset(&mut self)
  {
    const NULL_PAIR : (Move, Move) = (Move::NULL, Move::NULL);
    self.gainful       = [false; 128];
    self.killer_table  = [NULL_PAIR; 128];
    self.history_table = [[(0, 0); 64]; 16];
    self.state_history.clear();
    self.null          = [false; 128];
    self.exclude       = [Move::NULL; 128];
    self.nominal       =  0;
    for x in 0..128 { self.pv[x].clear(); }
    self.m_nodes_at_height = [0; 128];
    self.r_nodes_at_height = [0; 128];
    self.r_nodes_at_length = [0; 128];
    self.tb_hits = 0;
  }

  pub fn partial_reset(&mut self)
  {
    // does not clear state_history or statistics
    const NULL_PAIR : (Move, Move) = (Move::NULL, Move::NULL);
    self.gainful       = [false; 128];
    self.killer_table  = [NULL_PAIR; 128];
    self.history_table = [[(0, 0); 64]; 16];
    self.null          = [false; 128];
    self.exclude       = [Move::NULL; 128];
    self.nominal       =  0;
    for x in 0..128 { self.pv[x].clear(); }
  }

  pub fn lookup_history(&self, piece : Piece, dst : usize, height : u8) -> i8
  {
    let entry  = &self.history_table[piece][dst];
    let height = std::cmp::min(height as i32, 32);
    let score  = ((entry.0 as i32)*(32 - height) + (entry.1 as i32)*height) / (32*256);
    return score as i8;
  }

  pub fn update_history(&mut self, piece : Piece, dst : usize, height : u8, cutoff : bool)
  {
    let entry = &self.history_table[piece][dst];
    let height = std::cmp::min(height as i16, 32);
    let deep;
    let shallow;
    if cutoff {
      deep    = std::cmp::min(32_000, entry.0 + (32 - height));
      shallow = std::cmp::min(32_000, entry.1 + height       );
    }
    else {
      deep    = std::cmp::max(-32_000, entry.0 - (32 - height));
      shallow = std::cmp::max(-32_000, entry.1 - height       );
    }
    self.history_table[piece][dst] = (deep, shallow);
  }

  // pub fn print_history(&self)
  // {
  //   for color in 0..2 {
  //     for kind in 0..6 {
  //       let piece = color*8 + kind;
  //       eprintln!("{}", FULLNAME[piece]);
  //       for dst in 0..64 {
  //         let (deep, shallow) = self.history_table[piece][dst];
  //         let deep_color =
  //           if deep > 0 { "\x1B[92m" } else if deep < 0 { "\x1B[91m" } else { "" };
  //         let shallow_color =
  //           if shallow > 0 { "\x1B[92m" } else if shallow < 0 { "\x1B[91m" } else { "" };
  //         eprintln!(
  //           "  {} {}{:+6}\x1B[39m {}{:+6}\x1B[39m",
  //           (dst as u8).algebraic(), deep_color, deep, shallow_color, shallow
  //         );
  //       }
  //     }
  //   }
  // }
}
