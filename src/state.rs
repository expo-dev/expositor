use crate::basis::START_KEY;
use crate::color::Color::{self, *};
use crate::color::WB;
use crate::misc::{WHITE_HOME, BLACK_HOME};
// use crate::nnue::N1;
use crate::nnext::N1;
use crate::piece::Kind::{self, *};
use crate::piece::Piece::{self, *};
use crate::piece::{KQRBNP, PNBRQK};
use crate::score::{CoScore, CoOutcome};

#[derive(Clone)]
pub struct MiniState {
  pub positions : [[i8; 16]; 2],
  pub variable  : [(Piece, i8); 2],
  pub rights : u8,
  pub packed : u8,
  pub score  : CoScore,
}

// TODO combine rights and packed

impl MiniState {
  pub const KIND : [Kind; 16] = [
    King, Queen, Rook, Rook, Bishop, Bishop, Knight, Knight,
    Pawn, Pawn,  Pawn, Pawn, Pawn,   Pawn,   Pawn,   Pawn
  ];

  pub fn from(state : &State) -> Option<Self>
  {
    let mut positions = [[-1; 16]; 2];
    let mut variable  = [(Null, -1); 2];

    let alloc = [1, 1, 2, 2, 2, 8];
    let start = [0, 1, 2, 4, 6, 8];

    let mut n = 0;
    for color in WB {
      for kind in KQRBNP {
        let mut sources = state.boards[color+kind];
        let mut count = 0;
        while sources != 0 {
          let src = sources.trailing_zeros() as i8;
          if count >= alloc[kind] {
            if n >= 2 { return None; }
            variable[n] = (color+kind, src);
            n += 1;
          }
          else {
            let idx = start[kind] + count;
            positions[color][idx] = src;
          }
          count += 1;
          sources &= sources - 1;
        }
      }
    }
    return Some(MiniState {
      positions: positions,
      variable:  variable,
      rights: state.rights,
      packed: 0b_1010 | state.turn as u8,
      score:  CoScore::SENTINEL,
    });
  }

  #[inline]
  pub const fn turn(&self) -> Color
  {
    return unsafe { std::mem::transmute(self.packed & 0b_0001) };
  }

  #[inline]
  pub fn outcome(&self) -> CoOutcome
  {
    // 101. incomplete = i8::MIN
    // 000. black win  = -1
    // 001. draw       =  0
    // 010. white win  = +1
    let p = self.packed as u16;
    let d = (p << 4 | p >> 1) & 0b_1000_0011;
    return unsafe { std::mem::transmute((d as i8).wrapping_add(-1)) };
  }

  pub fn set_outcome(&mut self, outcome : CoOutcome)
  {
    let new = match outcome {
      CoOutcome::Unknown => 0b_1010,
      CoOutcome::Black   => 0b_0000,
      CoOutcome::Draw    => 0b_0010,
      CoOutcome::White   => 0b_0100,
    };
    self.packed = (self.packed & 0b_0001) | new;
  }
}

// I expect the compiler to lay out
//   this struct roughly as follows:
//
//   boards    128   128
//   squares    64   192
//   sides      16   208
//   s1         24   232
//   key         8   240
//   dfz         2   242
//   ply         2   244
//   rights      1   245
//   enpass      1   246
//   incheck     1   247
//   turn        1   248
//  (padding     8   256)

#[derive(Clone)]
pub struct State {
  pub sides   : [u64; 2],     // side   -> composite board
  pub boards  : [u64; 16],    // piece  -> board
  pub squares : [Piece; 64],  // square -> piece

  pub rights  : u8,     // .... qkQK
  pub enpass  : i8,     // square or -1
  pub incheck : bool,   // side to move
  pub turn    : Color,  // side to move
  pub dfz     : u16,    // depth from zeroing
  pub ply     : u16,    // zero-indexed
  pub key     : u64,    // zobrist key

  pub s1 : Vec<[[i16; N1]; 2]>
}

pub struct SavedMetadata {
  pub sides    : [u64; 2],
  pub rights   : u8,
  pub enpass   : i8,
  pub incheck  : bool,
  pub dfz      : u16,
  pub key      : u64,
}

impl State {
  pub const fn new() -> Self
  {
    let state = Self {
      sides: [WHITE_HOME, BLACK_HOME],
      boards: [
        0b_0001_0000,
        0b_0000_1000,
        0b_1000_0001,
        0b_0010_0100,
        0b_0100_0010,
        0b_1111_1111 << 8,
        0,
        0,
        0b_0001_0000 << 56,
        0b_0000_1000 << 56,
        0b_1000_0001 << 56,
        0b_0010_0100 << 56,
        0b_0100_0010 << 56,
        0b_1111_1111 << 48,
        0,
        0,
      ],
      squares: [
        WhiteRook, WhiteKnight, WhiteBishop, WhiteQueen, WhiteKing, WhiteBishop, WhiteKnight, WhiteRook ,
        WhitePawn, WhitePawn  , WhitePawn  , WhitePawn , WhitePawn, WhitePawn  , WhitePawn  , WhitePawn ,
        Null     , Null       , Null       , Null      , Null     , Null       , Null       , Null      ,
        Null     , Null       , Null       , Null      , Null     , Null       , Null       , Null      ,
        Null     , Null       , Null       , Null      , Null     , Null       , Null       , Null      ,
        Null     , Null       , Null       , Null      , Null     , Null       , Null       , Null      ,
        BlackPawn, BlackPawn  , BlackPawn  , BlackPawn , BlackPawn, BlackPawn  , BlackPawn  , BlackPawn ,
        BlackRook, BlackKnight, BlackBishop, BlackQueen, BlackKing, BlackBishop, BlackKnight, BlackRook ,
      ],
      rights:  0b_1111,
      enpass:  -1,
      incheck: false,
      turn:    White,
      dfz:     0,
      ply:     0,
      key:     START_KEY,
      s1:      Vec::new(),
    };
    return state;
  }

  pub fn clone_empty(&self) -> State
  {
    return State {
      sides:   self.sides,
      boards:  self.boards,
      squares: self.squares,
      rights:  self.rights,
      enpass:  self.enpass,
      incheck: self.incheck,
      turn:    self.turn,
      dfz:     self.dfz,
      ply:     self.ply,
      key:     self.key,
      s1:      Vec::new()
    };
  }

  pub fn clone_truncated(&self) -> State
  {
    let mut c = State {
      sides:   self.sides,
      boards:  self.boards,
      squares: self.squares,
      rights:  self.rights,
      enpass:  self.enpass,
      incheck: self.incheck,
      turn:    self.turn,
      dfz:     self.dfz,
      ply:     self.ply,
      key:     self.key,
      s1:      Vec::new()
    };
    let len = self.s1.len();
    if len > 0 { c.s1.push(self.s1[len-1]); }
    return c;
  }

  pub fn truncate(&mut self)
  {
    let old_len = self.s1.len();
    if old_len < 2 { return; }
    let old_end : *const [[i16; N1]; 2] = &self.s1[old_len-1];
    let new_end :   *mut [[i16; N1]; 2] = &mut self.s1[0];
    unsafe { std::ptr::copy_nonoverlapping(old_end, new_end, 1); }
    self.s1.truncate(1);
  }

  pub fn from(mini : &MiniState) -> Self
  {
    let mut state = State {
      sides:   [0; 2],
      boards:  [0; 16],
      squares: [Null; 64],
      rights:  mini.rights,
      enpass:  -1,
      incheck: false,
      turn:    mini.turn(),
      dfz:     0,
      ply:     0,
      key:     0,
      s1:      Vec::new(),
    };
    for color in WB {
      let posns = &mini.positions[color];
      for ofs in 0..16 {
        if posns[ofs] < 0 { continue; }
        let idx = posns[ofs] as usize;
        let kind = MiniState::KIND[ofs];
        state.sides[color]       |= 1 << idx;
        state.boards[color+kind] |= 1 << idx;
        state.squares[idx] = color+kind;
      }
    }
    for n in 0..2 {
      let (piece, posn) = mini.variable[n];
      if posn < 0 { continue; }
      let idx = posn as usize;
      state.sides[piece.color()] |= 1 << idx;
      state.boards[piece]        |= 1 << idx;
      state.squares[idx] = piece;
    }
    state.incheck = state.in_check(state.turn);
    state.key = state.zobrist();
    return state;
  }

  pub fn save(&self) -> SavedMetadata
  {
    return SavedMetadata {
      sides:   self.sides,
      rights:  self.rights,
      enpass:  self.enpass,
      incheck: self.incheck,
      dfz:     self.dfz,
      key:     self.key,
    };
  }

  pub fn restore(&mut self, saved : &SavedMetadata)
  {
    self.sides   = saved.sides;
    self.rights  = saved.rights;
    self.enpass  = saved.enpass;
    self.incheck = saved.incheck;
    self.dfz     = saved.dfz;
    self.key     = saved.key;
  }

  // For use when the squares field may not be set properly
  pub fn at_square(&self, square : usize, hint : Color) -> Piece
  {
    let mask = 1 << square;
    for k in PNBRQK {
      if self.boards[hint+k] & mask != 0 { return hint+k; }
    }
    return Null;
  }

  pub fn new324(x : usize) -> Self
  {
    assert!(x < 324,  "starting position must be less than 324");
    //
    // See http://talkchess.com/forum3/viewtopic.php?f=2&t=80482
    //
    // In Chess324, "the kings and rooks are placed on their normal positions.
    //   All the other pieces for White and Black are placed randomly, with no
    //   symmetry requirement, with the only restriction being that for each
    //   side the bishops must be on opposite colored squares."
    //
    // There are then 3 positions for the light bishop
    //              × 2 positions for the dark bishop
    //              × 3 positions for the queen
    // and then the knights take the two remaining positions, for a total of
    // 18 configurations per side, and 18 × 18 = 324 positions total.
    //
    // "Light bishop" is a misnomer – it's the light bishop for white, but the
    //   dark bishop for black – but we'll use the terminology for the sake of
    //   convenience.
    //
    let mut boards = [
      0b_0001_0000,
      0,
      0b_1000_0001,
      0,
      0,
      0b_1111_1111 << 8,
      0,
      0,
      0b_0001_0000 << 56,
      0,
      0b_1000_0001 << 56,
      0,
      0,
      0b_1111_1111 << 48,
      0,
      0,
    ];
    let mut squares = [
      WhiteRook, Null     , Null     , Null     , WhiteKing, Null     , Null     , WhiteRook,
      WhitePawn, WhitePawn, WhitePawn, WhitePawn, WhitePawn, WhitePawn, WhitePawn, WhitePawn,
      Null     , Null     , Null     , Null     , Null     , Null     , Null     , Null     ,
      Null     , Null     , Null     , Null     , Null     , Null     , Null     , Null     ,
      Null     , Null     , Null     , Null     , Null     , Null     , Null     , Null     ,
      Null     , Null     , Null     , Null     , Null     , Null     , Null     , Null     ,
      BlackPawn, BlackPawn, BlackPawn, BlackPawn, BlackPawn, BlackPawn, BlackPawn, BlackPawn,
      BlackRook, Null     , Null     , Null     , BlackKing, Null     , Null     , BlackRook,
    ];

    let configurations = [x / 18, x % 18];
    for color in WB {
      let config = configurations[color];

      let lt_idx =  config / 6;
      let dk_idx = (config % 6) / 3;
      let  q_idx = (config % 6) % 3;

      let lt_idx = 1 + lt_idx*2;
      let dk_idx = 2 + dk_idx*4;

      let home = color as usize * 56;

      let lt_sq = home + lt_idx;
      let dk_sq = home + dk_idx;
      boards[color+Bishop] |= 1 << lt_sq;
      boards[color+Bishop] |= 1 << dk_sq;
      squares[lt_sq] = color+Bishop;
      squares[dk_sq] = color+Bishop;

      let mut empty : u32 = !(0b_1001_0001 | (1 << lt_idx) | (1 << dk_idx));
      for x in 0..3 {
        let piece = color + if x == q_idx { Queen } else { Knight };
        let square = home + empty.trailing_zeros() as usize;
        boards[piece] |= 1 << square;
        squares[square] = piece;
        empty &= empty - 1;
      }
    }

    let mut state = Self {
      sides: [WHITE_HOME, BLACK_HOME],
      boards:  boards,
      squares: squares,
      rights:  0b_1111,
      enpass:  -1,
      incheck: false,
      turn:    White,
      dfz:     0,
      ply:     0,
      key:     0,
      s1:      Vec::new(),
    };
    state.key = state.zobrist();
    return state;
  }
}
