use std::ffi::CString;

use crate::color::Color::*;
use crate::global::enable_syzygy;
use crate::movegen::Selectivity::Everything;
use crate::movetype::Move;
use crate::piece::Piece::*;
use crate::score::PovScore;
use crate::state::State;

// This file contains the interface with the library Fathom by Ronald de Man,
//   basil, and Jon Dart; see the directory /fathom for license information.
// This file was written using as reference Asymptote by Maximilian Lupke and
//   Mantissa by Jeremy Wright.

const TB_LOSS         : u32 = 0;
const TB_BLESSED_LOSS : u32 = 1;
const TB_DRAW         : u32 = 2;
const TB_CURSED_WIN   : u32 = 3;
const TB_WIN          : u32 = 4;
const TB_FAILED       : u32 = 0xFFFF_FFFF;

const TB_ENTRY_WDL_MASK      : u32 = 0x0000000F;
const TB_ENTRY_TO_MASK       : u32 = 0x000003F0;
const TB_ENTRY_FROM_MASK     : u32 = 0x0000FC00;
const TB_ENTRY_PROMOTES_MASK : u32 = 0x00070000;
const TB_ENTRY_EP_MASK       : u32 = 0x00080000;
const TB_ENTRY_DTZ_MASK      : u32 = 0xFFF00000;

const TB_ENTRY_WDL_SHIFT      : u32 =  0;
const TB_ENTRY_TO_SHIFT       : u32 =  4;
const TB_ENTRY_FROM_SHIFT     : u32 = 10;
const TB_ENTRY_PROMOTES_SHIFT : u32 = 16;
const TB_ENTRY_EP_SHIFT       : u32 = 19;
const TB_ENTRY_DTZ_SHIFT      : u32 = 20;

pub fn disable_syzygy() {
  unsafe { c::tb_free(); }
  enable_syzygy(false);
}

pub fn initialize_syzygy(path : &str) -> bool
{
  let mut status = false;
  if let Ok(cstr) = CString::new(path) {
    status = unsafe { c::tb_init(cstr.as_ptr()) };
  }
  enable_syzygy(status);
  return status;
}

pub fn syzygy_support() -> u32 { return unsafe { c::TB_LARGEST }; }

pub fn probe_syzygy_wdl(state : &State, height : u8) -> Option<PovScore>
{
  let outcome : u32 = unsafe { c::tb_probe_wdl(
    state.sides[White],
    state.sides[Black],
    state.boards[WhiteKing  ] | state.boards[BlackKing  ] ,
    state.boards[WhiteQueen ] | state.boards[BlackQueen ] ,
    state.boards[WhiteRook  ] | state.boards[BlackRook  ] ,
    state.boards[WhiteBishop] | state.boards[BlackBishop] ,
    state.boards[WhiteKnight] | state.boards[BlackKnight] ,
    state.boards[WhitePawn  ] | state.boards[BlackPawn  ] ,
    state.dfz as u32,
    0,
    std::cmp::max(state.enpass, 0) as u32,
    !state.turn as u8
  ) };
  return match outcome {
    TB_LOSS         => Some(PovScore::tablebase_loss(height)),
    TB_BLESSED_LOSS => Some(PovScore::ZERO),
    TB_DRAW         => Some(PovScore::ZERO),
    TB_CURSED_WIN   => Some(PovScore::ZERO),
    TB_WIN          => Some(PovScore::tablebase_win(height)),
    _               => None
  };
}

pub fn probe_syzygy_move(state : &State) -> Option<(PovScore, Move)>
{
  let entry : u32 = unsafe { c::tb_probe_root(
    state.sides[White],
    state.sides[Black],
    state.boards[WhiteKing  ] | state.boards[BlackKing  ] ,
    state.boards[WhiteQueen ] | state.boards[BlackQueen ] ,
    state.boards[WhiteRook  ] | state.boards[BlackRook  ] ,
    state.boards[WhiteBishop] | state.boards[BlackBishop] ,
    state.boards[WhiteKnight] | state.boards[BlackKnight] ,
    state.boards[WhitePawn  ] | state.boards[BlackPawn  ] ,
    state.dfz as u32,
    0,
    std::cmp::max(state.enpass, 0) as u32,
    !state.turn as u8
  ) };

  if entry == TB_FAILED { return None; }

  let wdl = (entry & TB_ENTRY_WDL_MASK) >> TB_ENTRY_WDL_SHIFT;
  let dtz = (entry & TB_ENTRY_DTZ_MASK) >> TB_ENTRY_DTZ_SHIFT;
  let dtz = if dtz > 255 { 255 } else { dtz as u8 };

  let score = match wdl {
    TB_LOSS         => PovScore::tablebase_loss(dtz),
    TB_BLESSED_LOSS => PovScore::ZERO,
    TB_DRAW         => PovScore::ZERO,
    TB_CURSED_WIN   => PovScore::ZERO,
    TB_WIN          => PovScore::tablebase_win(dtz),
    _               => panic!("wdl = {:08x}", wdl)
  };

  let src = ((entry & TB_ENTRY_FROM_MASK    ) >> TB_ENTRY_FROM_SHIFT    ) as i8;
  let dst = ((entry & TB_ENTRY_TO_MASK      ) >> TB_ENTRY_TO_SHIFT      ) as i8;
  let pro = ((entry & TB_ENTRY_PROMOTES_MASK) >> TB_ENTRY_PROMOTES_SHIFT) as u8;

  // By nice coincidence, the values used to encode piece kinds in the promotion field
  //   match the values used by Expositor, and 0 is used to encode TB_PROMOTES_NONE.

  let legal_moves = state.collect_legal_moves(Everything);

  if legal_moves.is_empty() {
    return Some(
      (if state.incheck { PovScore::LOST } else { PovScore::ZERO }, Move::NULL)
    );
  }

  for mv in legal_moves.into_iter() {
    if mv.src != src { continue; }
    if mv.dst != dst { continue; }
    if pro != 0 && mv.promotion.kind() as u8 != pro { continue; }
    return Some((score, mv));
  }

  panic!("src = {}, dst = {}, pro = {}", src, dst, pro);
}

pub fn probe_syzygy_line(state : &mut State) -> Option<(PovScore, Vec<Move>)>
{
  let entry : u32 = unsafe { c::tb_probe_root(
    state.sides[White],
    state.sides[Black],
    state.boards[WhiteKing  ] | state.boards[BlackKing  ] ,
    state.boards[WhiteQueen ] | state.boards[BlackQueen ] ,
    state.boards[WhiteRook  ] | state.boards[BlackRook  ] ,
    state.boards[WhiteBishop] | state.boards[BlackBishop] ,
    state.boards[WhiteKnight] | state.boards[BlackKnight] ,
    state.boards[WhitePawn  ] | state.boards[BlackPawn  ] ,
    state.dfz as u32,
    0,
    std::cmp::max(state.enpass, 0) as u32,
    !state.turn as u8
  ) };

  if entry == TB_FAILED { return None; }

  let wdl = (entry & TB_ENTRY_WDL_MASK) >> TB_ENTRY_WDL_SHIFT;
  let dtz = (entry & TB_ENTRY_DTZ_MASK) >> TB_ENTRY_DTZ_SHIFT;
  let dtz = if dtz > 255 { 255 } else { dtz as u8 };

  let score = match wdl {
    TB_LOSS         => PovScore::tablebase_loss(dtz),
    TB_BLESSED_LOSS => PovScore::ZERO,
    TB_DRAW         => PovScore::ZERO,
    TB_CURSED_WIN   => PovScore::ZERO,
    TB_WIN          => PovScore::tablebase_win(dtz),
    _               => panic!("wdl = {:08x}", wdl)
  };

  let src = ((entry & TB_ENTRY_FROM_MASK    ) >> TB_ENTRY_FROM_SHIFT    ) as i8;
  let dst = ((entry & TB_ENTRY_TO_MASK      ) >> TB_ENTRY_TO_SHIFT      ) as i8;
  let pro = ((entry & TB_ENTRY_PROMOTES_MASK) >> TB_ENTRY_PROMOTES_SHIFT) as u8;

  // By nice coincidence, the values used to encode piece kinds in the promotion field
  //   match the values used by Expositor, and 0 is used to encode TB_PROMOTES_NONE.

  let legal_moves = state.collect_legal_moves(Everything);

  if legal_moves.is_empty() {
    return Some((if state.incheck { PovScore::LOST } else { PovScore::ZERO }, Vec::new()));
  }

  let metadata = state.save();
  for mv in legal_moves.into_iter() {
    if mv.src == src && mv.dst == dst && (pro == 0 || mv.promotion.kind() as u8 == pro) {
      if score.is_zero() {
        return Some((PovScore::ZERO, vec![mv]));
      }
      state.apply(&mv);
      let line = probe_syzygy_line(state);
      state.undo(&mv);
      state.restore(&metadata);
      if let Some((_, mut pv)) = line {
        pv.insert(0, mv);
        return Some((score, pv));
      }
      else {
        return Some((score, vec![mv]));
      }
    }
  }

  panic!("src = {}, dst = {}, pro = {}", src, dst, pro);
}

mod c {
  unsafe extern "C" {
    pub static TB_LARGEST : u32;
    pub fn tb_init(path : *const i8) -> bool;
    pub fn tb_free();
    pub fn tb_probe_wdl(
      white   : u64,
      black   : u64,
      kings   : u64,
      queens  : u64,
      rooks   : u64,
      bishops : u64,
      knights : u64,
      pawns   : u64,
      dfz     : u32,
      rights  : u32,
      enpass  : u32,
      turn    : u8
    ) -> u32;
    pub fn tb_probe_root(
      white   : u64,
      black   : u64,
      kings   : u64,
      queens  : u64,
      rooks   : u64,
      bishops : u64,
      knights : u64,
      pawns   : u64,
      dfz     : u32,
      rights  : u32,
      enpass  : u32,
      turn    : u8
    ) -> u32;
  }
}
