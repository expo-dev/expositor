use std::ffi::CString;

use crate::color::*;
use crate::movegen::*;
use crate::movetype::*;
use crate::piece::*;
use crate::score::*;
use crate::state::*;

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

static mut TB_ENABLED : bool = false;

pub fn initialize_syzygy(path : &str) -> bool {
  let mut status = false;
  if let Ok(cstr) = CString::new(path) {
    status = unsafe { c::tb_init(cstr.as_ptr()) };
  }
  unsafe { TB_ENABLED = status; }
  return status;
}

pub fn syzygy_active()  -> bool { return unsafe { TB_ENABLED };    }

pub fn syzygy_support() -> u32  { return unsafe { c::TB_LARGEST }; }

pub fn probe_syzygy_wdl(state : &State, height : i16) -> Option<i16>
{
  let outcome : u32 = unsafe { c::tb_probe_wdl(
    state.sides[W],
    state.sides[B],
    state.boards[WHITE+KING]   | state.boards[BLACK+KING]   ,
    state.boards[WHITE+QUEEN]  | state.boards[BLACK+QUEEN]  ,
    state.boards[WHITE+ROOK]   | state.boards[BLACK+ROOK]   ,
    state.boards[WHITE+BISHOP] | state.boards[BLACK+BISHOP] ,
    state.boards[WHITE+KNIGHT] | state.boards[BLACK+KNIGHT] ,
    state.boards[WHITE+PAWN]   | state.boards[BLACK+PAWN]   ,
    state.dfz as u32,
    0,
    std::cmp::max(state.enpass, 0) as u32,
    !state.turn as u8
  ) };
  return match outcome {
    TB_LOSS         => Some(TABLEBASE_LOSS + height),
    TB_BLESSED_LOSS => Some(0),
    TB_DRAW         => Some(0),
    TB_CURSED_WIN   => Some(0),
    TB_WIN          => Some(TABLEBASE_MATE - height),
    _               => None
  };
}

pub fn probe_syzygy_line(state : &mut State) -> Option<(i16, Vec<Move>)>
{
  let entry : u32 = unsafe { c::tb_probe_root(
    state.sides[W],
    state.sides[B],
    state.boards[WHITE+KING]   | state.boards[BLACK+KING]   ,
    state.boards[WHITE+QUEEN]  | state.boards[BLACK+QUEEN]  ,
    state.boards[WHITE+ROOK]   | state.boards[BLACK+ROOK]   ,
    state.boards[WHITE+BISHOP] | state.boards[BLACK+BISHOP] ,
    state.boards[WHITE+KNIGHT] | state.boards[BLACK+KNIGHT] ,
    state.boards[WHITE+PAWN]   | state.boards[BLACK+PAWN]   ,
    state.dfz as u32,
    0,
    std::cmp::max(state.enpass, 0) as u32,
    !state.turn as u8
  ) };

  if entry == TB_FAILED { return None; }

  let wdl =  (entry & TB_ENTRY_WDL_MASK) >> TB_ENTRY_WDL_SHIFT;
  let dtz = ((entry & TB_ENTRY_DTZ_MASK) >> TB_ENTRY_DTZ_SHIFT) as i16;

  let score = match wdl {
    TB_LOSS         => TABLEBASE_LOSS + dtz,
    TB_BLESSED_LOSS => 0,
    TB_DRAW         => 0,
    TB_CURSED_WIN   => 0,
    TB_WIN          => TABLEBASE_MATE - dtz,
    _               => panic!("entry wdl {:x}", wdl)
  };

  let src = ((entry & TB_ENTRY_FROM_MASK    ) >> TB_ENTRY_FROM_SHIFT    ) as i8;
  let dst = ((entry & TB_ENTRY_TO_MASK      ) >> TB_ENTRY_TO_SHIFT      ) as i8;
  let pro = ((entry & TB_ENTRY_PROMOTES_MASK) >> TB_ENTRY_PROMOTES_SHIFT) as usize;

  // By nice coincidence, the values used to encode piece kinds in the promotion field
  //   match the values used by Expositor, and 0 is used to encode TB_PROMOTES_NONE.

  let mut early_moves = Vec::with_capacity(16);
  let mut  late_moves = Vec::with_capacity(32);
  state.generate_legal_moves(Selectivity::Everything, &mut early_moves, &mut late_moves);
  early_moves.append(&mut late_moves);

  if early_moves.len() == 0 {
    return Some((if state.incheck { PROVEN_LOSS } else { 0 }, Vec::new()));
  }

  let metadata = state.save();
  for mv in early_moves {
    if mv.src == src && mv.dst == dst && (pro == 0 || mv.promotion.kind() == pro) {
      if score == 0 {
        return Some((0, vec![mv]));
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

  panic!("entry src {} dst {} pro {}", src, dst, pro);
}

mod c {
  extern "C" {
    pub static TB_LARGEST : u32;
    pub fn tb_init(filename : *const i8) -> bool;
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
