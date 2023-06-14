use crate::misc::NodeKind;
use crate::global::{GLOB, index_mask, set_index_mask};

use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

// This struct is layed out like so:
//
//   key          8    8
//   generation   2   10
//   hint_move    2   12
//   hint_score   2   14
//   depth        1   15
//   kind         1   16

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(C)]
pub struct TableEntry {
  pub key        : u64,
  pub generation : u16,
  pub hint_move  : u16,
  pub hint_score : i16,
  pub depth      : u8,
  pub kind       : NodeKind,
}

// NOTE that this is mega jank.
fn quick_nonkey(a : &TableEntry) -> u64
{
  let ptr : *const u16 = &a.generation;
  return unsafe { *(ptr as *const u64) };
}

impl TableEntry {
  pub const NULL : TableEntry = TableEntry {
    key:        0,
    generation: 0,
    hint_move:  0,
    hint_score: 0,
    depth:      0,
    kind:       NodeKind::Unk,
  };
}

pub fn initialize_cache(size : usize)
{
  // The minimum transposition table size is 64 kilobytes (4096 entries)
  let mut size = std::cmp::max(size, 65_536) / 16;
  let mut log2 = 0;
  loop { size >>= 1; if size == 0 { break; } log2 += 1; }
  let size : usize = 1 << log2;
  unsafe {
    // We replace the cache instead of calling clear so
    //   that the allocation size is actually decreased
    GLOB.cache = Vec::with_capacity(size);
    for _ in 0..size { GLOB.cache.push(TableEntry::NULL); }
  }
  set_index_mask(size - 1);
}

pub fn table_prefetch(key : u64)
{
  let index = key as usize & index_mask();
  unsafe { _mm_prefetch(&GLOB.cache[index] as *const TableEntry as *const i8, _MM_HINT_T0); }
}

pub fn table_lookup(key : u64) -> TableEntry
{
  let index = key as usize & index_mask();
  let entry = unsafe { GLOB.cache[index].clone() };
  let deciphered = entry.key ^ quick_nonkey(&entry);
  return if deciphered == key { entry } else { TableEntry::NULL };
}

pub fn table_update(mut update : TableEntry)
{
  let index = update.key as usize & index_mask();
  let preexisting = unsafe { &mut GLOB.cache[index] };
  if update.generation != preexisting.generation
    || update.depth > preexisting.depth
    || (update.depth == preexisting.depth && update.kind as u8 >= preexisting.kind as u8)
  { update.key ^= quick_nonkey(&update); *preexisting = update; }
}
