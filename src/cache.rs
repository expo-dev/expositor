use crate::misc::*;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// This struct is layed out like so:
//
//   key          8    8
//   generation   2   10
//   hint_move    2   12
//   hint_score   2   14
//   depth        1   15
//   kind         1   16

#[derive(Clone, PartialEq, Eq)]
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

pub const NULL_ENTRY : TableEntry = TableEntry {
  key:        0,
  generation: 0,
  hint_move:  0,
  hint_score: 0,
  depth:      0,
  kind:       NodeKind::Unk,
};

static mut CACHE : Vec<TableEntry> = Vec::new();
static mut INDEX_MASK : usize = 0;

pub fn initialize_cache(size : usize)
{
  // The minimum transposition table size is 64 kilobytes (4096 entries)
  let mut size = std::cmp::max(size, 65536) / 16;
  let mut log2 = 0;
  loop { size >>= 1; if size == 0 { break; } log2 += 1; }
  let size = 1usize << log2;
  unsafe {
    // We replace the cache instead of calling clear so
    //   that the allocation size is actually decreased
    CACHE = Vec::with_capacity(size);
    for _ in 0..size { CACHE.push(NULL_ENTRY); }
    INDEX_MASK = size - 1;
  }
}

#[inline]
pub fn table_lookup(key : u64) -> TableEntry
{
  unsafe {
    let index = key as usize & INDEX_MASK;
    let entry = CACHE[index].clone();
    let deciphered = entry.key ^ quick_nonkey(&entry);
    return if deciphered == key { entry } else { NULL_ENTRY };
  }
}

#[inline]
pub fn table_update(mut update : TableEntry)
{
  unsafe {
    let index = update.key as usize & INDEX_MASK;
    let preexisting = &mut CACHE[index];
    if update.generation != preexisting.generation
      || update.depth > preexisting.depth
      || (update.depth == preexisting.depth && update.kind as u8 >= preexisting.kind as u8)
    { update.key ^= quick_nonkey(&update); *preexisting = update; }
  }
}
