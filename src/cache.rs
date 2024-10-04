use crate::global::{GLOB, num_blocks, set_num_blocks};
use crate::misc::NodeKind;
use crate::score::PovScore;

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

// This struct is layed out like so:
//
//   key          2    2
//   generation   2    4
//   hint_move    2    6
//   hint_score   2    8
//   depth        1    9
//   kind         1   10

// TODO
//   key          2    2
//   hint_move    2    4
//   hint_score   2    6
//   static_score 2    8
//   depth        1    9
//   kind+gen     1   10

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(C)]
pub struct TableEntry {
  pub key        : u16, // lowest 16 bits of the key
  pub generation : u16,
  pub hint_move  : u16,
  pub hint_score : PovScore,
  pub depth      : u8,
  pub kind       : NodeKind,
}

impl TableEntry {
  pub const NULL : TableEntry = TableEntry {
    key:        0,
    generation: 0,
    hint_move:  0,
    hint_score: PovScore::ZERO,
    depth:      0,
    kind:       NodeKind::Unk,
  };
}

// This struct is layed out like so:
//
//   entry[0]    10   10
//   entry[1]    10   20
//   entry[2]    10   30
//  (padding      2   32)

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(align(32))]
#[repr(C)]
pub struct TableBlock {
  pub entry : [TableEntry; 3]
}

impl TableBlock {
  pub const NULL : TableBlock = TableBlock {
    entry: [TableEntry::NULL; 3]
  };
}

#[repr(align(2097152))]
pub struct HugePage { ary : [u8; 2_097_152] }

pub fn initialize_cache(size : usize)
{
  // The minimum transposition table size is 2 megabytes
  //   and we round down to a multiple of 2 megabytes
  //   (65'536 blocks = 196'608 entries).
  let num_pages = std::cmp::max(1, size / 2_097_152);
  let size = num_pages * 2_097_152;
  set_num_blocks(size / 32);

  unsafe {
    GLOB.cache = Vec::with_capacity(num_pages);

    // NOTE that we can't populate the cache, e.g. with
    //
    //     const NULL_PAGE : HugePage = HugePage { contents: [NULL; 131072] };
    //     for _ in 0..num_pages { CACHE.push(NULL_PAGE); }
    //
    //   because this causes a stack overflow (since Rust creates objects on the
    //   stack and then copies them over). If we really cared about zeroing the
    //   memory, we could write to it using our usual method of writing entries,
    //   but we genuinely don't care, since we already can handle corrupted
    //   entries without any problems.
    //
    // It’s obviously wrong to leave its length zero, so we commit a faux pas.
    //   NOTE THAT THIS CAUSES UNDEFINED BEHAVIOR.

    GLOB.cache.set_len(num_pages);

    // Transparent huge pages are usually enabled by default, so this madvise
    //   call is likely unnecessary, but we have it here just in case.
    #[cfg(target_os="linux")]
    {
      use crate::util::{STDERR, isatty};

      let ret     : i64;
      let madvise : u64 = 28;
      let addr    : u64 = GLOB.cache.as_ptr() as u64;
      let length  : u64 = size as u64;
      let advice  : u64 = 14;

      std::arch::asm!(
        "syscall"
        , inout("rax") madvise => ret
        ,    in("rdi") addr
        ,    in("rsi") length
        ,    in("rdx") advice
        ,   out("r10") _
        ,   out("r8" ) _
        ,   out("r9" ) _
        ,   out("rcx") _
        ,   out("r11") _
      );
      if ret != 0 && isatty(STDERR) {
        eprintln!("note: madvise returned {ret}");
      }
    }

    // For the sake of performance, it's crucial that we fill out the page
    //   table now. We could zero out the entire region with
    //     for x in 0..size { *ptr.add(x) = 0; }
    //   but that's not necessary – triggering the page fault is enough.
    //  (The operating system will zero out the page for us in any case.)
    // There is a potential side effect: if the allocator is reusing a
    //   previous allocation, initializing the table may not reset the
    //   table (that is, old entries may be preserved).
    // This can take a while if page faults are handled one at a time,
    //   so we issue many in parallel if we can.

    let small_pages = size / 4096;
    let n = std::cmp::max(1, crate::util::num_cores() / 2);
    let n = std::cmp::min(n, 16);

    if n > 1 {
      let mut handles = Vec::new();
      for id in 0..n {
        handles.push(
          std::thread::spawn(move || {
            let ptr = GLOB.cache.as_mut_ptr() as *mut u8;
            let start = small_pages *    id    / n;
            let end   = small_pages * (id + 1) / n;
            for x in start..end { std::ptr::write(ptr.add(x * 4096), 0); }
          })
        );
      }
      for h in handles { h.join().unwrap(); }
    }
    else {
      let ptr = GLOB.cache.as_mut_ptr() as *mut u8;
      for x in 0..small_pages { std::ptr::write(ptr.add(x * 4096), 0); }
    }
  }
}

// The reason we use the lower bits of the key in the entry (rather than the
//   upper bits) is that they are more likely to be distinct. In fact, they do
//   not figure into the index whatsoever.
//
// If we imagine a and b (the key and number of blocks, respectively) as two
//   halves, high and low, viz. a = (aᴴ << 32) + aᴸ and b = (bᴴ << 32) + bᴸ,
//   when multiplied we have
//
//   (aᴴ × bᴴ) << 64  +  (aᴴ × bᴸ + aᴸ × bᴴ) << 32  +  (aᴸ + bᴸ)
//
// Note that the sum aᴸ + bᴸ is only 33 bits wide, so after taking the high
//   word (equivalently, right shifting the product by 64) this disappears
//   entirely (except perhaps for propagated carries). In practice, b is
//   about 2²³ to 2³³, so bᴴ is usually zero, and then aᴸ × hᴴ is zero
//   – and so aᴸ does not affect the result.

fn himul(a : u64, b : u64) -> u64
{
  return ((a as u128).wrapping_mul(b as u128) >> 64) as u64;
}

pub fn table_prefetch(key : u64)
{
  unsafe {
    let index = himul(key, num_blocks() as u64) as usize;
    let offset = index * 4;
    let ptr = std::mem::transmute::<_, *const AtomicU64>(GLOB.cache.as_ptr());
    _mm_prefetch(ptr.add(offset) as *const i8, _MM_HINT_T0);
  }
}

pub fn table_lookup(key : u64) -> TableEntry
{
  let block = unsafe {
    let index = himul(key, num_blocks() as u64) as usize;
    let offset = index * 4;
    /*
    let ptr = std::mem::transmute::<_, *mut AtomicU64>(GLOB.cache.as_mut_ptr());
    // TODO if AVX is available perform a single 256-bit load
    let fst = (*ptr.add(offset+0)).load(Relaxed);
    let snd = (*ptr.add(offset+1)).load(Relaxed);
    let thd = (*ptr.add(offset+2)).load(Relaxed);
    let fth = (*ptr.add(offset+3)).load(Relaxed);
    std::mem::transmute::<_, TableBlock>([fst, snd, thd, fth])
    */
    // WARNING extremely undefined
    let ptr = (GLOB.cache.as_ptr() as *const u64).add(offset) as *const [u64; 4];
    std::mem::transmute::<_, TableBlock>(*ptr)
  };
  for x in 0..3 {
    if block.entry[x].key == key as u16 { return block.entry[x]; }
  }
  return TableEntry::NULL;
}

pub fn table_update(key : u64, entry : TableEntry)
{
  let index = himul(key, num_blocks() as u64) as usize;
  let offset = index * 4;
  /*
  let ptr = unsafe {
    std::mem::transmute::<_, *mut AtomicU64>(GLOB.cache.as_mut_ptr())
  };
  let mut block = unsafe {
    let fst = (*ptr.add(offset+0)).load(Relaxed);
    let snd = (*ptr.add(offset+1)).load(Relaxed);
    let thd = (*ptr.add(offset+2)).load(Relaxed);
    let fth = (*ptr.add(offset+3)).load(Relaxed);
    std::mem::transmute::<_, TableBlock>([fst, snd, thd, fth])
  };
  */
  // WARNING extremely undefined
  let ptr = unsafe {
    (GLOB.cache.as_mut_ptr() as *mut u64).add(offset) as *mut [u64; 4]
  };
  let mut block = unsafe {
    std::mem::transmute::<_, TableBlock>(*ptr)
  };

  for x in 0..3 {
    let extant = &block.entry[x];
    if entry.key != extant.key { continue; }
    // The argument can update an extant entry if
    // • its generation is newer,
    // • its depth is higher, or
    // • its depth is equal and its kind is not less
    // than that of the extant entry.
    if entry.generation != extant.generation
      || entry.depth > extant.depth
      || (entry.depth == extant.depth && entry.kind >= extant.kind)
    {
      block.entry[x] = entry;
      unsafe {
        let block = std::mem::transmute::<_, [u64; 4]>(block);
        /*
        (*ptr.add(offset+0)).store(block[0], Relaxed);
        (*ptr.add(offset+1)).store(block[1], Relaxed);
        (*ptr.add(offset+2)).store(block[2], Relaxed);
        (*ptr.add(offset+3)).store(block[3], Relaxed);
        */
        // WARNING extremely undefined
        std::ptr::write(ptr, block);
      }
    }
    return;
  }

  let mut pessimum = 255;
  let mut sel = usize::MAX;
  for x in 0..3 {
    let extant = &block.entry[x];
    // The argument can overwrite an extant entry if
    // • its generation is newer,
    // • its depth is higher, or
    // • its depth is equal and its kind is not less
    // than that of the extant entry.
    if !(
      entry.generation != extant.generation
      || entry.depth > extant.depth
      || (entry.depth == extant.depth && entry.kind >= extant.kind)
    ) { continue; }

    // TODO this breaks determinism when persist is set
    let age = std::cmp::min(8,
      entry.generation.wrapping_sub(extant.generation)
    ) as u8;
    // Since the maximum depth is roughly 192, the maximum
    //   utility is about 8×2 + 192 + 4 = 212, giving us 43
    //   points of headroom for check extensions.
    let utility = (8 - age) * 2 + extant.depth + extant.kind as u8;
    if utility < pessimum {
      pessimum = utility;
      sel = x;
    }
  }

  if sel == usize::MAX { return; }

  block.entry[sel] = entry;
  unsafe {
    let block = std::mem::transmute::<_, [u64; 4]>(block);
    /*
    (*ptr.add(offset+0)).store(block[0], Relaxed);
    (*ptr.add(offset+1)).store(block[1], Relaxed);
    (*ptr.add(offset+2)).store(block[2], Relaxed);
    (*ptr.add(offset+3)).store(block[3], Relaxed);
    */
    // WARNING extremely undefined
    std::ptr::write(ptr, block);
  }
}

pub fn table_utilization(generation : u16) -> u16
{
  // Examine 2000×3 = 6000 entries and then
  //   divide by 6 to get entries per mille.
  // We pick 2000 rather than 1000 so that we
  //   can add 3 to round rather than truncate.
  let mut matches = 0;
  for index in 0..2000 {
    let offset = index * 4;
    let ptr = unsafe {
      std::mem::transmute::<_, *mut AtomicU64>(GLOB.cache.as_mut_ptr())
    };
    let block = unsafe {
      let fst = (*ptr.add(offset+0)).load(Relaxed);
      let snd = (*ptr.add(offset+1)).load(Relaxed);
      let thd = (*ptr.add(offset+2)).load(Relaxed);
      let fth = (*ptr.add(offset+3)).load(Relaxed);
      std::mem::transmute::<_, TableBlock>([fst, snd, thd, fth])
    };
    for x in 0..3 {
      if block.entry[x].generation == generation { matches += 1; }
    }
  }
  return (matches + 3) / 6;
}
