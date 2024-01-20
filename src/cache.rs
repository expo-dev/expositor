use crate::global::{GLOB, num_entries, set_num_entries};
use crate::misc::NodeKind;

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
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

#[repr(align(2097152))]
pub struct HugePage { ary : [u8; 2_097_152] }

pub fn initialize_cache(size : usize)
{
  // The minimum transposition table size is 2 megabytes (131072 entries)
  //   and we round down to a multiple of 2 megabytes.
  let num_pages = std::cmp::max(1, size / 2_097_152);
  let size = num_pages * 2_097_152;
  set_num_entries(size / 16);

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
    //   but we genuinely don't care, since (using the xor trick) we already can
    //   handle corrupted entries without any problems.
    //
    // It’s obviously wrong to leave its length zero, so we commit a faux pas.
    //   NOTE that this causes undefined behavior!

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

    // For the sake of performance, it's crucial
    //   that we fill out the page table now.
    // We could zero out the entire region with
    //     for x in 0..size { *ptr.add(x) = 0; }
    //   but that's not necessary – triggering the page fault is enough.
    //   (The operating system will zero out the page for us in any case.)
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
            let ptr = std::mem::transmute::<_, *mut u8>(GLOB.cache.as_mut_ptr());
            let start = small_pages *    id    / n;
            let end   = small_pages * (id + 1) / n;
            for x in start..end { *ptr.add(x * 4096) = 0; }
          })
        );
      }
      for h in handles { h.join().unwrap(); }
    }
    else {
      let ptr = std::mem::transmute::<_, *mut u8>(GLOB.cache.as_mut_ptr());
      for x in 0..small_pages { *ptr.add(x * 4096) = 0; }
    }
  }
}

fn mul(a : u64, b : u64) -> u64
{
  return ((a as u128).wrapping_mul(b as u128) >> 64) as u64;
}

pub fn table_prefetch(key : u64)
{
  unsafe {
    let index = mul(key, num_entries() as u64) as usize;
    let offset = index * 2;
    let ptr = std::mem::transmute::<_, *mut AtomicU64>(GLOB.cache.as_mut_ptr());
    _mm_prefetch(ptr.add(offset) as *const i8, _MM_HINT_T0);
  }
}

pub fn table_lookup(key : u64) -> TableEntry
{
  unsafe {
    let index = mul(key, num_entries() as u64) as usize;
    let offset = index * 2;
    let ptr = std::mem::transmute::<_, *mut AtomicU64>(GLOB.cache.as_mut_ptr());
    let fst = (*ptr.add(offset+0)).load(Relaxed);
    let snd = (*ptr.add(offset+1)).load(Relaxed);
    let fst = fst ^ snd;
    return if fst == key {
      std::mem::transmute::<_, TableEntry>([fst, snd])
    }
    else {
      TableEntry::NULL
    };
  }
}

pub fn table_update(entry : &TableEntry)
{
  unsafe {
    let index = mul(entry.key, num_entries() as u64) as usize;
    let offset = index * 2;
    let ptr = std::mem::transmute::<_, *mut AtomicU64>(GLOB.cache.as_mut_ptr());
    let fst = (*ptr.add(offset+0)).load(Relaxed);
    let snd = (*ptr.add(offset+1)).load(Relaxed);
    // we don't bother to decipher the key
    let extant = std::mem::transmute::<_, TableEntry>([fst, snd]);
    if entry.generation != extant.generation
      || entry.depth > extant.depth
      || (entry.depth == extant.depth && entry.kind as u8 >= extant.kind as u8)
    {
      let entry = std::mem::transmute::<_, &[u64; 2]>(entry);
      (*ptr.add(offset+0)).store(entry[0] ^ entry[1], Relaxed);
      (*ptr.add(offset+1)).store(entry[1],            Relaxed);
    }
  }
}
