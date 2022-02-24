use crate::algebraic::*;
use crate::misc::*;
use crate::movetype::*;
use crate::piece::*;

type KillerTable      =    [(Move, Move); 128]; // [height]
type HistoryTable     = [[(i16, i16); 64]; 16]; // [piece][to-square] -> (score, score)
type CountermoveTable =         [[Op; 64]; 16]; // [piece][to-square] -> op

pub struct Context {                     // Read          Written         Index
  pub current_line   : [  Op; 128],      //   main          main            height
  pub gainful        : [bool; 128],      //   resolving     resolving       length
  pub killer_table   : KillerTable,      //   main          main            height
  pub history_table  : HistoryTable,     //   main          main            ...
  pub counter_table  : CountermoveTable, //   main          main            ...
  pub state_history  : Vec<(u64, bool)>, //   main          main            ...
  pub null           : [bool; 128],      //   main          main            ...
  pub exclude        : [Move; 128],      //   main          main            ...
  pub nominal        : u8,
}

pub struct Statistics {
  pub m_nodes_at_height : [usize; 128],
  pub r_nodes_at_height : [usize; 128],
  pub r_nodes_at_length : [usize; 128],

  pub nodes_w_num_moves : [usize; 128],
  pub last_at_movenum_w_hit  : [usize; 128],
  pub  cut_at_movenum_w_hit  : [usize; 128],
  pub   pv_at_movenum_w_hit  : [usize; 128],
  pub last_at_movenum_w_miss : [usize; 128],
  pub  cut_at_movenum_w_miss : [usize; 128],
  pub   pv_at_movenum_w_miss : [usize; 128],

  pub full_etc_zwrd           : usize,
  pub full_etc_zwrd_zwfd      : usize,
  pub full_etc_zwfd           : usize,
  pub full_etc_zwrd_zwfd_fwfd : usize,
  pub full_etc_zwfd_fwfd      : usize,
  pub full_fst_fwfd           : usize,
  pub zero_etc_zwrd           : usize,
  pub zero_etc_zwrd_fwfd      : usize,
  pub zero_etc_fwfd           : usize,
  pub zero_fst_fwfd           : usize,
}

impl Context {
  pub const fn new() -> Self
  {
    const NULL_PAIR : (Move, Move) = (NULL_MOVE, NULL_MOVE);
    return Self {
      current_line:   [NOP; 128],
      gainful:        [false; 128],
      killer_table:   [NULL_PAIR; 128],
      history_table:  [[(0, 0); 64]; 16],
      counter_table:  [[NOP; 64]; 16],
      state_history:  Vec::new(),
      null:           [false; 128],
      exclude:        [NULL_MOVE; 128],
      nominal:        0,
    };
  }

  pub fn reset(&mut self)
  {
    const NULL_PAIR : (Move, Move) = (NULL_MOVE, NULL_MOVE);
    self.current_line  = [NOP; 128];
    self.gainful       = [false; 128];
    self.killer_table  = [NULL_PAIR; 128];
    self.history_table = [[(0, 0); 64]; 16];
    self.counter_table = [[NOP; 64]; 16];
    self.state_history.clear();
    self.null          = [false; 128];
    self.exclude       = [NULL_MOVE; 128];
    self.nominal       =  0;
  }

  pub fn lookup_history(&self, piece : usize, dst : usize, height : u8) -> i8
  {
    let entry  = &self.history_table[piece][dst];
    let height = std::cmp::min(height as i32, 32);
    let score  = ((entry.0 as i32)*(32 - height) + (entry.1 as i32)*height) / (32*256);
    return score as i8;
  }

  pub fn update_history(&mut self, piece : usize, dst : usize, height : u8, cutoff : bool)
  {
    let entry = &self.history_table[piece][dst];
    let height = std::cmp::min(height as i16, 32);
    let deep;
    let shallow;
    if cutoff {
      deep    = std::cmp::min(32000, entry.0 + (32 - height));
      shallow = std::cmp::min(32000, entry.1 + height       );
    }
    else {
      deep    = std::cmp::max(-32000, entry.0 - (32 - height));
      shallow = std::cmp::max(-32000, entry.1 - height       );
    }
    self.history_table[piece][dst] = (deep, shallow);
  }

  pub fn print_history(&self)
  {
    for color in 0..2 {
      for kind in 0..6 {
        let piece = color*8 + kind;
        eprintln!("{}", FULLNAME[piece]);
        for dst in 0..64 {
          let (deep, shallow) = self.history_table[piece][dst];
          let deep_color =
            if deep > 0 { "\x1B[92m" } else if deep < 0 { "\x1B[91m" } else { "" };
          let shallow_color =
            if shallow > 0 { "\x1B[92m" } else if shallow < 0 { "\x1B[91m" } else { "" };
          eprintln!(
            "  {} {}{:+6}\x1B[39m {}{:+6}\x1B[39m",
            (dst as u8).algebraic(), deep_color, deep, shallow_color, shallow
          );
        }
      }
    }
  }
}

// Search traces are of the form 0mss_rrfz where
//
//    z = 0   not in zero-window search
//        1   within zero-window search
//
//    f = 0   not first move
//        1   first move
//
//   rr = 00  did not run ZWRD search
//        01  ran ZWRD search and failed
//        11  ran ZWRD search and succeeded
//
//   ss = 00  did not run ZWFD search
//        01  ran ZWFD search and failed
//        11  ran ZWFD search and succeeded
//
//    m = 0   did not run FWFD search
//        1   ran FWFD search
//
pub const IN_ZW : u8 = 0b_0000_0001;
pub const FST   : u8 = 0b_0000_0010;
pub const ZR    : u8 = 0b_0000_0100;
pub const ZR_OK : u8 = 0b_0000_1000;
pub const ZF    : u8 = 0b_0001_0000;
pub const ZF_OK : u8 = 0b_0010_0000;
pub const FF    : u8 = 0b_0100_0000;

pub const FULL_ETC_ZWRD           : u8 =              ZR | ZR_OK         ;
pub const FULL_ETC_ZWRD_ZWFD      : u8 =              ZR | ZF    | ZF_OK ;
pub const FULL_ETC_ZWFD           : u8 =                   ZF    | ZF_OK ;
pub const FULL_ETC_ZWRD_ZWFD_FWFD : u8 =              ZR | ZF    | FF    ;
pub const FULL_ETC_ZWFD_FWFD      : u8 =                   ZF    | FF    ;
pub const FULL_FST_FWFD           : u8 =        FST              | FF    ;
pub const ZERO_ETC_ZWRD           : u8 = IN_ZW      | ZR | ZR_OK         ;
pub const ZERO_ETC_ZWRD_FWFD      : u8 = IN_ZW      | ZR         | FF    ;
pub const ZERO_ETC_FWFD           : u8 = IN_ZW                   | FF    ;
pub const ZERO_FST_FWFD           : u8 = IN_ZW |FST              | FF    ;

impl Statistics {
  pub const fn new() -> Self
  {
    return Self {
      m_nodes_at_height:      [0; 128],
      r_nodes_at_height:      [0; 128],
      r_nodes_at_length:      [0; 128],
      nodes_w_num_moves:      [0; 128],
      last_at_movenum_w_hit:  [0; 128],
       cut_at_movenum_w_hit:  [0; 128],
        pv_at_movenum_w_hit:  [0; 128],
      last_at_movenum_w_miss: [0; 128],
       cut_at_movenum_w_miss: [0; 128],
        pv_at_movenum_w_miss: [0; 128],
      full_etc_zwrd:           0,
      full_etc_zwrd_zwfd:      0,
      full_etc_zwfd:           0,
      full_etc_zwrd_zwfd_fwfd: 0,
      full_etc_zwfd_fwfd:      0,
      full_fst_fwfd:           0,
      zero_etc_zwrd:           0,
      zero_etc_zwrd_fwfd:      0,
      zero_etc_fwfd:           0,
      zero_fst_fwfd:           0,
    };
  }

  pub fn reset(&mut self)
  {
    self.m_nodes_at_height       = [0; 128];
    self.r_nodes_at_height       = [0; 128];
    self.r_nodes_at_length       = [0; 128];
    self.nodes_w_num_moves       = [0; 128];
    self.last_at_movenum_w_hit   = [0; 128];
     self.cut_at_movenum_w_hit   = [0; 128];
      self.pv_at_movenum_w_hit   = [0; 128];
    self.last_at_movenum_w_miss  = [0; 128];
     self.cut_at_movenum_w_miss  = [0; 128];
      self.pv_at_movenum_w_miss  = [0; 128];
    self.full_etc_zwrd           =  0;
    self.full_etc_zwrd_zwfd      =  0;
    self.full_etc_zwfd           =  0;
    self.full_etc_zwrd_zwfd_fwfd =  0;
    self.full_etc_zwfd_fwfd      =  0;
    self.full_fst_fwfd           =  0;
    self.zero_etc_zwrd           =  0;
    self.zero_etc_zwrd_fwfd      =  0;
    self.zero_etc_fwfd           =  0;
    self.zero_fst_fwfd           =  0;
  }

  pub fn add(&mut self, other : &Self)
  {
    for x in 0..128 { self.m_nodes_at_height[x] += other.m_nodes_at_height[x]; }
    for x in 0..128 { self.r_nodes_at_height[x] += other.r_nodes_at_height[x]; }
    for x in 0..128 { self.r_nodes_at_length[x] += other.r_nodes_at_length[x]; }
    for x in 0..128 { self.nodes_w_num_moves[x] += other.nodes_w_num_moves[x]; }
    for x in 0..128 { self.last_at_movenum_w_hit[x]  += other.last_at_movenum_w_hit[x];  }
    for x in 0..128 {  self.cut_at_movenum_w_hit[x]  +=  other.cut_at_movenum_w_hit[x];  }
    for x in 0..128 {   self.pv_at_movenum_w_hit[x]  +=   other.pv_at_movenum_w_hit[x];  }
    for x in 0..128 { self.last_at_movenum_w_miss[x] += other.last_at_movenum_w_miss[x]; }
    for x in 0..128 {  self.cut_at_movenum_w_miss[x] +=  other.cut_at_movenum_w_miss[x]; }
    for x in 0..128 {   self.pv_at_movenum_w_miss[x] +=   other.pv_at_movenum_w_miss[x]; }
    self.full_etc_zwrd           += other.full_etc_zwrd;
    self.full_etc_zwrd_zwfd      += other.full_etc_zwrd_zwfd;
    self.full_etc_zwfd           += other.full_etc_zwfd;
    self.full_etc_zwrd_zwfd_fwfd += other.full_etc_zwrd_zwfd_fwfd;
    self.full_etc_zwfd_fwfd      += other.full_etc_zwfd_fwfd;
    self.full_fst_fwfd           += other.full_fst_fwfd;
    self.zero_etc_zwrd           += other.zero_etc_zwrd;
    self.zero_etc_zwrd_fwfd      += other.zero_etc_zwrd_fwfd;
    self.zero_etc_fwfd           += other.zero_etc_fwfd;
    self.zero_fst_fwfd           += other.zero_fst_fwfd;
  }

  pub fn print_stats(&self)
  {
    let total_nodes = self.nodes_w_num_moves.iter().sum::<usize>() as f64;
    let total_last  = self.last_at_movenum_w_hit .iter().sum::<usize>() as f64
                    + self.last_at_movenum_w_miss.iter().sum::<usize>() as f64;
    let total_cut   = self .cut_at_movenum_w_hit .iter().sum::<usize>() as f64
                    + self .cut_at_movenum_w_miss.iter().sum::<usize>() as f64;
    let total_pv    = self  .pv_at_movenum_w_hit .iter().sum::<usize>() as f64
                    + self  .pv_at_movenum_w_miss.iter().sum::<usize>() as f64;
    let mut cum_nodes = 0;
    let mut cum_last_both = 0;
    let mut cum_last_hit  = 0;
    let mut cum_last_miss = 0;
    let mut cum_cut_both  = 0;
    let mut cum_cut_hit   = 0;
    let mut cum_cut_miss  = 0;
    let mut cum_pv_both   = 0;
    let mut cum_pv_hit    = 0;
    let mut cum_pv_miss   = 0;
    eprint!("\x1B[A");
    eprintln!("──────────────── ────────────── Last ────────────── ─────────────── Cut ────────────── ─────────────── PV ──────────────");
    eprintln!("        Nodes        Total       Hit       Miss         Total       Hit       Miss         Total       Hit       Miss");
    eprintln!("──────────────── ────────────────────────────────── ────────────────────────────────── ─────────────────────────────────");
    for x in 0..35 {
      let nodes = self.nodes_w_num_moves[x];
      cum_nodes += nodes;
      let     pct_nodes =     nodes as f64 * 100.0 / total_nodes;
      let pct_cum_nodes = cum_nodes as f64 * 100.0 / total_nodes;

      let last_hit  = self.last_at_movenum_w_hit[x];
      let last_miss = self.last_at_movenum_w_miss[x];
      let last_both = last_hit + last_miss;
      cum_last_both += last_both;
      cum_last_hit  += last_hit;
      cum_last_miss += last_miss;
      let     pct_last_both =     last_both as f64 * 100.0 / total_last;
      let     pct_last_hit  =     last_hit  as f64 * 100.0 / total_last;
      let     pct_last_miss =     last_miss as f64 * 100.0 / total_last;
      let pct_cum_last_both = cum_last_both as f64 * 100.0 / total_last;
      let pct_cum_last_hit  = cum_last_hit  as f64 * 100.0 / total_last;
      let pct_cum_last_miss = cum_last_miss as f64 * 100.0 / total_last;

      let cut_hit  = self.cut_at_movenum_w_hit[x];
      let cut_miss = self.cut_at_movenum_w_miss[x];
      let cut_both = cut_hit + cut_miss;
      cum_cut_both += cut_both;
      cum_cut_hit  += cut_hit;
      cum_cut_miss += cut_miss;
      let     pct_cut_both =     cut_both as f64 * 100.0 / total_cut;
      let     pct_cut_hit  =     cut_hit  as f64 * 100.0 / total_cut;
      let     pct_cut_miss =     cut_miss as f64 * 100.0 / total_cut;
      let pct_cum_cut_both = cum_cut_both as f64 * 100.0 / total_cut;
      let pct_cum_cut_hit  = cum_cut_hit  as f64 * 100.0 / total_cut;
      let pct_cum_cut_miss = cum_cut_miss as f64 * 100.0 / total_cut;

      let pv_hit  = self.pv_at_movenum_w_hit[x];
      let pv_miss = self.pv_at_movenum_w_miss[x];
      let pv_both = pv_hit + pv_miss;
      cum_pv_both += pv_both;
      cum_pv_hit  += pv_hit;
      cum_pv_miss += pv_miss;
      let     pct_pv_both =     pv_both as f64 * 100.0 / total_pv;
      let     pct_pv_hit  =     pv_hit  as f64 * 100.0 / total_pv;
      let     pct_pv_miss =     pv_miss as f64 * 100.0 / total_pv;
      let pct_cum_pv_both = cum_pv_both as f64 * 100.0 / total_pv;
      let pct_cum_pv_hit  = cum_pv_hit  as f64 * 100.0 / total_pv;
      let pct_cum_pv_miss = cum_pv_miss as f64 * 100.0 / total_pv;

      eprintln!(
        "{:2}   {:5.2} \x1B[2m{:4.1}\x1B[22m   \
         \x1B[1m{:5.2}\x1B[22m \x1B[2m{:4.1}\x1B[22m {:5.2} \x1B[2m{:4.1}\x1B[22m {:5.2} \x1B[2m{:4.1}\x1B[22m   \
         \x1B[1m{:5.2}\x1B[22m \x1B[2m{:4.1}\x1B[22m {:5.2} \x1B[2m{:4.1}\x1B[22m {:5.2} \x1B[2m{:4.1}\x1B[22m   \
         \x1B[1m{:5.2}\x1B[22m \x1B[2m{:4.1}\x1B[22m {:5.2} \x1B[2m{:4.1}\x1B[22m {:5.2} \x1B[2m{:4.1}\x1B[22m",
        x,
        pct_nodes,     pct_cum_nodes,

        pct_last_both, pct_cum_last_both,
        pct_last_hit,  pct_cum_last_hit,
        pct_last_miss, pct_cum_last_miss,

        pct_cut_both,  pct_cum_cut_both,
        pct_cut_hit,   pct_cum_cut_hit,
        pct_cut_miss,  pct_cum_cut_miss,

        pct_pv_both,   pct_cum_pv_both,
        pct_pv_hit,    pct_cum_pv_hit,
        pct_pv_miss,   pct_cum_pv_miss,
      );
    }
    eprintln!("──────────────── ────────────────────────────────── ────────────────────────────────── ─────────────────────────────────");
  }

  pub fn print_trace(&self)
  {
    let total = self.full_etc_zwrd + self.full_etc_zwrd_zwfd + self.full_etc_zwfd
              + self.full_fst_fwfd + self.full_etc_zwfd_fwfd + self.zero_fst_fwfd
              + self.zero_etc_zwrd + self.zero_etc_zwrd_fwfd + self.zero_etc_fwfd
              + self.full_etc_zwrd_zwfd_fwfd;
    let total = total as f64;

    let full_window = self.full_etc_zwrd + self.full_etc_zwrd_zwfd + self.full_etc_zwfd
                    + self.full_etc_zwrd_zwfd_fwfd + self.full_etc_zwfd_fwfd + self.full_fst_fwfd;

    let zero_window = self.zero_etc_zwrd + self.zero_etc_zwrd_fwfd
                    + self.zero_etc_fwfd + self.zero_fst_fwfd;

    eprint!("\x1B[A");
    eprintln!("\
Within Full-window \x1B[2m{:6.3}\x1B[22m
 │                                ╭──────╮
 ╰──┬── not first ──┬── reduced ──┤ ZwRd ├──┬─── \x1B[2mZwRd\x1B[22m {:6.3}
    │               │             ╰──────╯  │
    │               │                       │
    │               │                       │  ╭──────╮     ┌ \x1B[2mZwRd→ZwFd\x1B[22m {:6.3}
    │               ╰── unreduced ──────────┴──┤ ZwFd ├──┬──┴      \x1B[2mZwFd\x1B[22m {:6.3}
    │                                          ╰──────╯  │
    │                                                    │
    │                                                    │  ╭──────╮  ┌ \x1B[2mZwRd→ZwFd→FwFd\x1B[22m {:6.3}
    ╰── first ───────────────────────────────────────────┴──┤ FwFd ├──┤      \x1B[2mZwFd→FwFd\x1B[22m {:6.3}
                                                            ╰──────╯  └           \x1B[2mFwFd\x1B[22m {:6.3}

Within Zero-window \x1B[2m{:6.3}\x1B[22m
 │                                ╭──────╮
 ╰──┬── not first ──┬── reduced ──┤ ZwRd ├──┬─── \x1B[2mZwRd\x1B[22m {:6.3}
    │               │             ╰──────╯  │
    │               │                       │
    │               ╰── unreduced ──────────┴────────────┐
    │                                                    │
    │                                                    │  ╭──────╮  ┌ \x1B[2mZwRd→FwFd\x1B[22m {:6.3}
    ╰── first ───────────────────────────────────────────┴──┤ FwFd ├──┤      \x1B[2mFwFd\x1B[22m {:6.3}
                                                            ╰──────╯  └  \x1B[2mFst FwFd\x1B[22m {:6.3}",

      full_window                  as f64 * 100.0 / total,
      self.full_etc_zwrd           as f64 * 100.0 / total,
      self.full_etc_zwrd_zwfd      as f64 * 100.0 / total,
      self.full_etc_zwfd           as f64 * 100.0 / total,
      self.full_etc_zwrd_zwfd_fwfd as f64 * 100.0 / total,
      self.full_etc_zwfd_fwfd      as f64 * 100.0 / total,
      self.full_fst_fwfd           as f64 * 100.0 / total,
      zero_window                  as f64 * 100.0 / total,
      self.zero_etc_zwrd           as f64 * 100.0 / total,
      self.zero_etc_zwrd_fwfd      as f64 * 100.0 / total,
      self.zero_etc_fwfd           as f64 * 100.0 / total,
      self.zero_fst_fwfd           as f64 * 100.0 / total,
    );
  }
}
