pub const RFP_OFFSET : i16 = 20;      // centipawn
pub const RFP_SCALE  : i16 = 72;      // centipawn/ply

pub const NULL_BASE  : u32 = 10880;   // ply×4096
pub const NULL_SCALE : u32 =  1024;   // 4096

pub const SSE_BASE   : i32 = 1920;    // ply×4096
pub const SSE_SCALE  : i32 = 2368;    // 4096

pub const SSE_MARGIN_OFFSET : i32 = 2144; // centipawn×256
pub const SSE_MARGIN_SCALE  : i32 =  512; // centipawn×256/ply

pub const FP_OFFSET : i16 = 32;       // centipawn
pub const FP_SCALE  : i16 = 12;       // centipawn/ply
pub const FP_THRESH : i8  = 40;       // historyScore

pub const LMR_BASE  : f64 = 0.700;    // ply
pub const LMR_SCALE : f64 = 0.300;    // ply
pub const HST_SCALE : f64 = 0.015625; // 1÷historyScore
pub const FW_RATIO  : f64 = 0.500;    // (unitless)

pub const DELTA_SCALE  : i32 = 6144;  // 4096
pub const DELTA_MARGIN : i32 =  320;  // centipawn
