pub const RFP_OFFSET        : i16 =  8;       // centipawn
pub const RFP_SCALE         : i16 = 48;       // centipawn/ply

pub const NULL_BASE         : u32 = 10240;    // ply×4096
pub const NULL_SCALE        : u32 =  1216;    // 4096

pub const SSE_BASE          : i32 = -512;     // ply×4096
pub const SSE_SCALE         : i32 = 2304;     // 4096

pub const SSE_MARGIN_OFFSET : i32 = 1024;     // centipawn×256
pub const SSE_MARGIN_SCALE  : i32 =  480;     // centipawn×256/ply

pub const FP_OFFSET         : i16 = 27;       // centipawn
pub const FP_SCALE          : i16 = 11;       // centipawn/ply
pub const FP_THRESH         : i8  = 40;       // historyScore

pub const LMR_BASE          : f64 = 0.600;    // ply
pub const LMR_SCALE         : f64 = 0.290;    // ply
pub const HST_SCALE         : f64 = 0.016129; // 1÷historyScore
pub const FW_RATIO          : f64 = 0.500;    // (unitless)

pub const DELTA_SCALE       : i32 = 6144;     // 4096
pub const DELTA_MARGIN      : i32 =  330;     // centipawn
