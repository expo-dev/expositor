#[derive(Copy, Clone)]
#[repr(align(64))]
pub struct SearchConstants {
  pub rfp_offset        : i16,  // centipawn
  pub rfp_scale         : i16,  // centipawn/ply

  pub null_base         : u32,  // ply×4096
  pub null_scale        : u32,  // 4096

  pub sse_base          : i32,  // ply×4096
  pub sse_scale         : i32,  // 4096

  pub sse_margin_offset : i32,  // centipawn×256
  pub sse_margin_scale  : i32,  // centipawn×256/ply

  pub fp_offset         : i16,  // centipawn
  pub fp_scale          : i16,  // centipawn/ply
  pub fp_thresh         :  i8,  // historyScore

  pub lmr_base          : f64,  // ply
  pub lmr_scale         : f64,  // ply
  pub hst_scale         : f64,  // 1÷historyScore
  pub fw_ratio          : f64,  // (unitless)

  pub delta_scale       : i32,  // 4096
  pub delta_margin      : i32,  // centipawn
}

pub const DEFAULT_CONSTANTS : SearchConstants =
  SearchConstants {
    rfp_offset:             8,        // centipawn
    rfp_scale:             48,        // centipawn/ply
    null_base:          10240,        // ply×4096
    null_scale:          1216,        // 4096
    sse_base:            -512,        // ply×4096
    sse_scale:           2304,        // 4096
    sse_margin_offset:   1024,        // centipawn×256
    sse_margin_scale:     480,        // centipawn×256/ply
    fp_offset:             27,        // centipawn
    fp_scale:              11,        // centipawn/ply
    fp_thresh:             40,        // historyScore
    lmr_base:               0.600,    // ply
    lmr_scale:              0.290,    // ply
    hst_scale:              0.016129, // 1÷historyScore
    fw_ratio:               0.500,    // (unitless)
    delta_scale:         6144,        // 4096
    delta_margin:         330,        // centipawn
  };
