// Absolute scores in the range 0.00 to 79.99 pawn are non-mate evaluations, absolute scores
//   in the range 80.00 to 89.99 indicate mates that are unproven but inevitable with reasonable
//   play, absolute scores in the range 90.00 to 99.99 indicate inevitable tablebase entry
//   (leading to mate), an absolute score of 100.00 indicates the position is within the
//   tablebase (and leads to mate), and absolute scores in the range 100.01 to 249.99 indicate
//   proven mates. A score of -250.00 means the game is over.
//
// The range for inevitable mates is somewhat small (10 pawn as opposed to, say, 100 pawn)
//   because we want e.g. to use the same widths for aspiration windows and have them be
//   sensible.

pub const PROVEN_MATE : i16 =  250_00;
pub const PROVEN_LOSS : i16 = -250_00;

pub const MINIMUM_PROVEN_MATE : i16 =  100_01;
pub const MINIMUM_PROVEN_LOSS : i16 = -100_01;

pub const TABLEBASE_MATE : i16 =  100_00;
pub const TABLEBASE_LOSS : i16 = -100_00;

pub const MINIMUM_TB_MATE : i16 =  90_00;
pub const MINIMUM_TB_LOSS : i16 = -90_00;

pub const INEVITABLE_MATE : i16 =  80_00;
pub const INEVITABLE_LOSS : i16 = -80_00;

pub const LIKELY_MATE : i16 =  10_00;
pub const LIKELY_LOSS : i16 = -10_00;

fn clip(x : i16, lo : i16, hi : i16) -> i16 { return std::cmp::max(lo, std::cmp::min(hi, x)); }

pub fn format_score(score : i16) -> String
{
  if score == i16::MAX   { return String::from("MAX");   }
  if score == i16::MIN   { return String::from("MIN");   }
  if score == i16::MIN+1 { return String::from("MIN+1"); }

  if score >= MINIMUM_PROVEN_MATE { return format!("#+{}", (250_01 - score) / 2); }
  if MINIMUM_PROVEN_LOSS >= score { return format!("#-{}", (250_01 + score) / 2); }

  if score == 0 { return String::from("0.00"); }

  return format!("{:+.2}", score as f64 / 100.0);
}

pub fn format_uci_score(score : i16) -> String
{
  if score >= MINIMUM_PROVEN_MATE { return format!("mate {}",  (250_01 - score) / 2); }
  if MINIMUM_PROVEN_LOSS >= score { return format!("mate -{}", (250_01 + score) / 2); }
  return format!("cp {}", score);
}
