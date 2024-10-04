use crate::context::Context;
use crate::misc::Op;
use crate::movegen::Selectivity;
use crate::movetype::{Move, fast_eq};
use crate::state::State;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Stage {
  GenerateMoves               = 0,
  ScoreCaptures               = 1,  // also includes promotions
  EmitWinningNeutralCaptures  = 2,  //  "    "    "    "    "
  ScoreQuietMoves             = 3,
  EmitQuietMoves              = 4,
  EmitLosingCaptures          = 5,
  Done                        = 6,
}

pub struct MoveSelector {
  pub stage   : Stage,
  selectivity : Selectivity,
  height      : u8,
  hint_move   : Move,

  early_moves  : [Move;  64],
  late_moves   : [Move; 128],
  early_len    : u32,
  late_len     : u32,
  next_capture : u32,
  next_quiet   : u32,
}

impl MoveSelector {
  pub fn new(
    requested : Selectivity,
    height    : u8,
    hint_move : Move,
  ) -> Self
  {
    return Self {
      stage:        Stage::GenerateMoves,
      selectivity:  requested,
      height:       height,
      hint_move:    hint_move,
      early_moves:  [Move::NULL;  64],
      late_moves:   [Move::NULL; 128],
      early_len:    0,
      late_len:     0,
      next_capture: 0,
      next_quiet:   0,
    };
  }

  // Increments history score for the last-emitted late move and
  //   decrements the history scores for the previous late moves.
  pub fn update_history_scores(&self, context : &mut Context)
  {
    if self.stage < Stage::EmitQuietMoves { return; }

    // The empty slot is at next_quiet-1, so the last-emitted move is at next_quiet-2
    if self.next_quiet < 2 { return; }
    let last = &self.late_moves[(self.next_quiet-2) as usize];
    context.update_history(last.piece, last.src as usize, last.dst as usize, true);

    for x in 0..(self.next_quiet-2) {
      let mv = &self.late_moves[x as usize];
      context.update_history(mv.piece, mv.src as usize, mv.dst as usize, false);
    }
  }

  pub fn len(&self) -> u32
  {
    if self.stage == Stage::GenerateMoves { return 0; }
    return self.early_len + self.late_len - 2;
  }

  pub fn next(&mut self, state : &mut State, context : &Context) -> Option<Move>
  {
    loop {
      match self.stage {
        Stage::GenerateMoves => {
          // Perhaps make this stage a separate method
          //   to make life easier on the branch predictors.
          if state.incheck { self.selectivity = Selectivity::Everything; }
          self.next_capture = 1;
          self.next_quiet = 1;
          (self.early_len, self.late_len) = state.generate_legal_moves(
            self.selectivity, &mut self.early_moves, &mut self.late_moves, 1
          );
          self.stage = Stage::ScoreCaptures;
          continue;
        }

        Stage::ScoreCaptures => {
          for x in 1..self.early_len {
            let mv = &mut self.early_moves[x as usize];
            // Put promotions after captures that might win a queen or rook's
            //   worth of material but before captures that win anything less
            // It's not clear how to easily score en passant, so we simply
            //   score it as cleanly winning a pawn
            if mv.is_unusual() {
              mv.score = if mv.is_promotion() { 40 } else { 10 };
              continue;
            }
            let attacker = Op {square: mv.src, piece: mv.piece};
            let target = Op {square: mv.dst, piece: mv.captured};
            let prediction = state.analyze_exchange(attacker, target);
            debug_assert!(prediction.abs() < 128, "absurd prediction {}", prediction);
            mv.score = prediction as i8;
          }
          self.stage = Stage::EmitWinningNeutralCaptures;
          continue;
        }

        Stage::EmitWinningNeutralCaptures => {
          if self.next_capture >= self.early_len {
            self.stage = Stage::ScoreQuietMoves;
            continue;
          }
          //    next capture
          //    │     best move
          //    ┴─    ┴─
          // .. M1 M2 M3 M4 M5  before
          // M3 M1 M2 .. M4 M5  after first clone
          // M3 .. M2 M1 M4 M5  after second clone

          let mut selection_index = self.next_capture;
          let mut selection_score = self.early_moves[self.next_capture as usize].score;

          for x in (self.next_capture+1)..(self.early_len) {
            if self.early_moves[x as usize].score > selection_score {
              selection_index = x;
              selection_score = self.early_moves[x as usize].score;
            }
          }
          if selection_score < 0 { self.stage = Stage::ScoreQuietMoves; continue; }
          self.early_moves[(self.next_capture-1) as usize] = self.early_moves[selection_index as usize].clone();
          self.early_moves[selection_index as usize] = self.early_moves[self.next_capture as usize].clone();
          self.next_capture += 1;
          let mv = &self.early_moves[(self.next_capture-2) as usize];
          if fast_eq(mv, &self.hint_move) {
            continue;
          }
          return Some(mv.clone());
        }

        Stage::ScoreQuietMoves => {
          let killer_0 = &context.killer_table[self.height as usize].0;
          let killer_1 = &context.killer_table[self.height as usize].1;
          for x in 1..self.late_len {
            let mv = &mut self.late_moves[x as usize];
            let mut score = context.lookup_history(mv.piece, mv.src as usize, mv.dst as usize);

            if      fast_eq(mv, killer_0) { score = 96 + (score >> 2); }
            else if fast_eq(mv, killer_1) { score = 95 + (score >> 2); }

            mv.score = score;
          }
          self.stage = Stage::EmitQuietMoves;
          continue;
        }

        Stage::EmitQuietMoves => {
          if self.next_quiet >= self.late_len {
            self.stage = Stage::EmitLosingCaptures;
            continue;
          }
          let mut selection_index = self.next_quiet;
          let mut selection_score = self.late_moves[self.next_quiet as usize].score;

          for x in (self.next_quiet+1)..(self.late_len) {
            if self.late_moves[x as usize].score > selection_score {
              selection_index = x;
              selection_score = self.late_moves[x as usize].score;
            }
          }
          self.late_moves[(self.next_quiet-1) as usize] = self.late_moves[selection_index as usize].clone();
          self.late_moves[selection_index as usize] = self.late_moves[self.next_quiet as usize].clone();
          self.next_quiet += 1;
          let mv = &self.late_moves[(self.next_quiet-2) as usize];
          if fast_eq(mv, &self.hint_move) {
            continue;
          }
          return Some(mv.clone());
        }

        Stage::EmitLosingCaptures => {
          if self.next_capture >= self.early_len {
            self.stage = Stage::Done;
            continue;
          }
          let mut selection_index = self.next_capture;
          let mut selection_score = self.early_moves[self.next_capture as usize].score;

          for x in (self.next_capture+1)..(self.early_len) {
            if self.early_moves[x as usize].score > selection_score {
              selection_index = x;
              selection_score = self.early_moves[x as usize].score;
            }
          }
          self.early_moves[(self.next_capture-1) as usize] = self.early_moves[selection_index as usize].clone();
          self.early_moves[selection_index as usize] = self.early_moves[self.next_capture as usize].clone();
          self.next_capture += 1;
          let mv = &self.early_moves[(self.next_capture-2) as usize];
          if fast_eq(mv, &self.hint_move) {
            continue;
          }
          return Some(mv.clone());
        }

        Stage::Done => { return None; }
      }
    }
  }
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

// Since rust does not have a fall-through variant of match, we could resort to labelled breaks
//   to get the control flow graph we really want. Here is an example:
//
//   pub fn demo(stage : usize, x : i8) -> i8
//   {
//     let mut x = x;
//     'e: loop {
//       'd: loop {
//         'c: loop {
//           'b: loop {
//             match stage {
//               0 => {} //    a
//               1 => { break 'b; }
//               2 => { break 'c; }
//               3 => { break 'd; }
//               _ => { break 'e; }
//             }
//             // case A
//             x *= 2;
//             break;
//           }
//           // case B
//           x *= 3;
//           break;
//         }
//         // case C
//         x *= 5;
//         break;
//       }
//       // case D
//       x *= 7;
//       break;
//     }
//     // case E
//     x *= 11;
//     return x;
//   }
//
// Using rustc 1.55.0, this compiles to
//
//   example::demo:
//           cmp     rdi, 3
//           ja      .LBB0_3
//           lea     rax, [rip + .LJTI0_0]
//           movsxd  rcx, dword ptr [rax + 4*rdi]
//           add     rcx, rax
//           jmp     rcx
//   .LBB0_4:
//           add     sil, sil
//   .LBB0_5:
//           movzx   eax, sil
//           lea     esi, [rax + 2*rax]
//   .LBB0_6:
//           movzx   eax, sil
//           lea     esi, [rax + 4*rax]
//   .LBB0_2:
//           movzx   eax, sil
//           lea     esi, [8*rax]
//           sub     esi, eax
//   .LBB0_3:
//           movzx   eax, sil
//           lea     ecx, [rax + 4*rax]
//           lea     eax, [rax + 2*rcx]
//           ret
//   .LJTI0_0:
//           .long   .LBB0_4-.LJTI0_0
//           .long   .LBB0_5-.LJTI0_0
//           .long   .LBB0_6-.LJTI0_0
//           .long   .LBB0_2-.LJTI0_0
//
// When there are four stages rather than five, the compiler emits a series of test+jump
//   instructions, but happily, when the number of cases is sufficiently large, the compiler
//   instead emits a jump table (as seen above).
