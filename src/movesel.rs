use crate::context::*;
use crate::misc::*;
use crate::movegen::*;
use crate::movetype::*;
use crate::state::*;

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

  early_moves  : Vec<Move>,
  late_moves   : Vec<Move>,
  next_capture : usize,
  next_quiet   : usize,
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
      early_moves:  Vec::new(),
      late_moves:   Vec::new(),
      next_capture: 0,
      next_quiet:   0,
    };
  }

  // Increments history score for the last-emitted late move and
  //   decrements the history scores for the previous late moves.
  pub fn update_history_scores(&self, context : &mut Context, height : u8)
  {
    if self.stage < Stage::EmitQuietMoves { return; }

    // The empty slot is at next_quiet-1, so the last-emitted move is at next_quiet-2
    if self.next_quiet < 2 { return; }
    let last = &self.late_moves[self.next_quiet-2];
    context.update_history(last.piece as usize, last.dst as usize, height, true);

    for x in 0..(self.next_quiet-2) {
      let mv = &self.late_moves[x];
      context.update_history(mv.piece as usize, mv.dst as usize, height, false);
    }
  }

  pub fn len(&self) -> usize
  {
    if self.stage == Stage::GenerateMoves { return 0; }
    return self.early_moves.len() + self.late_moves.len() - 2;
  }

  pub fn next(&mut self, state : &mut State, context : &Context) -> Option<Move>
  {
    loop {
      match self.stage {
        Stage::GenerateMoves => {
          if state.incheck { self.selectivity = Selectivity::Everything; }
          self.early_moves.reserve(16);
          self.late_moves.reserve(32);
          self.early_moves.push(NULL_MOVE);
          self.late_moves.push(NULL_MOVE);
          self.next_capture = 1;
          self.next_quiet = 1;
          state.generate_legal_moves(
            self.selectivity, &mut self.early_moves, &mut self.late_moves
          );
          self.stage = Stage::ScoreCaptures;
          continue;
        }

        Stage::ScoreCaptures => {
          for x in 1..self.early_moves.len() {
            let mv = &mut self.early_moves[x];
            // Put promotions after captures that might win a queen or rook's
            //   worth of material but before captures that win anything less
            // It's not clear how to easily score en passant, so we simply
            //   score it as a neutral capture
            if mv.is_unusual() {
              mv.score = if mv.is_promotion() { 60 / 2 } else { 0 };
              continue;
            }
            let attacker = Op {square: mv.src, piece: mv.piece};
            let target = Op {square: mv.dst, piece: mv.captured};
            let prediction = state.analyze_exchange(attacker, target);
            debug_assert!(prediction.abs() < 256, "absurd prediction {}", prediction);
            mv.score = (prediction / 2) as i8;
          }
          self.stage = Stage::EmitWinningNeutralCaptures;
          continue;
        }

        Stage::EmitWinningNeutralCaptures => {
          if self.next_capture >= self.early_moves.len() {
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
          let mut selection_score = self.early_moves[self.next_capture].score;

          for x in (self.next_capture+1)..(self.early_moves.len()) {
            if self.early_moves[x].score > selection_score {
              selection_index = x;
              selection_score = self.early_moves[x].score;
            }
          }
          if selection_score < 0 { self.stage = Stage::ScoreQuietMoves; continue; }
          self.early_moves[self.next_capture-1] = self.early_moves[selection_index].clone();
          self.early_moves[selection_index] = self.early_moves[self.next_capture].clone();
          self.next_capture += 1;
          let mv = &self.early_moves[self.next_capture-2];
          if quick_eq(mv, &self.hint_move) {
            continue;
          }
          return Some(mv.clone());
        }

        Stage::ScoreQuietMoves => {
          let killer_0 = &context.killer_table[self.height as usize].0;
          let killer_1 = &context.killer_table[self.height as usize].1;
          for x in 1..self.late_moves.len() {
            let mv = &mut self.late_moves[x];
            mv.score = context.lookup_history(mv.piece as usize, mv.dst as usize, self.height);
            if quick_eq(mv, killer_1) { mv.score = 126; }
            if quick_eq(mv, killer_0) { mv.score = 127; }
          }
          self.stage = Stage::EmitQuietMoves;
          continue;
        }

        Stage::EmitQuietMoves => {
          if self.next_quiet >= self.late_moves.len() {
            self.stage = Stage::EmitLosingCaptures;
            continue;
          }
          let mut selection_index = self.next_quiet;
          let mut selection_score = self.late_moves[self.next_quiet].score;

          for x in (self.next_quiet+1)..(self.late_moves.len()) {
            if self.late_moves[x].score > selection_score {
              selection_index = x;
              selection_score = self.late_moves[x].score;
            }
          }
          self.late_moves[self.next_quiet-1] = self.late_moves[selection_index].clone();
          self.late_moves[selection_index] = self.late_moves[self.next_quiet].clone();
          self.next_quiet += 1;
          let mv = &self.late_moves[self.next_quiet-2];
          if quick_eq(mv, &self.hint_move) {
            continue;
          }
          return Some(mv.clone());
        }

        Stage::EmitLosingCaptures => {
          if self.next_capture >= self.early_moves.len() {
            self.stage = Stage::Done;
            continue;
          }
          let mut selection_index = self.next_capture;
          let mut selection_score = self.early_moves[self.next_capture].score;

          for x in (self.next_capture+1)..(self.early_moves.len()) {
            if self.early_moves[x].score > selection_score {
              selection_index = x;
              selection_score = self.early_moves[x].score;
            }
          }
          self.early_moves[self.next_capture-1] = self.early_moves[selection_index].clone();
          self.early_moves[selection_index] = self.early_moves[self.next_capture].clone();
          self.next_capture += 1;
          let mv = &self.early_moves[self.next_capture-2];
          if quick_eq(mv, &self.hint_move) {
            continue;
          }
          return Some(mv.clone());
        }

        Stage::Done => { return None; }
      }
    }
  }
}
