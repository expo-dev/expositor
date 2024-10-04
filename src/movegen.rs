use crate::color::Color::*;
use crate::dest::*;
use crate::misc::{
  piece_destinations,
  pawn_attacks,
  line_span,
  crux_span,
  salt_span,
  line_thru,
  CASTLE_BTWN,
  CASTLE_CHCK,
};
use crate::movetype::MoveType::*;
use crate::movetype::Move;
use crate::piece::Piece;
use crate::piece::Kind::*;
use crate::piece::{QRBN, QRB};
use crate::state::State;

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
//                                ---- Non-captures ----
//                     Captures   Checking  Non-checking   Promotions
pub enum Selectivity {
  Everything  = 0,  //    X           X           X           X
  ActiveOnly  = 1,  //    X           X           .           X
  GainfulOnly = 2,  //    X           .           .           X
}

// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

impl State {
  pub fn generate_legal_moves(
    &self,
    selectivity : Selectivity,
    early_moves : &mut [Move;  64],
    late_moves  : &mut [Move; 128],
    reserve     : u32
  ) -> (u32, u32)
  {
    let mut early_sz = reserve;
    let mut  late_sz = reserve;

    macro_rules! push_early { ($mv : expr) => { early_moves[early_sz as usize] = $mv; early_sz += 1; } }
    macro_rules! push_late  { ($mv : expr) => {  late_moves[ late_sz as usize] = $mv;  late_sz += 1; } }

    let player   =  self.turn;
    let opponent = !self.turn;

    let player_side   = &self.sides[player];
    let opponent_side = &self.sides[opponent];

    let composite = player_side | opponent_side;

    // Step 0. Calculate locations that would give check

      // Direct checks
      //   If a piece moves to a destination within the vantage of its kind, we know check is
      //   being delivered
      let opp_king_sq = self.boards[opponent+King].trailing_zeros() as usize;
      let   rook_vantage =   rook_destinations(composite, opp_king_sq);
      let bishop_vantage = bishop_destinations(composite, opp_king_sq);
      let vantage_mask = [
        0,
        rook_vantage | bishop_vantage,
        rook_vantage,
        bishop_vantage,
        knight_destinations(opp_king_sq),
        pawn_attacks(opponent, 1 << opp_king_sq)
      ];

      // Discovered checks
      //   If a piece is on a waylay square, we will calculate its span to the enemy king,
      //   and if the piece moves to a destination outside that span, we know check is being
      //   delivered
      let mut waylay_mask = 0;
      for attacker in QRB {
        let mut atk_sources = self.boards[player+attacker];
        while atk_sources != 0 {
          let atk_src = atk_sources.trailing_zeros() as usize;
          let span = match attacker {
            Queen  => line_span(opp_king_sq, atk_src),
            Rook   => crux_span(opp_king_sq, atk_src),
            Bishop => salt_span(opp_king_sq, atk_src),
            _ => unreachable!()
          };
          let intermediate = span & composite;
          if intermediate.count_ones() == 1 {
            waylay_mask |= intermediate;
          }
          atk_sources &= atk_sources - 1;
        }
      }

    // Step 1. Generate moves for the king

      let danger;

      let king_src = self.boards[player+King].trailing_zeros() as usize;
      let king_dests = king_destinations(king_src) & !player_side;
      if king_dests == 0 {
        danger = 0;
      }
      else {
        danger = self.attacks_by(opponent, composite & !self.boards[player+King]);
        let mut legal_dests = king_dests & !danger;
        if legal_dests != 0 {
          let discovery_mask = if (1 << king_src) & waylay_mask == 0
            { !0 } else { line_thru(king_src, opp_king_sq) };
          while legal_dests != 0 {
            let dst = legal_dests.trailing_zeros() as usize;
            let dst_piece = self.squares[dst];

            let gives_check = (1 << dst) & discovery_mask == 0;
            if dst_piece.is_null() {
              let activity = if gives_check { 1 } else { 0 };
              if activity >= selectivity as u8 {
                push_late!(Move {
                  movetype:   Manoeuvre,
                  givescheck: (gives_check as u8) << 1,
                  src:        king_src as i8,
                  dst:        dst as i8,
                  piece:      player+King,
                  captured:   Piece::ZERO,
                  promotion:  Piece::ZERO,
                  score:      0,
                });
              }
            }
            else {
              push_early!(Move {
                movetype:   Capture,
                givescheck: (gives_check as u8) << 1,
                src:        king_src as i8,
                dst:        dst as i8,
                piece:      player+King,
                captured:   dst_piece,
                promotion:  Piece::ZERO,
                score:      0,
              });
            }
            legal_dests &= legal_dests - 1;
          }
        }
      }

    // Step 2. Calculate current checks

      let check_sources = self.attackers(king_src, opponent);
      let num_checks = check_sources.count_ones();

      debug_assert!(self.incheck == (num_checks > 0), "check mismatch {}", self.to_fen());

      // In double check, the only legal moves are king moves
      if num_checks > 1 { return (early_sz, late_sz); }

    // Step 3. Generate moves for pinned pieces

      // This is how we keep track of which pieces we've generated moves for in
      //   this step (so that we don't try to generate moves for them in step 5)
      let mut pinned_mask = 0;

      for attacker in QRB {
        let mut atk_sources = self.boards[opponent+attacker];
        while atk_sources != 0 {
          let atk_src = atk_sources.trailing_zeros() as usize;
          let span = match attacker {
            Queen  => line_span(king_src, atk_src),
            Rook   => crux_span(king_src, atk_src),
            Bishop => salt_span(king_src, atk_src),
            _ => unreachable!()
          };
          let blockers = span & composite;
          if blockers.count_ones() != 1 {
            atk_sources &= atk_sources - 1;
            continue;
          }
          let block_src = blockers.trailing_zeros() as usize;
          let pinned_piece = self.squares[block_src];
          if pinned_piece.color() != player {
            atk_sources &= atk_sources - 1;
            continue;
          }
          pinned_mask |= blockers;
          // If the king is currently in check, a pinned piece cannot
          //   also block that check or take the checking piece
          if num_checks > 0 {
            atk_sources &= atk_sources - 1;
            continue;
          }
          if pinned_piece.kind() == Pawn {
            if (block_src % 8) == (atk_src % 8) {
              let dst = match player {
                White => block_src + 8,
                Black => block_src - 8,
              };
              if composite & (1 << dst) == 0 {
                let discovery_mask = if (1 << block_src) & waylay_mask == 0
                  { !0 } else { line_thru(block_src, opp_king_sq) };
                let gives_check = (((1 << dst) & vantage_mask[Pawn] != 0) as u8)
                                | ((((1 << dst) & discovery_mask == 0) as u8) << 1);
                let activity = if gives_check != 0 { 1 } else { 0 };
                if activity >= selectivity as u8 {
                  push_late!(Move {
                    movetype:   PawnManoeuvre,
                    givescheck: gives_check,
                    src:        block_src as i8,
                    dst:        dst as i8,
                    piece:      pinned_piece,
                    captured:   Piece::ZERO,
                    promotion:  Piece::ZERO,
                    score:      0,
                  });
                }
                let double_step = match player {
                  White => block_src < 16,
                  Black => block_src > 47
                };
                if double_step {
                  let pass = dst;
                  let dst = match player {
                    White => pass + 8,
                    Black => pass - 8,
                  };
                  if composite & (1 << dst) == 0 {
                    let gives_check = (((1 << dst) & vantage_mask[Pawn] != 0) as u8)
                                    | ((((1 << dst) & discovery_mask == 0) as u8) << 1);
                    let activity = if gives_check != 0 { 1 } else { 0 };
                    if activity >= selectivity as u8 {
                      push_late!(Move {
                        movetype:   PawnManoeuvre,
                        givescheck: gives_check,
                        src:        block_src as i8,
                        dst:        dst as i8,
                        piece:      pinned_piece,
                        captured:   Piece::ZERO,
                        promotion:  Piece::ZERO,
                        score:      0,
                      });
                    }
                  }
                }
              }
            }
            else {
              let can_capture = match player {
                White => { block_src + 7 == atk_src || block_src + 9 == atk_src },
                Black => { block_src - 7 == atk_src || block_src - 9 == atk_src }
              };
              if can_capture {
                let discovery_mask = if (1 << block_src) & waylay_mask == 0
                  { !0 } else { line_thru(block_src, opp_king_sq) };
                if atk_src > 55 || atk_src < 8 {
                  let discovery_check = (1 << atk_src) & discovery_mask == 0;
                  for promo in QRBN {
                    let direct_check =
                      piece_destinations(promo, atk_src, composite & !(1 << block_src))
                      & (1 << opp_king_sq) != 0;
                    let gives_check = ((discovery_check as u8) << 1) | (direct_check as u8);
                    push_early!(Move {
                      movetype:   PromotionByCapture,
                      givescheck: gives_check,
                      src:        block_src as i8,
                      dst:        atk_src as i8,
                      piece:      pinned_piece,
                      captured:   opponent+attacker,
                      promotion:  player+promo,
                      score:      0,
                    });
                  }
                }
                else {
                  let gives_check = (((1 << atk_src) & vantage_mask[Pawn] != 0) as u8)
                                  | ((((1 << atk_src) & discovery_mask == 0) as u8) << 1);
                  push_early!(Move {
                    movetype:   PawnCapture,
                    givescheck: gives_check,
                    src:        block_src as i8,
                    dst:        atk_src as i8,
                    piece:      pinned_piece,
                    captured:   opponent+attacker,
                    promotion:  Piece::ZERO,
                    score:      0,
                  });
                }
              }
            }
          }
          else {
            let dests = piece_destinations(pinned_piece.kind(), block_src, composite);

            let discovery_mask = if (1 << block_src) & waylay_mask == 0
              { !0 } else { line_thru(block_src, opp_king_sq) };

            if dests & (1 << atk_src) != 0 {
              let gives_check =
                (((1 << atk_src) & vantage_mask[pinned_piece.kind()] != 0) as u8) |
                ((((1 << atk_src) & discovery_mask == 0) as u8) << 1);
              push_early!(Move {
                movetype:   Capture,
                givescheck: gives_check,
                src:        block_src as i8,
                dst:        atk_src as i8,
                piece:      pinned_piece,
                captured:   opponent+attacker,
                promotion:  Piece::ZERO,
                score:      0,
              });
            }
            let mut dests = dests & span;
            while dests != 0 {
              let dst = dests.trailing_zeros() as usize;
              let gives_check =
                (((1 << dst) & vantage_mask[pinned_piece.kind()] != 0) as u8) |
                ((((1 << atk_src) & discovery_mask == 0) as u8) << 1);
              let activity = if gives_check != 0 { 1 } else { 0 };
              if activity >= selectivity as u8 {
                push_late!(Move {
                  movetype:   Manoeuvre,
                  givescheck: gives_check,
                  src:        block_src as i8,
                  dst:        dst as i8,
                  piece:      pinned_piece,
                  captured:   Piece::ZERO,
                  promotion:  Piece::ZERO,
                  score:      0,
                });
              }
              dests &= dests - 1;
            }
          }
          atk_sources &= atk_sources - 1;
        }
      }

    // Step 4. Calculate legal destinations

      // If we are in (single) check, besides moving the king, we can either capture the
      //   checking piece or block the check
      let legal_mask;
      if num_checks == 0 {
        legal_mask = !0;
      }
      else {
        let chk_src = check_sources.trailing_zeros() as usize;
        let block_mask =
          if self.squares[chk_src].is_ranging() { line_span(king_src, chk_src) } else { 0 };
        legal_mask = check_sources | block_mask;
      }

    // Step 5. Generate moves for unpinned pieces

      let pinned_mask = !pinned_mask;

      // Queens, rooks, bishops, and knights

      for piece in QRBN {
        let mut sources = self.boards[player+piece] & pinned_mask;
        while sources != 0 {
          let src = sources.trailing_zeros() as usize;
          let mut dests = piece_destinations(piece, src, composite);
          dests &= !player_side;
          dests &= legal_mask;

          if dests == 0 {
            sources &= sources - 1;
            continue;
          }

          let discovery_mask = if (1 << src) & waylay_mask == 0
            { !0 } else { line_thru(src, opp_king_sq) };

          let mut capture_dests = dests & composite;
          while capture_dests != 0 {
            let dst = capture_dests.trailing_zeros() as usize;
            let gives_check = (((1 << dst) & vantage_mask[piece] != 0) as u8)
                            | ((((1 << dst) & discovery_mask == 0) as u8) << 1);
            push_early!(Move {
              movetype:   Capture,
              givescheck: gives_check,
              src:        src as i8,
              dst:        dst as i8,
              piece:      player+piece,
              captured:   self.squares[dst],
              promotion:  Piece::ZERO,
              score:      0,
            });
            capture_dests &= capture_dests - 1;
          }

          let mut manoeuvre_dests = dests & !composite;
          while manoeuvre_dests != 0 {
            let dst = manoeuvre_dests.trailing_zeros();
            let gives_check = (((1 << dst) & vantage_mask[piece] != 0) as u8)
                            | ((((1 << dst) & discovery_mask == 0) as u8) << 1);
            let activity = if gives_check != 0 { 1 } else { 0 };
            if activity >= selectivity as u8 {
              push_late!(Move {
                movetype:   Manoeuvre,
                givescheck: gives_check,
                src:        src as i8,
                dst:        dst as i8,
                piece:      player+piece,
                captured:   Piece::ZERO,
                promotion:  Piece::ZERO,
                score:      0,
              });
            }
            manoeuvre_dests &= manoeuvre_dests - 1;
          }
          sources &= sources - 1;
        }
      }

      // Pawns

      let mut sources = self.boards[player+Pawn] & pinned_mask;
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;

        let discovery_mask = if (1 << src) & waylay_mask == 0
          { !0 } else { line_thru(src, opp_king_sq) };

        // Manoeuvres
        let dst = match player { White => src + 8, Black => src - 8 };
        if composite & (1 << dst) == 0 {
          // Promotion
          if dst > 55 || dst < 8 {
            if legal_mask & (1 << dst) != 0 {
              let discovery_check = (1 << dst) & discovery_mask == 0;
              for promo in QRBN {
                let direct_check =
                  piece_destinations(promo, dst, composite & !(1 << src))
                  & (1 << opp_king_sq) != 0;
                let gives_check = ((discovery_check as u8) << 1) | (direct_check as u8);
                push_early!(Move {
                  movetype:   Promotion,
                  givescheck: gives_check,
                  src:        src as i8,
                  dst:        dst as i8,
                  piece:      player+Pawn,
                  captured:   Piece::ZERO,
                  promotion:  player+promo,
                  score:      0,
                });
              }
            }
          }
          // Single and double step
          else {
            if legal_mask & (1 << dst) != 0 {
              let gives_check = (((1 << dst) & vantage_mask[Pawn] != 0) as u8)
                              | ((((1 << dst) & discovery_mask == 0) as u8) << 1);
              let activity = if gives_check != 0 { 1 } else { 0 };
              if activity >= selectivity as u8 {
                push_late!(Move {
                  movetype:   PawnManoeuvre,
                  givescheck: gives_check,
                  src:        src as i8,
                  dst:        dst as i8,
                  piece:      player+Pawn,
                  captured:   Piece::ZERO,
                  promotion:  Piece::ZERO,
                  score:      0,
                });
              }
            }
            let double_step = match player {
              White => src < 16,
              Black => src > 47
            };
            if double_step {
              let pass = dst;
              let dst = match player {
                White => pass + 8,
                Black => pass - 8,
              };
              if composite & (1 << dst) == 0 {
                if legal_mask & (1 << dst) != 0 {
                  let gives_check = (((1 << dst) & vantage_mask[Pawn] != 0) as u8)
                                  | ((((1 << dst) & discovery_mask == 0) as u8) << 1);
                  let activity = if gives_check != 0 { 1 } else { 0 };
                  if activity >= selectivity as u8 {
                    push_late!(Move {
                      movetype:   PawnManoeuvre,
                      givescheck: gives_check,
                      src:        src as i8,
                      dst:        dst as i8,
                      piece:      player+Pawn,
                      captured:   Piece::ZERO,
                      promotion:  Piece::ZERO,
                      score:      0,
                    });
                  }
                }
              }
            }
          }
        }

        // Captures
        let mut capture_squares = match player {
          White => [src as i8 + 7, src as i8 + 9],
          Black => [src as i8 - 9, src as i8 - 7],
        };
        if src % 8 == 0 { capture_squares[0] = -1; }
        if src % 8 == 7 { capture_squares[1] = -1; }
        for x in 0..2 {
          let dst = capture_squares[x];
          if dst < 0 { continue; }
          let dst = dst as usize;
          if legal_mask & (1 << dst) == 0 { continue; }
          if opponent_side & (1 << dst) == 0 { continue; }
          let captured_piece = self.squares[dst];
          // Promotion by capture
          if dst > 55 || dst < 8 {
            for promo in QRBN {
              let discovered_check = (1 << dst) & discovery_mask == 0;
              let direct_check =
                piece_destinations(promo, dst, composite & !(1 << src))
                & (1 << opp_king_sq) != 0;
              let gives_check = ((discovered_check as u8) << 1) | (direct_check as u8);
              push_early!(Move {
                movetype:   PromotionByCapture,
                givescheck: gives_check,
                src:        src as i8,
                dst:        dst as i8,
                piece:      player+Pawn,
                captured:   captured_piece,
                promotion:  player+promo,
                score:      0,
              });
            }
          }
          // Capture ordinaire
          else {
            let gives_check = (((1 << dst) & vantage_mask[Pawn] != 0) as u8)
                            | ((((1 << dst) & discovery_mask == 0) as u8) << 1);
            push_early!(Move {
              movetype:   PawnCapture,
              givescheck: gives_check,
              src:        src as i8,
              dst:        dst as i8,
              piece:      player+Pawn,
              captured:   captured_piece,
              promotion:  Piece::ZERO,
              score:      0,
            });
          }
        }

        sources &= sources - 1;
      }

    // Step 6. Generate a capture en passant (if possible)

      if self.enpass >= 0 {
        let mut sources = match player {
          White => [self.enpass - 9, self.enpass - 7],
          Black => [self.enpass + 7, self.enpass + 9],
        };
        if self.enpass % 8 == 0 { sources[0] = -1; }
        if self.enpass % 8 == 7 { sources[1] = -1; }
        for x in 0..2 {
          let src = sources[x];
          if src < 0 { continue; }
          let src = src as usize;
          if self.boards[player+Pawn] & (1 << src) == 0 { continue; }
          let opsq = match player {
            White => self.enpass - 8,
            Black => self.enpass + 8,
          };

          let mut scratch = State {
            sides:   self.sides,
            boards:  self.boards,
            squares: [Piece::ZERO; 64],
            rights:  0,
            enpass:  0,
            incheck: false,
            turn:    White,
            dfz:     0,
            ply:     0,
            key:     0,
            s1:      Vec::new(),
          };

          scratch.boards[player+Pawn]   ^= (1 << src) | (1 << self.enpass);
          scratch.sides[player]         ^= (1 << src) | (1 << self.enpass);
          scratch.boards[opponent+Pawn] ^= 1 << opsq;
          scratch.sides[opponent]       ^= 1 << opsq;

          let legal = !scratch.in_check(player);

          if legal {
            let gives_check = (scratch.in_check(opponent) as u8) << 2;
            // NOTE that givescheck is not set properly here (although gives_check may be
            //   true, gives_direct_check and gives_discovered_check will always be false)
            push_early!(Move {
              movetype:   CaptureEnPassant,
              givescheck: gives_check,
              src:        src as i8,
              dst:        self.enpass,
              piece:      player+Pawn,
              captured:   opponent+Pawn,
              promotion:  Piece::ZERO,
              score:      0,
            });
          }
        }
      }

    // Step 7. Generate castling moves (if possible)

      if num_checks == 0 {

        let rights_mask = match player { White => [1, 2], Black => [4, 8] };
        let intermediate = &CASTLE_BTWN[player];
        let travel = &CASTLE_CHCK[player];
        let king_start = match player { White => 4,      Black => 60       };
        let king_stop  = match player { White => [6, 2], Black => [62, 58] };
        let rook_stop  = match player { White => [5, 3], Black => [61, 59] };

        for x in 0..2 {
          if self.rights & rights_mask[x] == 0 { continue; }
          if composite & intermediate[x] != 0 { continue; }
          if danger & travel[x] != 0 { continue; }

          // Castling can deliver both direct checks (in the obvious way,
          //   up the file) but can also deliver discovered checks, e.g.:
          //
          //   O-O+ from 1K2k2r/rp3pp1/2p1p2p/3p4/3q1P2/8/7P/q7 b k - 1 35
          //
          let direct_check = (1 << rook_stop[x]) & vantage_mask[Rook] != 0;
          let discovered_check = ((king_start as u8)/8 == (opp_king_sq as u8)/8)
            && ((1 << king_start) & waylay_mask != 0);
          let gives_check = ((discovered_check as u8) << 1) | (direct_check as u8);

          let activity = if gives_check > 0 { 1 } else { 0 };
          if activity >= selectivity as u8 {
            push_late!(Move {
              movetype:   Castle,
              givescheck: gives_check,
              src:        king_start,
              dst:        king_stop[x],
              piece:      player+King,
              captured:   Piece::ZERO,
              promotion:  Piece::ZERO,
              score:      0,
            });
          }
        }
      }

    // End

    return (early_sz, late_sz);
  }

  pub fn collect_legal_moves(&self, selectivity : Selectivity) -> Vec<Move>
  {
    let mut early_moves = [Move::NULL;  64];
    let mut late_moves  = [Move::NULL; 128];
    let (early_sz, late_sz) =
      self.generate_legal_moves(selectivity, &mut early_moves, &mut late_moves, 0);
    let mut moves = Vec::new();
    for x in 0..early_sz { moves.push(early_moves[x as usize]); }
    for x in 0..late_sz  { moves.push( late_moves[x as usize]); }
    return moves;
  }

  pub fn legal_moves(&self, selectivity : Selectivity) -> LegalMoves
  {
    let mut moves = LegalMoves {
      early: [Move::NULL;  64],
      late:  [Move::NULL; 128],
      early_len: 0,
      late_len:  0,
      index:     0,
      snd:   false
    };
    let (e_sz, l_sz) =
      self.generate_legal_moves(selectivity, &mut moves.early, &mut moves.late, 0);
    moves.early_len = e_sz;
    moves.late_len  = l_sz;
    return moves;
  }
}

pub struct LegalMoves {
  pub early : [Move;  64],
  pub late  : [Move; 128],
  pub early_len : u32,
  pub late_len  : u32,
  index : u32,
  snd   : bool,
}

impl LegalMoves {
  pub fn length(&self) -> u32
  {
    return self.early_len + self.late_len;
  }
}

impl Iterator for LegalMoves {
  type Item = Move;

  fn next(&mut self) -> Option<Self::Item> {
    if !self.snd {
      if self.index < self.early_len {
        let mv = self.early[self.index as usize];
        self.index += 1;
        return Some(mv);
      }
      self.snd = true;
      self.index = 0;
    }
    if self.index < self.late_len {
      let mv = self.late[self.index as usize];
      self.index += 1;
      return Some(mv);
    }
    return None;
  }
}
