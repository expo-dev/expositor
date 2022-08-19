use crate::color::*;
use crate::dest::*;
use crate::misc::*;
use crate::movetype::*;
use crate::piece::*;
use crate::state::*;

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
    early_moves : &mut Vec<Move>,
    late_moves  : &mut Vec<Move>,
  )
  {
    let player   =  self.turn as usize * 8;
    let opponent = !self.turn as usize * 8;

    let player_side = &self.sides[self.turn as usize];
    let composite = player_side | self.sides[!self.turn as usize];

    let danger;

    // Step 0. Calculate locations that would give check

      // Direct checks
      //   If a piece moves to a destination within the vantage of its kind, we know check is
      //   being delivered
      let opp_king_sq = self.boards[opponent+KING].trailing_zeros() as usize;
      let   rook_vantage =   rook_destinations(composite, opp_king_sq);
      let bishop_vantage = bishop_destinations(composite, opp_king_sq);
      let vantage_mask = [
        0, rook_vantage | bishop_vantage, rook_vantage, bishop_vantage,
        knight_destinations(opp_king_sq), pawn_attacks(!self.turn, 1u64 << opp_king_sq)
      ];

      // Discovered checks
      //   If a piece is on a waylay square, we will calculate its span to the enemy king,
      //   and if the piece moves to a destination outside that span, we know check is being
      //   delivered
      let mut waylay_mask = 0;
      for attacker in 1..4 {
        let mut atk_sources = self.boards[player+attacker];
        while atk_sources != 0 {
          let atk_src = atk_sources.trailing_zeros() as usize;
          let span = match attacker {
            QUEEN  =>  any_span(opp_king_sq, atk_src),
            ROOK   => crux_span(opp_king_sq, atk_src),
            BISHOP => salt_span(opp_king_sq, atk_src),
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

      let king_src = self.boards[player+KING].trailing_zeros() as usize;
      let king_dests = king_destinations(king_src) & !player_side;
      if king_dests == 0 {
        danger = 0;
      }
      else {
        danger = self.attacked_by(!self.turn, composite & !self.boards[player+KING]);
        let mut legal_dests = king_dests & !danger;
        if legal_dests != 0 {
          let discovery_mask = if (1u64 << king_src) & waylay_mask == 0
            { 0xFFFFFFFFFFFFFFFF } else { line_through(king_src as usize, opp_king_sq) };
          while legal_dests != 0 {
            let dst = legal_dests.trailing_zeros() as i8;
            let dst_piece = self.squares[dst as usize];

            let gives_check = (1u64 << dst) & discovery_mask == 0;
            if dst_piece.is_null() {
              let activity = if gives_check { 1 } else { 0 };
              if activity >= selectivity as u8 {
                late_moves.push(Move {
                  movetype:   MoveType::Manoeuvre,
                  givescheck: (gives_check as u8) << 1,
                  src:        king_src as i8,
                  dst:        dst,
                  piece:      Piece::from_usize(player+KING),
                  captured:   ZERO_PIECE,
                  promotion:  ZERO_PIECE,
                  score:      0,
                });
              }
            }
            else {
              early_moves.push(Move {
                movetype:   MoveType::Capture,
                givescheck: (gives_check as u8) << 1,
                src:        king_src as i8,
                dst:        dst,
                piece:      Piece::from_usize(player+KING),
                captured:   dst_piece,
                promotion:  ZERO_PIECE,
                score:      0,
              });
            }
            legal_dests &= legal_dests - 1;
          }
        }
      }

    // Step 2. Calculate current checks

      let check_sources = self.attackers_of(king_src, !self.turn);
      let num_checks = check_sources.count_ones();

      // In double check, the only legal moves are king moves
      if num_checks > 1 { return; }

      debug_assert!(self.incheck == (num_checks > 0), "check mismatch @ {}", self.to_fen());

    // Step 3. Generate moves for pinned pieces

      // This is how we keep track of which pieces we've generated moves for in this step
      //   (so that we don't try to generate moves for them in step 5)
      let mut pinned_mask = 0;

      for attacker in 1..4 {
        let mut atk_sources = self.boards[opponent+attacker];
        while atk_sources != 0 {
          let atk_src = atk_sources.trailing_zeros() as usize;
          let span = match attacker {
            QUEEN  =>  any_span(king_src, atk_src),
            ROOK   => crux_span(king_src, atk_src),
            BISHOP => salt_span(king_src, atk_src),
            _ => unreachable!()
          };
          let blockers = span & composite;
          if blockers.count_ones() != 1 {
            atk_sources &= atk_sources - 1;
            continue;
          }
          let block_src = blockers.trailing_zeros() as usize;
          let pinned_piece = self.squares[block_src];
          if pinned_piece.color() != self.turn {
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
          if pinned_piece.kind() == PAWN {
            if (block_src % 8) == (atk_src % 8) {
              let dst = match self.turn {
                Color::White => block_src + 8,
                Color::Black => block_src - 8,
              };
              if composite & (1u64 << dst) == 0 {
                let discovery_mask = if (1u64 << block_src) & waylay_mask == 0
                  { 0xFFFFFFFFFFFFFFFF } else { line_through(block_src, opp_king_sq) };
                let gives_check = (((1u64 << dst) & vantage_mask[PAWN] != 0) as u8)
                                | ((((1u64 << dst) & discovery_mask == 0) as u8) << 1);
                let activity = if gives_check != 0 { 1 } else { 0 };
                if activity >= selectivity as u8 {
                  late_moves.push(Move {
                    movetype:   MoveType::PawnManoeuvre,
                    givescheck: gives_check,
                    src:        block_src as i8,
                    dst:        dst as i8,
                    piece:      pinned_piece,
                    captured:   ZERO_PIECE,
                    promotion:  ZERO_PIECE,
                    score:      0,
                  });
                }
                let double_step = match self.turn {
                  Color::White => block_src < 16,
                  Color::Black => block_src > 47
                };
                if double_step {
                  let pass = dst;
                  let dst = match self.turn {
                    Color::White => pass + 8,
                    Color::Black => pass - 8,
                  };
                  if composite & (1u64 << dst) == 0 {
                    let gives_check = (((1u64 << dst) & vantage_mask[PAWN] != 0) as u8)
                                    | ((((1u64 << dst) & discovery_mask == 0) as u8) << 1);
                    let activity = if gives_check != 0 { 1 } else { 0 };
                    if activity >= selectivity as u8 {
                      late_moves.push(Move {
                        movetype:   MoveType::PawnManoeuvre,
                        givescheck: gives_check,
                        src:        block_src as i8,
                        dst:        dst as i8,
                        piece:      pinned_piece,
                        captured:   ZERO_PIECE,
                        promotion:  ZERO_PIECE,
                        score:      0,
                      });
                    }
                  }
                }
              }
            }
            else {
              let can_capture = match self.turn {
                Color::White => { block_src + 7 == atk_src || block_src + 9 == atk_src },
                Color::Black => { block_src - 7 == atk_src || block_src - 9 == atk_src }
              };
              if can_capture {
                let discovery_mask = if (1u64 << block_src) & waylay_mask == 0
                  { 0xFFFFFFFFFFFFFFFF } else { line_through(block_src, opp_king_sq) };
                if atk_src > 55 || atk_src < 8 {
                  let discovery_check = (1u64 << atk_src) & discovery_mask == 0;
                  for promo in 1..5 {
                    let direct_check = 
                      piece_destinations(promo, atk_src, composite & !(1u64 << block_src))
                      & (1u64 << opp_king_sq) != 0;
                    let gives_check = ((discovery_check as u8) << 1) | (direct_check as u8);
                    early_moves.push(Move {
                      movetype:   MoveType::PromotionByCapture,
                      givescheck: gives_check,
                      src:        block_src as i8,
                      dst:        atk_src as i8,
                      piece:      pinned_piece,
                      captured:   Piece::from_usize(opponent+attacker),
                      promotion:  Piece::from_usize(player+promo),
                      score:      0,
                    });
                  }
                }
                else {
                  let gives_check = (((1u64 << atk_src) & vantage_mask[PAWN] != 0) as u8)
                                  | ((((1u64 << atk_src) & discovery_mask == 0) as u8) << 1);
                  early_moves.push(Move {
                    movetype:   MoveType::PawnCapture,
                    givescheck: gives_check,
                    src:        block_src as i8,
                    dst:        atk_src as i8,
                    piece:      pinned_piece,
                    captured:   Piece::from_usize(opponent+attacker),
                    promotion:  ZERO_PIECE,
                    score:      0,
                  });
                }
              }
            }
          }
          else {
            let dests = piece_destinations(pinned_piece.kind(), block_src, composite);

            let discovery_mask = if (1u64 << block_src) & waylay_mask == 0
              { 0xFFFFFFFFFFFFFFFF } else { line_through(block_src as usize, opp_king_sq) };

            if dests & (1u64 << atk_src) != 0 {
              let gives_check =
                (((1u64 << atk_src) & vantage_mask[pinned_piece.kind()] != 0) as u8) |
                ((((1u64 << atk_src) & discovery_mask == 0) as u8) << 1);
              early_moves.push(Move {
                movetype:   MoveType::Capture,
                givescheck: gives_check,
                src:        block_src as i8,
                dst:        atk_src as i8,
                piece:      pinned_piece,
                captured:   Piece::from_usize(opponent+attacker),
                promotion:  ZERO_PIECE,
                score:      0,
              });
            }
            let mut dests = dests & span;
            while dests != 0 {
              let dst = dests.trailing_zeros() as i8;
              let gives_check =
                (((1u64 << dst) & vantage_mask[pinned_piece.kind()] != 0) as u8) |
                ((((1u64 << atk_src) & discovery_mask == 0) as u8) << 1);
              let activity = if gives_check != 0 { 1 } else { 0 };
              if activity >= selectivity as u8 {
                late_moves.push(Move {
                  movetype:   MoveType::Manoeuvre,
                  givescheck: gives_check,
                  src:        block_src as i8,
                  dst:        dst,
                  piece:      pinned_piece,
                  captured:   ZERO_PIECE,
                  promotion:  ZERO_PIECE,
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
        legal_mask = 0xFFFFFFFFFFFFFFFF;
      }
      else {
        let chk_src = check_sources.trailing_zeros() as usize;
        let block_mask =
          if self.squares[chk_src].is_ranging() { any_span(king_src, chk_src) } else { 0 };
        legal_mask = check_sources | block_mask;
      }

    // Step 5. Generate moves for unpinned pieces

      let pinned_mask = !pinned_mask;

      // Queens, rooks, bishops, and knights

      for piece in 1..5 {
        let mut sources = self.boards[player+piece] & pinned_mask;
        while sources != 0 {
          let src = sources.trailing_zeros() as i8;
          let mut dests = piece_destinations(piece, src as usize, composite);
          dests &= !player_side;
          dests &= legal_mask;

          if dests == 0 {
            sources &= sources - 1;
            continue;
          }

          let discovery_mask = if (1u64 << src) & waylay_mask == 0
            { 0xFFFFFFFFFFFFFFFF } else { line_through(src as usize, opp_king_sq) };

          let mut capture_dests = dests & composite;
          while capture_dests != 0 {
            let dst = capture_dests.trailing_zeros() as i8;
            let gives_check = (((1u64 << dst) & vantage_mask[piece] != 0) as u8)
                            | ((((1u64 << dst) & discovery_mask == 0) as u8) << 1);
            early_moves.push(Move {
              movetype:   MoveType::Capture,
              givescheck: gives_check,
              src:        src,
              dst:        dst,
              piece:      Piece::from_usize(player+piece),
              captured:   self.squares[dst as usize],
              promotion:  ZERO_PIECE,
              score:      0,
            });
            capture_dests &= capture_dests - 1;
          }

          let mut manoeuvre_dests = dests & !composite;
          while manoeuvre_dests != 0 {
            let dst = manoeuvre_dests.trailing_zeros() as i8;
            let gives_check = (((1u64 << dst) & vantage_mask[piece] != 0) as u8)
                            | ((((1u64 << dst) & discovery_mask == 0) as u8) << 1);
            let activity = if gives_check != 0 { 1 } else { 0 };
            if activity >= selectivity as u8 {
              late_moves.push(Move {
                movetype:   MoveType::Manoeuvre,
                givescheck: gives_check,
                src:        src,
                dst:        dst,
                piece:      Piece::from_usize(player+piece),
                captured:   ZERO_PIECE,
                promotion:  ZERO_PIECE,
                score:      0,
              });
            }
            manoeuvre_dests &= manoeuvre_dests - 1;
          }
          sources &= sources - 1;
        }
      }

      // Pawns

      let opponent_side = &self.sides[!self.turn as usize];

      let mut sources = self.boards[player+PAWN] & pinned_mask;
      while sources != 0 {
        let src = sources.trailing_zeros() as usize;

        let discovery_mask = if (1u64 << src) & waylay_mask == 0
          { 0xFFFFFFFFFFFFFFFF } else { line_through(src, opp_king_sq) };

        // Manoeuvres
        let dst = match self.turn { Color::White => src + 8, Color::Black => src - 8 };
        if composite & (1u64 << dst) == 0 {
          // Promotion
          if dst > 55 || dst < 8 {
            if legal_mask & (1u64 << dst) != 0 {
              let discovery_check = (1u64 << dst) & discovery_mask == 0;
              for promo in 1..5 {
                let direct_check =
                  piece_destinations(promo, dst, composite & !(1u64 << src))
                  & (1u64 << opp_king_sq) != 0;
                let gives_check = ((discovery_check as u8) << 1) | (direct_check as u8);
                early_moves.push(Move {
                  movetype:   MoveType::Promotion,
                  givescheck: gives_check,
                  src:        src as i8,
                  dst:        dst as i8,
                  piece:      Piece::from_usize(player+PAWN),
                  captured:   ZERO_PIECE,
                  promotion:  Piece::from_usize(player+promo),
                  score:      0,
                });
              }
            }
          }
          // Single and double step
          else {
            if legal_mask & (1u64 << dst) != 0 {
              let gives_check = (((1u64 << dst) & vantage_mask[PAWN] != 0) as u8)
                              | ((((1u64 << dst) & discovery_mask == 0) as u8) << 1);
              let activity = if gives_check != 0 { 1 } else { 0 };
              if activity >= selectivity as u8 {
                late_moves.push(Move {
                  movetype:   MoveType::PawnManoeuvre,
                  givescheck: gives_check,
                  src:        src as i8,
                  dst:        dst as i8,
                  piece:      Piece::from_usize(player+PAWN),
                  captured:   ZERO_PIECE,
                  promotion:  ZERO_PIECE,
                  score:      0,
                });
              }
            }
            let double_step = match self.turn {
              Color::White => src < 16,
              Color::Black => src > 47
            };
            if double_step {
              let pass = dst;
              let dst = match self.turn {
                Color::White => pass + 8,
                Color::Black => pass - 8,
              };
              if composite & (1u64 << dst) == 0 {
                if legal_mask & (1u64 << dst) != 0 {
                  let gives_check = (((1u64 << dst) & vantage_mask[PAWN] != 0) as u8)
                                  | ((((1u64 << dst) & discovery_mask == 0) as u8) << 1);
                  let activity = if gives_check != 0 { 1 } else { 0 };
                  if activity >= selectivity as u8 {
                    late_moves.push(Move {
                      movetype:   MoveType::PawnManoeuvre,
                      givescheck: gives_check,
                      src:        src as i8,
                      dst:        dst as i8,
                      piece:      Piece::from_usize(player+PAWN),
                      captured:   ZERO_PIECE,
                      promotion:  ZERO_PIECE,
                      score:      0,
                    });
                  }
                }
              }
            }
          }
        }

        // Captures
        let mut capture_squares = match self.turn {
          Color::White => [src as i8 + 7, src as i8 + 9],
          Color::Black => [src as i8 - 9, src as i8 - 7],
        };
        if src % 8 == 0 { capture_squares[0] = -1; }
        if src % 8 == 7 { capture_squares[1] = -1; }
        for x in 0..2 {
          let dst = capture_squares[x];
          if dst < 0 { continue; }
          if legal_mask & (1u64 << dst) == 0 { continue; }
          if opponent_side & (1u64 << dst) == 0 { continue; }
          let captured_piece = self.squares[dst as usize];
          // Promotion by capture
          if dst > 55 || dst < 8 {
            for promo in 1..5 {
              let discovered_check = (1u64 << dst) & discovery_mask == 0;
              let direct_check =
                piece_destinations(promo, dst as usize, composite & !(1u64 << src))
                & (1u64 << opp_king_sq) != 0;
              let gives_check = ((discovered_check as u8) << 1) | (direct_check as u8);
              early_moves.push(Move {
                movetype:   MoveType::PromotionByCapture,
                givescheck: gives_check,
                src:        src as i8,
                dst:        dst,
                piece:      Piece::from_usize(player+PAWN),
                captured:   captured_piece,
                promotion:  Piece::from_usize(player+promo),
                score:      0,
              });
            }
          }
          // Capture ordinaire
          else {
            let gives_check = (((1u64 << dst) & vantage_mask[PAWN] != 0) as u8)
                            | ((((1u64 << dst) & discovery_mask == 0) as u8) << 1);
            early_moves.push(Move {
              movetype:   MoveType::PawnCapture,
              givescheck: gives_check,
              src:        src as i8,
              dst:        dst,
              piece:      Piece::from_usize(player+PAWN),
              captured:   captured_piece,
              promotion:  ZERO_PIECE,
              score:      0,
            });
          }
        }

        sources &= sources - 1;
      }

    // Step 6. Generate a capture en passant (if possible)

      if self.enpass >= 0 {
        let mut sources = match self.turn {
          Color::White => [self.enpass - 9, self.enpass - 7],
          Color::Black => [self.enpass + 7, self.enpass + 9],
        };
        if self.enpass % 8 == 0 { sources[0] = -1; }
        if self.enpass % 8 == 7 { sources[1] = -1; }
        for x in 0..2 {
          let src = sources[x];
          if src < 0 { continue; }
          if self.boards[player+PAWN] & (1u64 << src) == 0 { continue; }
          let opsq = match self.turn {
            Color::White => self.enpass - 8,
            Color::Black => self.enpass + 8,
          };

          let mut scratch = State {
            sides:   self.sides,
            boards:  self.boards,
            squares: [ZERO_PIECE; 64],
            rights:  0,
            enpass:  0,
            incheck: false,
            turn:    Color::White,
            dfz:     0,
            ply:     0,
            key:     0,
            s1:      Vec::new(),
          };

          scratch.boards[player+PAWN]        ^= (1u64 << src) | (1u64 << self.enpass);
          scratch.sides[self.turn as usize]  ^= (1u64 << src) | (1u64 << self.enpass);
          scratch.boards[opponent+PAWN]      ^= 1u64 << opsq;
          scratch.sides[!self.turn as usize] ^= 1u64 << opsq;

          let legal = !scratch.in_check(self.turn);

          if legal {
            let gives_check = (scratch.in_check(!self.turn) as u8) << 2;
            // NOTE that givescheck is not set properly here (although gives_check may be
            //   true, gives_direct_check and gives_discovered_check will always be false)
            early_moves.push(Move {
              movetype:   MoveType::CaptureEnPassant,
              givescheck: gives_check,
              src:        src,
              dst:        self.enpass,
              piece:      Piece::from_usize(player+PAWN),
              captured:   Piece::from_usize(opponent+PAWN),
              promotion:  ZERO_PIECE,
              score:      0,
            });
          }
        }
      }

    // Step 7. Generate castling moves (if possible)

      if num_checks == 0 {

        let rights_mask = match self.turn {
          Color::White => [0x01, 0x02], Color::Black => [0x04, 0x08]
        };
        let intermediate = match self.turn {
          Color::White => [0x0000000000000060, 0x000000000000000E],
          Color::Black => [0x6000000000000000, 0x0E00000000000000],
        };
        let travel = match self.turn {
          Color::White => [0x0000000000000070, 0x000000000000001C],
          Color::Black => [0x7000000000000000, 0x1C00000000000000],
        };
        let king_start = match self.turn { Color::White => 4,      Color::Black => 60       };
        let king_stop  = match self.turn { Color::White => [6, 2], Color::Black => [62, 58] };
        let rook_stop  = match self.turn { Color::White => [5, 3], Color::Black => [61, 59] };

        for x in 0..2 {
          if self.rights & rights_mask[x] == 0 { continue; }
          if composite & intermediate[x] != 0 { continue; }
          if danger & travel[x] != 0 { continue; }

          // Castling can deliver both direct checks (in the obvious way,
          //   up the file) but can also deliver discovered checks, e.g.
          //
          //   O-O+ from 1K2k2r/rp3pp1/2p1p2p/3p4/3q1P2/8/7P/q7 b k - 1 35
          //
          let direct_check = (1u64 << rook_stop[x]) & vantage_mask[ROOK] != 0;
          let discovered_check = ((king_start as u8)/8 == (opp_king_sq as u8)/8)
            && ((1u64 << king_start) & waylay_mask != 0);
          let gives_check = ((discovered_check as u8) << 1) | (direct_check as u8);

          let activity = if gives_check > 0 { 1 } else { 0 };
          if activity >= selectivity as u8 {
            late_moves.push(Move {
              movetype:   MoveType::Castle,
              givescheck: gives_check,
              src:        king_start,
              dst:        king_stop[x],
              piece:      Piece::from_usize(player+KING),
              captured:   ZERO_PIECE,
              promotion:  ZERO_PIECE,
              score:      0,
            });
          }
        }
      }

    // End
  }

  pub fn legal_moves(&self, selectivity : Selectivity) -> (Vec<Move>, Vec<Move>)
  {
    let mut early_moves = Vec::with_capacity(16);
    let mut late_moves  = Vec::with_capacity(32);
    self.generate_legal_moves(selectivity, &mut early_moves, &mut late_moves);
    return (early_moves, late_moves);
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
