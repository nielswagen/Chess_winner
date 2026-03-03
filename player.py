from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, List, Tuple

import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_tournament.players import Player


@dataclass(frozen=True)
class SearchConfig:
    top_k: int = 4          # prune candidates for speed
    opp_top_k: int = 4      # prune opponent replies
    depth: int = 2          # keep at 2 for big gain, manageable compute
    max_legal: int = 80     # safety cap (rare positions can have many moves)


class TransformerPlayer(Player):
    """
    Transformer-based chess player that selects from legal moves by reranking
    candidate UCI strings using transformer log-probabilities, with depth-2 minimax.
    """

    def __init__(
        self,
        name: str,
        model_id: str = "distilgpt2",
        temperature: float = 0.0,
    ):
        # IMPORTANT: only `name` is required by the autograder;
        # all other args must have defaults.
        super().__init__(name)

        self.model_id = model_id
        self.temperature = temperature  # not used in scoring, but kept for extensibility
        self.cfg = SearchConfig()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    def get_move(self, fen: str) -> Optional[str]:
        """
        Returns a legal move in UCI format, or None if no legal moves exist.
        Must never crash (autograder + tournament).
        """
        try:
            board = chess.Board(fen)
            legal = list(board.legal_moves)
            if not legal:
                return None

            # safety cap in extremely branching positions
            if len(legal) > self.cfg.max_legal:
                legal = legal[: self.cfg.max_legal]

            # main decision
            move_uci = self._choose_depth2(board)
            if move_uci is not None:
                return move_uci

            # fallback: best by 1-ply rerank
            best = self._best_move_by_rerank(board)
            return best.uci()

        except Exception:
            # absolute last resort: return some legal move if possible
            try:
                board = chess.Board(fen)
                legal = list(board.legal_moves)
                return legal[0].uci() if legal else None
            except Exception:
                return None

    # -------------------------
    # Decision logic (depth-2)
    # -------------------------

    def _choose_depth2(self, board: chess.Board) -> Optional[str]:
        """
        Depth-2 minimax: choose move that maximizes worst-case reply.
        Uses transformer reranking to evaluate positions and prune.
        """
        if self.cfg.depth <= 1:
            return self._best_move_by_rerank(board).uci()

        my_ranked = self._top_moves_by_rerank(board, self.cfg.top_k)
        if not my_ranked:
            return None

        best_move: chess.Move = my_ranked[0][0]
        best_value: float = float("-inf")

        for my_move, _my_score in my_ranked:
            board.push(my_move)

            # If game ended, prefer it
            if board.is_checkmate():
                board.pop()
                return my_move.uci()
            if board.is_stalemate() or board.is_insufficient_material():
                # neutral-ish outcome
                value = 0.0
                board.pop()
                if value > best_value:
                    best_value, best_move = value, my_move
                continue

            # Opponent chooses reply that is worst for us
            opp_ranked = self._top_moves_by_rerank(board, self.cfg.opp_top_k)

            if not opp_ranked:
                # no replies => terminal
                value = self._terminal_value(board)
                board.pop()
                if value > best_value:
                    best_value, best_move = value, my_move
                continue

            worst_for_us = float("inf")

            for opp_move, _opp_score in opp_ranked:
                board.push(opp_move)

                # Avoid allowing mate-in-1 patterns (cheap but effective)
                mate_penalty = self._opponent_has_mate_in_1(board)
                if mate_penalty:
                    pos_value = float("-1e9")
                else:
                    pos_value = self._material_eval(board)

                worst_for_us = min(worst_for_us, pos_value)
                board.pop()

            board.pop()

            if worst_for_us > best_value:
                best_value = worst_for_us
                best_move = my_move

        return best_move.uci()

    def _terminal_value(self, board: chess.Board) -> float:
        # If no legal moves: checkmate or stalemate
        if board.is_checkmate():
            # side to move is checkmated => bad for side to move
            return float("-1e9")
        return 0.0

    def _opponent_has_mate_in_1(self, board: chess.Board) -> bool:
        """
        After both moves have been played, check if side-to-move has a checkmate in 1.
        This is a fast blunder detector.
        """
        for mv in board.legal_moves:
            board.push(mv)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return True
        return False

    # -------------------------
    # Transformer reranking
    # -------------------------

    def _best_move_by_rerank(self, board: chess.Board) -> chess.Move:
        ranked = self._top_moves_by_rerank(board, top_k=1)
        return ranked[0][0]

    def _material_eval(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1e9

        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        score = 0.0
        for piece_type, val in values.items():
            score += len(board.pieces(piece_type, board.turn)) * val
            score -= len(board.pieces(piece_type, not board.turn)) * val

        return score

    def _top_moves_by_rerank(self, board: chess.Board, top_k: int) -> List[Tuple[chess.Move, float]]:
        legal = list(board.legal_moves)
        if not legal:
            return []

        prompt = self._make_prompt(board.fen())
        scored: List[Tuple[chess.Move, float]] = []

        for mv in legal:
            uci = mv.uci()
            # score probability of outputting " <uci>" next
            s = self._score_completion(prompt, " " + uci)
            scored.append((mv, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, top_k)]

    def _make_prompt(self, fen: str) -> str:
        # Stable, short prompt = more consistent scoring, faster inference
        return f"FEN: {fen}\nBest move (UCI):"

    @lru_cache(maxsize=200_000)
    def _score_completion(self, prompt: str, completion: str) -> float:
        """
        Log-probability of `completion` tokens given `prompt`.
        Cached heavily for speed in search.
        """
        with torch.no_grad():
            # Tokenize prompt and prompt+completion
            p = self.tokenizer(prompt, return_tensors="pt")
            pc = self.tokenizer(prompt + completion, return_tensors="pt")

            p_ids = p["input_ids"].to(self.device)
            pc_ids = pc["input_ids"].to(self.device)

            p_len = int(p_ids.shape[1])
            full = pc_ids  # [1, T]

            out = self.model(full)
            logits = out.logits  # [1, T, V]
            log_probs = torch.log_softmax(logits, dim=-1)

            total = 0.0
            # Score tokens in completion span: indices [p_len .. T-1]
            for idx in range(p_len, full.shape[1]):
                token_id = int(full[0, idx].item())
                pred_pos = idx - 1
                total += float(log_probs[0, pred_pos, token_id].item())

            return total