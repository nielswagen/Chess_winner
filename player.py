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
    top_k: int = 5
    opp_top_k: int = 5
    depth: int = 2
    endgame_depth: int = 3
    endgame_threshold: int = 10
    max_legal: int = 80


FEW_SHOT_EXAMPLES = (
    "FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\n"
    "Best move (UCI): e7e5\n"
    "FEN: rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2\n"
    "Best move (UCI): g1f3\n"
    "FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\n"
    "Best move (UCI): f1c4\n"
    "FEN: r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/RNBQK2R w KQkq - 4 4\n"
    "Best move (UCI): e1g1\n"
    "FEN: r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 6 5\n"
    "Best move (UCI): d2d3\n"
)


class TransformerPlayer(Player):
    """
    Transformer-based chess player using:
    - Few-shot prompted LM reranking of legal moves
    - Depth-2 minimax (depth-3 in endgame)
    - Material + positional evaluation (center, mobility, king safety)
    - Mate-in-1 blunder detector
    - LRU cache for speed in minimax search
    """

    def __init__(
        self,
        name: str,
        model_id: str = "donquichot/smollm2-chess",
        temperature: float = 0.0,
    ):
        super().__init__(name)
        self.model_id = model_id
        self.temperature = temperature
        self.cfg = SearchConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{name}] Loading {model_id} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    def get_move(self, fen: str) -> Optional[str]:
        try:
            board = chess.Board(fen)
            legal = list(board.legal_moves)
            if not legal:
                return None
            if len(legal) > self.cfg.max_legal:
                legal = legal[: self.cfg.max_legal]
            move_uci = self._choose(board)
            if move_uci is not None:
                return move_uci
            return self._best_move_by_rerank(board).uci()
        except Exception:
            try:
                board = chess.Board(fen)
                legal = list(board.legal_moves)
                return legal[0].uci() if legal else None
            except Exception:
                return None

    def _get_depth(self, board: chess.Board) -> int:
        if len(board.piece_map()) <= self.cfg.endgame_threshold:
            return self.cfg.endgame_depth
        return self.cfg.depth

    def _choose(self, board: chess.Board) -> Optional[str]:
        depth = self._get_depth(board)
        if depth <= 1:
            return self._best_move_by_rerank(board).uci()
        return self._minimax(board, depth)

    def _minimax(self, board: chess.Board, depth: int) -> Optional[str]:
        my_ranked = self._top_moves_by_rerank(board, self.cfg.top_k)
        if not my_ranked:
            return None

        best_move: chess.Move = my_ranked[0][0]
        best_value: float = float("-inf")

        for my_move, _ in my_ranked:
            board.push(my_move)

            if board.is_checkmate():
                board.pop()
                return my_move.uci()

            if board.is_stalemate() or board.is_insufficient_material():
                value = 0.0
                board.pop()
                if value > best_value:
                    best_value, best_move = value, my_move
                continue

            opp_ranked = self._top_moves_by_rerank(board, self.cfg.opp_top_k)
            worst_for_us = self._opponent_min(board, opp_ranked)
            board.pop()

            if worst_for_us > best_value:
                best_value = worst_for_us
                best_move = my_move

        return best_move.uci()

    def _opponent_min(self, board: chess.Board, opp_ranked: List[Tuple[chess.Move, float]]) -> float:
        if not opp_ranked:
            return self._terminal_value(board)
        worst_for_us = float("inf")
        for opp_move, _ in opp_ranked:
            board.push(opp_move)
            if self._opponent_has_mate_in_1(board):
                pos_value = float("-1e9")
            else:
                pos_value = self._material_eval(board)
            worst_for_us = min(worst_for_us, pos_value)
            board.pop()
        return worst_for_us

    def _terminal_value(self, board: chess.Board) -> float:
        return float("-1e9") if board.is_checkmate() else 0.0

    def _opponent_has_mate_in_1(self, board: chess.Board) -> bool:
        for mv in board.legal_moves:
            board.push(mv)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return True
        return False

    def _material_eval(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1e9

        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3.2,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        score = 0.0

        for piece_type, val in piece_values.items():
            score += len(board.pieces(piece_type, board.turn)) * val
            score -= len(board.pieces(piece_type, not board.turn)) * val

        for sq in [chess.E4, chess.D4, chess.E5, chess.D5]:
            piece = board.piece_at(sq)
            if piece:
                score += 0.3 if piece.color == board.turn else -0.3

        score += len(list(board.legal_moves)) * 0.05

        if len(board.piece_map()) > self.cfg.endgame_threshold:
            our_king = board.king(board.turn)
            their_king = board.king(not board.turn)
            if our_king is not None:
                score -= bin(int(board.attacks_mask(our_king))).count("1") * 0.05
            if their_king is not None:
                score += bin(int(board.attacks_mask(their_king))).count("1") * 0.05

        return score

    def _best_move_by_rerank(self, board: chess.Board) -> chess.Move:
        return self._top_moves_by_rerank(board, top_k=1)[0][0]

    def _top_moves_by_rerank(self, board: chess.Board, top_k: int) -> List[Tuple[chess.Move, float]]:
        legal = list(board.legal_moves)
        if not legal:
            return []
        prompt = self._make_prompt(board.fen())
        scored = [(mv, self._score_completion(prompt, " " + mv.uci())) for mv in legal]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(1, top_k)]

    def _make_prompt(self, fen: str) -> str:
        # Must match the format used in train.py
        return f"{FEW_SHOT_EXAMPLES}FEN: {fen}\nBest move (UCI):"

    @lru_cache(maxsize=200_000)
    def _score_completion(self, prompt: str, completion: str) -> float:
        with torch.no_grad():
            p  = self.tokenizer(prompt,              return_tensors="pt")
            pc = self.tokenizer(prompt + completion, return_tensors="pt")
            p_len = int(p["input_ids"].shape[1])
            full  = pc["input_ids"].to(self.device)
            logits    = self.model(full).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            total = 0.0
            for idx in range(p_len, full.shape[1]):
                token_id = int(full[0, idx].item())
                total += float(log_probs[0, idx - 1, token_id].item())
            return total