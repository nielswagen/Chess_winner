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

# Shared piece values — used in both _material_eval and tactical override
PIECE_VALUES = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3.2,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0,
}


class TransformerPlayer(Player):

    def __init__(self, name: str, model_id: str = "nlpguy/smolchess", temperature: float = 0.0):
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
        # UPGRADE 2: Fast O(N) pre-check — win material or mate before minimax
        quick_win = self._winning_capture_or_mate(board)
        if quick_win is not None:
            return quick_win
        if depth <= 1:
            return self._best_move_by_rerank(board).uci()
        return self._minimax(board, depth)

    def _winning_capture_or_mate(self, board: chess.Board) -> Optional[str]:
        """O(N) scan: return a move that mates or wins material outright."""
        for move in board.legal_moves:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return move.uci()
            if board.is_capture(move):
                victim   = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim is None:
                    continue  # en passant — pawn takes pawn
                if attacker is not None:
                    if PIECE_VALUES.get(victim.piece_type, 0) > PIECE_VALUES.get(attacker.piece_type, 0):
                        return move.uci()
        return None

    def _minimax(self, board: chess.Board, depth: int) -> Optional[str]:
        # UPGRADE 4: Widen search in endgame (top_k 5->6)
        is_endgame = len(board.piece_map()) <= self.cfg.endgame_threshold
        top_k     = self.cfg.top_k     + (1 if is_endgame else 0)
        opp_top_k = self.cfg.opp_top_k + (1 if is_endgame else 0)

        my_ranked = self._top_moves_by_rerank(board, top_k)
        if not my_ranked:
            return None

        best_move:  chess.Move = my_ranked[0][0]
        best_value: float      = float("-inf")

        for my_move, my_neural_score in my_ranked:
            board.push(my_move)
            if board.is_checkmate():
                board.pop()
                return my_move.uci()
            if board.is_stalemate() or board.is_insufficient_material():
                board.pop()
                if 0.0 > best_value:
                    best_value, best_move = 0.0, my_move
                continue
            opp_ranked   = self._top_moves_by_rerank(board, opp_top_k)
            worst_for_us = self._opponent_min(board, opp_ranked, my_neural_score)
            board.pop()
            if worst_for_us > best_value:
                best_value, best_move = worst_for_us, my_move

        return best_move.uci()

    def _opponent_min(self, board, opp_ranked, parent_neural_score: float) -> float:
        if not opp_ranked:
            return self._terminal_value(board)
        worst_for_us = float("inf")
        for opp_move, opp_neural_score in opp_ranked:
            board.push(opp_move)
            if self._opponent_has_mate_in_1(board):
                pos_value = float("-1e9")
            else:
                # UPGRADE 3: leaf = material + 0.15 * neural_prior (no extra model call)
                pos_value = self._material_eval(board) + 0.15 * opp_neural_score
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
        score = 0.0
        for piece_type, val in PIECE_VALUES.items():
            if piece_type == chess.KING:
                continue
            score += len(board.pieces(piece_type, board.turn))     * val
            score -= len(board.pieces(piece_type, not board.turn)) * val
        for sq in [chess.E4, chess.D4, chess.E5, chess.D5]:
            piece = board.piece_at(sq)
            if piece:
                score += 0.3 if piece.color == board.turn else -0.3
        score += len(list(board.legal_moves)) * 0.05
        if len(board.piece_map()) > self.cfg.endgame_threshold:
            our_king   = board.king(board.turn)
            their_king = board.king(not board.turn)
            if our_king is not None:
                score -= bin(int(board.attacks_mask(our_king))).count("1")   * 0.05
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
        move_ucis = [mv.uci() for mv in legal]

        # Score alle moves in één batch
        raw_scores = self._score_moves_batch(prompt, move_ucis)

        # Bonussen toepassen
        scored = []
        for mv, score in zip(legal, raw_scores):
            bonus = 0.0
            if board.is_capture(mv):
                bonus += 0.5
            board.push(mv)
            if board.is_check():
                bonus += 0.3
            board.pop()
            if mv.promotion is not None:
                bonus += 0.7
            scored.append((mv, score + bonus))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max(1, top_k)]

    def _make_prompt(self, fen: str) -> str:
        return f"{FEW_SHOT_EXAMPLES}FEN: {fen}\nBest move (UCI):"


    def _score_moves_batch(self, prompt: str, moves: List[str]) -> List[float]:
        # Left-padding for causal LM correctness
        self.tokenizer.padding_side = "left"

        texts = [prompt + " " + m for m in moves]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Compute p_len from tokenizer only (no extra model call)
        p_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

        with torch.no_grad():
            logits = self.model(**inputs).logits  # (B, T, V)

        log_probs = torch.log_softmax(logits, dim=-1)  # (B, T, V)

        # Vectorized gather instead of Python loop
        # log_probs[:, :-1] are predictions, input_ids[:, 1:] are targets
        token_ids = inputs["input_ids"][:, 1:].unsqueeze(-1)  # (B, T-1, 1)
        gathered = log_probs[:, :-1].gather(-1, token_ids).squeeze(-1)  # (B, T-1)

        # Mask: only score completion tokens (after prompt), ignore padding
        mask = inputs["attention_mask"][:, 1:].float()  # (B, T-1)

        # Zero out prompt tokens
        seq_len = gathered.shape[1]
        for i in range(min(p_len, seq_len)):
            mask[:, i] = 0.0

        scores = (gathered * mask).sum(dim=-1)  # (B,)
        return scores.tolist()