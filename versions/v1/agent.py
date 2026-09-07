"""AI Chessathon submission: alpha-beta search over a tapered piece-square evaluation.

Everything here is our own code, written on top of python-chess for move generation.

Structure
    evaluate()   material + piece-square tables, tapered by game phase, side-to-move relative
    quiesce()    captures-only search at the leaves so we never evaluate mid-exchange
    search()     negamax with alpha-beta, transposition table, null-move pruning, killers,
                 history heuristic and late-move reductions
    get_move()   iterative deepening under a wall-clock budget, with a legal fallback at all times

The process lives for one game, so module state (transposition table, game history for
repetition detection) carries across our moves and is discarded with the process.
"""

from __future__ import annotations

import time

import chess

# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------

# Piece values in centipawns: middlegame and endgame.
MG_VALUE = {
    chess.PAWN: 82,
    chess.KNIGHT: 337,
    chess.BISHOP: 365,
    chess.ROOK: 477,
    chess.QUEEN: 1025,
    chess.KING: 0,
}
EG_VALUE = {
    chess.PAWN: 94,
    chess.KNIGHT: 281,
    chess.BISHOP: 297,
    chess.ROOK: 512,
    chess.QUEEN: 936,
    chess.KING: 0,
}

# Phase weights: 24 total at the standard start, 0 with only kings and pawns.
PHASE_WEIGHT = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}

# Piece-square tables, written from White's point of view with rank 8 on the first row so
# they read like a board. They are flipped into a1=0 square order at import.
_PAWN_MG = [
    0, 0, 0, 0, 0, 0, 0, 0,
    60, 70, 60, 70, 65, 60, 40, 30,
    10, 15, 25, 30, 35, 40, 20, 5,
    -5, 5, 5, 20, 22, 10, 5, -10,
    -15, -5, 0, 15, 18, 5, -5, -20,
    -15, -5, -5, -5, 5, 0, 10, -15,
    -20, 0, -15, -20, -15, 15, 20, -20,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_PAWN_EG = [
    0, 0, 0, 0, 0, 0, 0, 0,
    120, 115, 105, 90, 95, 90, 110, 125,
    65, 70, 55, 40, 35, 35, 60, 60,
    20, 15, 5, -5, -5, 0, 10, 15,
    5, 5, -10, -10, -10, -10, 0, 0,
    0, 0, -5, 0, 0, -5, -5, -5,
    5, 5, 5, 5, 8, 0, 0, -5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_KNIGHT_MG = [
    -110, -60, -30, -30, -30, -30, -60, -110,
    -50, -25, 20, 10, 10, 20, -25, -50,
    -30, 20, 25, 40, 40, 25, 20, -30,
    -20, 10, 20, 35, 35, 20, 10, -20,
    -20, 5, 20, 25, 25, 20, 5, -20,
    -30, -5, 10, 15, 15, 10, -5, -30,
    -50, -25, -5, 5, 5, -5, -25, -50,
    -80, -40, -35, -25, -25, -35, -40, -80,
]
_KNIGHT_EG = [
    -60, -40, -15, -30, -30, -15, -40, -60,
    -25, -10, -25, 0, 0, -25, -10, -25,
    -25, -20, 10, 10, 10, 10, -20, -25,
    -20, 5, 20, 20, 20, 20, 5, -20,
    -20, -5, 15, 25, 25, 15, -5, -20,
    -25, -5, 0, 15, 15, 0, -5, -25,
    -40, -20, -10, -5, -5, -10, -20, -40,
    -60, -50, -25, -15, -15, -25, -50, -60,
]
_BISHOP_MG = [
    -30, 0, -60, -40, -25, -40, 5, -10,
    -25, 15, -15, -10, 30, 55, 20, -45,
    -15, 35, 40, 40, 35, 50, 35, 0,
    -5, 5, 20, 50, 35, 35, 5, 0,
    -5, 15, 15, 25, 35, 10, 10, 5,
    0, 15, 15, 15, 15, 25, 20, 10,
    5, 15, 15, 0, 5, 20, 30, 0,
    -35, -5, -15, -20, -15, -10, -40, -20,
]
_BISHOP_EG = [
    -15, -20, -10, -10, -5, -10, -15, -25,
    -10, -5, 5, -10, -5, -10, -5, -15,
    0, -10, 0, 0, 0, 5, 0, 5,
    -5, 10, 10, 10, 15, 10, 5, 0,
    -5, 5, 15, 20, 5, 10, -5, -10,
    -10, -5, 10, 10, 15, 5, -5, -15,
    -15, -20, -5, 0, 5, -10, -15, -25,
    -25, -10, -25, -5, -10, -15, -5, -15,
]
_ROOK_MG = [
    30, 40, 30, 50, 60, 10, 30, 45,
    25, 30, 60, 60, 80, 65, 25, 45,
    -5, 20, 25, 35, 15, 45, 60, 15,
    -25, -10, 5, 25, 25, 35, -10, -20,
    -35, -25, -10, 0, 10, -5, 5, -25,
    -45, -25, -15, -15, 5, 0, -5, -35,
    -45, -15, -20, -10, 0, 10, -5, -70,
    -20, -15, 0, 15, 15, 5, -35, -25,
]
_ROOK_EG = [
    15, 10, 20, 15, 10, 10, 10, 5,
    10, 15, 15, 10, -5, 5, 10, 5,
    5, 5, 5, 5, 5, -5, -5, -5,
    5, 5, 15, 0, 0, 0, 0, 0,
    5, 5, 10, 5, -5, -5, -10, -10,
    -5, 0, -5, 0, -5, -10, -10, -15,
    -5, -5, 0, 0, -10, -10, -10, -5,
    -10, 0, 5, 0, -5, -15, 5, -20,
]
_QUEEN_MG = [
    -30, 0, 30, 10, 60, 45, 45, 45,
    -25, -40, -5, 0, -15, 55, 30, 55,
    -15, -15, 5, 10, 30, 55, 45, 55,
    -25, -25, -15, -15, 0, 15, 0, 0,
    -10, -25, -10, -10, 0, -5, 5, 0,
    -15, 0, -10, 0, -5, 0, 15, 5,
    -35, -10, 10, 0, 10, 15, 0, 0,
    0, -20, -10, 10, -15, -25, -30, -50,
]
_QUEEN_EG = [
    -10, 20, 20, 25, 25, 20, 10, 20,
    -15, 20, 30, 40, 60, 25, 30, 0,
    -20, 5, 10, 50, 45, 35, 20, 10,
    5, 20, 25, 45, 55, 40, 55, 35,
    -20, 30, 20, 45, 30, 35, 40, 25,
    -15, -25, 15, 5, 10, 15, 10, 5,
    -20, -25, -30, -15, -15, -25, -35, -30,
    -35, -30, -20, -45, -5, -30, -20, -40,
]
_KING_MG = [
    -65, 25, 15, -15, -55, -35, 0, 15,
    30, 0, -20, -5, -10, -5, -40, -30,
    -10, 25, 0, -15, -20, 5, 20, -20,
    -15, -20, -10, -25, -30, -25, -15, -35,
    -50, 0, -25, -40, -45, -45, -35, -50,
    -15, -15, -20, -45, -45, -30, -15, -25,
    0, 5, -10, -65, -45, -15, 10, 10,
    -15, 35, 10, -55, 5, -30, 25, 15,
]
_KING_EG = [
    -75, -35, -20, -20, -10, 15, 5, -15,
    -10, 15, 15, 15, 15, 40, 25, 10,
    10, 15, 25, 15, 20, 45, 45, 15,
    -10, 20, 25, 25, 25, 35, 25, 5,
    -20, -5, 20, 25, 25, 25, 10, -10,
    -20, -5, 10, 20, 25, 15, 5, -10,
    -25, -10, 5, 15, 15, 5, -5, -15,
    -55, -35, -20, -10, -30, -15, -25, -45,
]


def _flip(table: list[int]) -> list[int]:
    """Convert a rank-8-first table into a1=0 square indexing for White."""
    out = [0] * 64
    for i, v in enumerate(table):
        rank = 7 - i // 8
        file = i % 8
        out[rank * 8 + file] = v
    return out


# PST[piece_type][color][square] with piece values folded in.
def _build(table: list[int], value: int) -> tuple[list[int], list[int]]:
    white = [v + value for v in _flip(table)]
    black = [white[sq ^ 56] for sq in range(64)]
    return white, black


MG_TABLE: dict[int, tuple[list[int], list[int]]] = {
    chess.PAWN: _build(_PAWN_MG, MG_VALUE[chess.PAWN]),
    chess.KNIGHT: _build(_KNIGHT_MG, MG_VALUE[chess.KNIGHT]),
    chess.BISHOP: _build(_BISHOP_MG, MG_VALUE[chess.BISHOP]),
    chess.ROOK: _build(_ROOK_MG, MG_VALUE[chess.ROOK]),
    chess.QUEEN: _build(_QUEEN_MG, MG_VALUE[chess.QUEEN]),
    chess.KING: _build(_KING_MG, MG_VALUE[chess.KING]),
}
EG_TABLE: dict[int, tuple[list[int], list[int]]] = {
    chess.PAWN: _build(_PAWN_EG, EG_VALUE[chess.PAWN]),
    chess.KNIGHT: _build(_KNIGHT_EG, EG_VALUE[chess.KNIGHT]),
    chess.BISHOP: _build(_BISHOP_EG, EG_VALUE[chess.BISHOP]),
    chess.ROOK: _build(_ROOK_EG, EG_VALUE[chess.ROOK]),
    chess.QUEEN: _build(_QUEEN_EG, EG_VALUE[chess.QUEEN]),
    chess.KING: _build(_KING_EG, EG_VALUE[chess.KING]),
}

BISHOP_PAIR = 30
TEMPO = 10
FILE_MASKS = [chess.BB_FILES[f] for f in range(8)]


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns from the side to move's point of view."""
    mg = 0
    eg = 0
    phase = 0
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]

    for piece_type, bb in (
        (chess.PAWN, board.pawns),
        (chess.KNIGHT, board.knights),
        (chess.BISHOP, board.bishops),
        (chess.ROOK, board.rooks),
        (chess.QUEEN, board.queens),
        (chess.KING, board.kings),
    ):
        mg_w, mg_b = MG_TABLE[piece_type]
        eg_w, eg_b = EG_TABLE[piece_type]
        weight = PHASE_WEIGHT[piece_type]

        pieces = bb & white
        while pieces:
            sq = (pieces & -pieces).bit_length() - 1
            pieces &= pieces - 1
            mg += mg_w[sq]
            eg += eg_w[sq]
            phase += weight

        pieces = bb & black
        while pieces:
            sq = (pieces & -pieces).bit_length() - 1
            pieces &= pieces - 1
            mg -= mg_b[sq]
            eg -= eg_b[sq]
            phase += weight

    # Bishop pair.
    if chess.popcount(board.bishops & white) >= 2:
        mg += BISHOP_PAIR
        eg += BISHOP_PAIR
    if chess.popcount(board.bishops & black) >= 2:
        mg -= BISHOP_PAIR
        eg -= BISHOP_PAIR

    # Pawn structure: doubled and passed pawns.
    wp = board.pawns & white
    bp = board.pawns & black
    for f in range(8):
        wf = wp & FILE_MASKS[f]
        bf = bp & FILE_MASKS[f]
        wc = chess.popcount(wf)
        bc = chess.popcount(bf)
        if wc > 1:
            mg -= 10 * (wc - 1)
            eg -= 20 * (wc - 1)
        if bc > 1:
            mg += 10 * (bc - 1)
            eg += 20 * (bc - 1)

    # Passed pawns, scored by rank.
    pieces = wp
    while pieces:
        sq = (pieces & -pieces).bit_length() - 1
        pieces &= pieces - 1
        if not (_front_span_white(sq) & bp):
            rank = sq >> 3
            mg += PASSED_MG[rank]
            eg += PASSED_EG[rank]
    pieces = bp
    while pieces:
        sq = (pieces & -pieces).bit_length() - 1
        pieces &= pieces - 1
        if not (_front_span_black(sq) & wp):
            rank = 7 - (sq >> 3)
            mg -= PASSED_MG[rank]
            eg -= PASSED_EG[rank]

    if phase > 24:
        phase = 24
    score = (mg * phase + eg * (24 - phase)) // 24
    score = score if board.turn == chess.WHITE else -score
    return score + TEMPO


PASSED_MG = [0, 5, 10, 15, 30, 50, 80, 0]
PASSED_EG = [0, 10, 20, 35, 60, 100, 150, 0]

# Squares in front of a pawn on its own and adjacent files, precomputed.
_FRONT_W: list[int] = []
_FRONT_B: list[int] = []
for _sq in range(64):
    _f = _sq & 7
    _r = _sq >> 3
    _files = 0
    for _df in (-1, 0, 1):
        if 0 <= _f + _df < 8:
            _files |= chess.BB_FILES[_f + _df]
    _ahead_w = 0
    for _rr in range(_r + 1, 8):
        _ahead_w |= chess.BB_RANKS[_rr]
    _ahead_b = 0
    for _rr in range(0, _r):
        _ahead_b |= chess.BB_RANKS[_rr]
    _FRONT_W.append(_files & _ahead_w)
    _FRONT_B.append(_files & _ahead_b)


def _front_span_white(sq: int) -> int:
    return _FRONT_W[sq]


def _front_span_black(sq: int) -> int:
    return _FRONT_B[sq]


# --------------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------------

MATE = 100_000
MATE_BOUND = MATE - 1000
INF = 1_000_000
MAX_PLY = 64

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# Most-valuable-victim / least-valuable-attacker ordering values.
MVV = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 2000,
}


class TimeUp(Exception):
    pass


class Searcher:
    def __init__(self) -> None:
        # key -> (depth, score, flag, move)
        self.tt: dict[int, tuple[int, int, int, chess.Move | None]] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 2)]
        self.history: dict[tuple[bool, int, int], int] = {}
        self.nodes = 0
        self.deadline = 0.0
        self.game_keys: dict[int, int] = {}  # positions seen in the game -> count
        self.path: list[int] = []
        self.abort_check = 0

    # -- helpers ----------------------------------------------------------------------------

    def key(self, board: chess.Board) -> int:
        return hash(board._transposition_key())

    def check_time(self) -> None:
        self.abort_check += 1
        if (self.abort_check & 1023) == 0 and time.perf_counter() >= self.deadline:
            raise TimeUp

    def is_repetition(self, key: int) -> bool:
        # Draw if this position already occurred in the game or earlier on the search path.
        # The referee claims threefold automatically, so we treat a second occurrence as drawn,
        # which is the standard conservative choice inside a search.
        if self.game_keys.get(key, 0) >= 1:
            return True
        return key in self.path

    def mvv_lva(self, board: chess.Board, move: chess.Move) -> int:
        victim = board.piece_type_at(move.to_square)
        if victim is None:
            victim = chess.PAWN  # en passant
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        return MVV[victim] * 10 - MVV[attacker] // 10

    def order_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        tt_move: chess.Move | None,
        ply: int,
    ) -> list[chess.Move]:
        killers = self.killers[ply]
        turn = board.turn
        scored: list[tuple[int, chess.Move]] = []
        for m in moves:
            if m == tt_move:
                s = 10_000_000
            elif board.is_capture(m):
                s = 1_000_000 + self.mvv_lva(board, m)
            elif m.promotion:
                s = 900_000 + (m.promotion or 0) * 1000
            elif m == killers[0]:
                s = 800_000
            elif m == killers[1]:
                s = 700_000
            else:
                s = self.history.get((turn, m.from_square, m.to_square), 0)
            scored.append((s, m))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored]

    # -- quiescence --------------------------------------------------------------------------

    def quiesce(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self.check_time()

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        if ply >= MAX_PLY:
            return stand_pat

        captures = list(board.generate_legal_captures())
        # Include queen promotions that are not captures.
        for m in board.generate_legal_moves(board.pawns, chess.BB_RANK_1 | chess.BB_RANK_8):
            if m.promotion == chess.QUEEN and not board.is_capture(m):
                captures.append(m)
        captures.sort(key=lambda m: self.mvv_lva(board, m), reverse=True)

        for m in captures:
            # Delta pruning: skip captures that cannot raise alpha even with a margin.
            victim = board.piece_type_at(m.to_square) or chess.PAWN
            gain = MVV[victim] + (900 if m.promotion else 0)
            if stand_pat + gain + 200 < alpha:
                continue
            board.push(m)
            score = -self.quiesce(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return alpha

    # -- main search -------------------------------------------------------------------------

    def search(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool,
    ) -> int:
        self.nodes += 1
        self.check_time()

        in_check = board.is_check()
        if in_check:
            depth += 1  # check extension

        if depth <= 0:
            return self.quiesce(board, alpha, beta, ply)

        key = self.key(board)
        if ply > 0:
            if self.is_repetition(key) or board.halfmove_clock >= 100:
                return 0
            if board.is_insufficient_material():
                return 0
            # Mate distance pruning.
            alpha = max(alpha, -MATE + ply)
            beta = min(beta, MATE - ply - 1)
            if alpha >= beta:
                return alpha

        tt_move: chess.Move | None = None
        entry = self.tt.get(key)
        if entry is not None:
            e_depth, e_score, e_flag, e_move = entry
            tt_move = e_move
            if e_depth >= depth and ply > 0:
                if e_flag == TT_EXACT:
                    return e_score
                if e_flag == TT_LOWER and e_score >= beta:
                    return e_score
                if e_flag == TT_UPPER and e_score <= alpha:
                    return e_score

        pv_node = beta - alpha > 1

        # Null-move pruning: if passing still beats beta we can cut.
        if (
            allow_null
            and not pv_node
            and not in_check
            and depth >= 3
            and ply > 0
            and self.has_non_pawn_material(board)
        ):
            static = evaluate(board)
            if static >= beta:
                r = 2 + depth // 4
                board.push(chess.Move.null())
                self.path.append(key)
                try:
                    score = -self.search(board, depth - 1 - r, -beta, -beta + 1, ply + 1, False)
                finally:
                    self.path.pop()
                    board.pop()
                if score >= beta and abs(score) < MATE_BOUND:
                    return beta

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0

        moves = self.order_moves(board, moves, tt_move, ply)

        best_score = -INF
        best_move: chess.Move | None = None
        orig_alpha = alpha
        self.path.append(key)
        try:
            for i, m in enumerate(moves):
                is_capture = board.is_capture(m)
                gives_check = board.gives_check(m)
                board.push(m)

                # Late move reductions on quiet, late moves at sufficient depth.
                reduce = 0
                if (
                    depth >= 3
                    and i >= 3
                    and not is_capture
                    and not gives_check
                    and not in_check
                    and not m.promotion
                ):
                    reduce = 1 if i < 8 else 2

                if i == 0:
                    score = -self.search(board, depth - 1, -beta, -alpha, ply + 1, True)
                else:
                    score = -self.search(
                        board, depth - 1 - reduce, -alpha - 1, -alpha, ply + 1, True
                    )
                    if score > alpha and (reduce > 0 or score < beta):
                        score = -self.search(board, depth - 1, -beta, -alpha, ply + 1, True)
                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = m
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    if not is_capture:
                        k = self.killers[ply]
                        if k[0] != m:
                            k[1] = k[0]
                            k[0] = m
                        hk = (board.turn, m.from_square, m.to_square)
                        self.history[hk] = self.history.get(hk, 0) + depth * depth
                    break
        finally:
            self.path.pop()

        if best_score <= orig_alpha:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        self.tt[key] = (depth, best_score, flag, best_move)
        return best_score

    def has_non_pawn_material(self, board: chess.Board) -> bool:
        us = board.occupied_co[board.turn]
        return bool(us & (board.knights | board.bishops | board.rooks | board.queens))

    # -- root --------------------------------------------------------------------------------

    def best_move(self, board: chess.Board, budget_s: float) -> chess.Move:
        start = time.perf_counter()
        self.deadline = start + budget_s
        self.nodes = 0
        self.abort_check = 0
        self.path = []
        for k in self.killers:
            k[0] = k[1] = None
        # Decay history so stale ordering does not dominate.
        for hk in list(self.history):
            self.history[hk] //= 8

        moves = list(board.legal_moves)
        if len(moves) == 1:
            return moves[0]

        root_key = self.key(board)
        best = moves[0]
        best_score = 0
        depth = 1
        try:
            while depth <= MAX_PLY:
                # Root: order by previous best first.
                tt_entry = self.tt.get(root_key)
                tt_move = tt_entry[3] if tt_entry else best
                ordered = self.order_moves(board, moves, tt_move, 0)

                alpha = -INF
                beta = INF
                iter_best = ordered[0]
                iter_score = -INF
                self.path = [root_key]
                for i, m in enumerate(ordered):
                    board.push(m)
                    if i == 0:
                        score = -self.search(board, depth - 1, -beta, -alpha, 1, True)
                    else:
                        score = -self.search(board, depth - 1, -alpha - 1, -alpha, 1, True)
                        if score > alpha:
                            score = -self.search(board, depth - 1, -beta, -alpha, 1, True)
                    board.pop()
                    if score > iter_score:
                        iter_score = score
                        iter_best = m
                    if score > alpha:
                        alpha = score
                self.path = []

                best = iter_best
                best_score = iter_score
                self.tt[root_key] = (depth, best_score, TT_EXACT, best)

                elapsed = time.perf_counter() - start
                print(
                    f"depth {depth} score {best_score} nodes {self.nodes} "
                    f"time {elapsed * 1000:.0f}ms best {best.uci()}"
                )
                if abs(best_score) >= MATE_BOUND:
                    break
                # Do not start an iteration we are unlikely to finish.
                if elapsed > budget_s * 0.4:
                    break
                depth += 1
        except TimeUp:
            print(f"time up at depth {depth}, nodes {self.nodes}")
        return best


# --------------------------------------------------------------------------------------------
# Game state and time management
# --------------------------------------------------------------------------------------------

SEARCHER = Searcher()
MOVES_PLAYED = 0

# Safety margin against the watchdog. Wall time includes process scheduling and python-chess
# overhead outside our own timer, and the match core is slower than a dev box.
SAFETY_MS = 150


def budget_ms(time_left_ms: int, board: chess.Board) -> float:
    """Milliseconds to spend on this move."""
    if time_left_ms <= 300:
        return 20.0
    if time_left_ms <= 2000:
        return max(20.0, time_left_ms * 0.05)

    # Expect roughly 35 more of our moves early on, fewer as material disappears.
    material = chess.popcount(board.occupied) - 2
    expected_moves = 18 + material  # 48 at the start, shrinking as pieces leave
    expected_moves = max(20, min(50, expected_moves))
    increment = 500
    budget = time_left_ms / expected_moves + increment * 0.8
    # Never spend more than a fifth of the clock on one move.
    budget = min(budget, time_left_ms * 0.2)
    return max(20.0, budget - SAFETY_MS)


def record_position(board: chess.Board) -> None:
    k = SEARCHER.key(board)
    SEARCHER.game_keys[k] = SEARCHER.game_keys.get(k, 0) + 1


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation."""
    global MOVES_PLAYED
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        # Should never happen: the referee ends the game first.
        return "0000"
    fallback = legal[0]

    try:
        record_position(board)
        ms = budget_ms(time_left_ms, board)
        move = SEARCHER.best_move(board, ms / 1000.0)
        if move not in legal:
            move = fallback
        board.push(move)
        record_position(board)
        MOVES_PLAYED += 1
        print(f"move {MOVES_PLAYED} clock {time_left_ms}ms budget {ms:.0f}ms -> {move.uci()}")
        return move.uci()
    except Exception as exc:
        print(f"search failed: {exc!r}, falling back to {fallback.uci()}")
        return fallback.uci()


# Warm-up at import so the first move pays no first-call costs.
_warm = chess.Board()
SEARCHER.best_move(_warm, 0.3)
SEARCHER.tt.clear()
SEARCHER.game_keys.clear()
