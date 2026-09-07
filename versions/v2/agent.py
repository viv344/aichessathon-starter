"""AI Chessathon submission entrypoint.

The search lives in nengine.py, a bitboard engine compiled with numba. This file owns time
management, game history for repetition detection, and validation of the engine's answer
with python-chess. If the compiled engine ever fails, fallback.py (our earlier pure-Python
search) plays the move instead, so a bug costs strength rather than the game.
"""

from __future__ import annotations

import time

import chess

import fallback
import nengine

# Safety margin against the watchdog. Wall time includes process scheduling and any work
# outside our own timer, and the match core is slower than a dev box.
SAFETY_MS = 150

ENGINE = nengine.Engine()
MOVES_PLAYED = 0


def budget_ms(time_left_ms: int, board: chess.Board) -> float:
    """Milliseconds to spend on this move."""
    if time_left_ms <= 300:
        return 20.0
    if time_left_ms <= 2000:
        return max(20.0, time_left_ms * 0.05)
    # Expect roughly 48 more of our moves at the start, fewer as material disappears.
    material = chess.popcount(board.occupied) - 2
    expected_moves = max(20, min(50, 18 + material))
    increment = 500
    budget = time_left_ms / expected_moves + increment * 0.8
    budget = min(budget, time_left_ms * 0.2)
    return max(20.0, budget - SAFETY_MS)


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation."""
    global MOVES_PLAYED
    start = time.perf_counter()
    board = chess.Board(fen)
    legal = {m.uci() for m in board.legal_moves}
    if not legal:
        return "0000"

    ms = budget_ms(time_left_ms, board)
    uci = ""
    try:
        pos = nengine.pos_from_fen(fen)
        ENGINE.record(pos)
        ENGINE.log = []
        uci, _score, _depth = ENGINE.think(pos, ms / 1000.0)
        for line in ENGINE.log[-3:]:
            print(line)
        if uci not in legal:
            print(f"engine returned illegal move {uci}, using fallback")
            uci = ""
        else:
            # Record the position after our move so repetition detection sees every ply.
            nxt = pos.copy()
            nengine.make_move(pos, nxt, nengine.uci_to_move(pos, uci))
            ENGINE.record(nxt)
    except Exception as exc:
        print(f"engine failed: {exc!r}, using fallback")
        uci = ""

    if not uci:
        try:
            remaining = ms - (time.perf_counter() - start) * 1000
            fb = fallback.SEARCHER.best_move(board, max(0.02, remaining / 1000.0 * 0.5))
            uci = fb.uci() if fb.uci() in legal else sorted(legal)[0]
        except Exception as exc:
            print(f"fallback failed: {exc!r}")
            uci = sorted(legal)[0]

    MOVES_PLAYED += 1
    elapsed = (time.perf_counter() - start) * 1000
    print(
        f"move {MOVES_PLAYED} clock {time_left_ms}ms budget {ms:.0f}ms "
        f"used {elapsed:.0f}ms -> {uci}"
    )
    return uci


# Warm-up at import: compiles every numba function on the paths a real move uses, so the
# compile cost lands inside the 90 s init budget instead of on the clock.
_t0 = time.perf_counter()
_warm = nengine.pos_from_fen(chess.STARTING_FEN)
ENGINE.record(_warm)
ENGINE.think(_warm, 0.5)
ENGINE.new_game()
print(f"init {time.perf_counter() - _t0:.1f}s")
