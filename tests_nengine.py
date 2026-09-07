"""Correctness tests for nengine against python-chess. Run: uv run python tests_nengine.py"""

import random
import sys
import time

import chess
import numpy as np

import nengine as ne

PERFT = [
    (chess.STARTING_FEN, 4, 197281),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89890),
]


def run_perft() -> bool:
    ok = True
    stack = np.zeros((ne.MAX_PLY + 2, ne.POS_LEN), dtype=np.uint64)
    for fen, depth, expected in PERFT:
        pos = ne.pos_from_fen(fen)
        t = time.perf_counter()
        got = ne.perft(pos, depth, stack, 0)
        dt = time.perf_counter() - t
        status = "ok" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"perft {status} d{depth} {got} (expected {expected}) {dt*1000:.0f}ms {fen}")
    return ok


def legal_set(pos: np.ndarray) -> set[str]:
    buf = np.zeros(ne.MAX_MOVES, dtype=np.int64)
    tmp = np.zeros(ne.POS_LEN, dtype=np.uint64)
    n = ne.legal_moves(pos, buf, tmp)
    return {ne.move_to_uci(int(buf[i])) for i in range(n)}


def run_fuzz(games: int, seed: int) -> bool:
    rng = random.Random(seed)
    ok = True
    checked = 0
    for _g in range(games):
        board = chess.Board()
        pos = ne.pos_from_fen(board.fen())
        for _ in range(rng.randint(20, 200)):
            if board.is_game_over():
                break
            ours = legal_set(pos)
            theirs = {m.uci() for m in board.legal_moves}
            if ours != theirs:
                ok = False
                print(f"FAIL movegen at {board.fen()}")
                print(f"  missing {theirs - ours}\n  extra {ours - theirs}")
                break
            if int(pos[ne.I_KEY]) != int(ne.compute_key(pos)):
                ok = False
                print(f"FAIL incremental key at {board.fen()}")
                break
            ref = ne.pos_from_fen(board.fen())
            # Compare bitboards, side, castling, ep (ep only matters if capture possible; fen
            # from python-chess omits ep square when no capture is legal, so compare loosely).
            if not np.array_equal(ref[:14], pos[:14]):
                ok = False
                print(f"FAIL state mismatch at {board.fen()}")
                break
            if ne.in_check(pos) != board.is_check():
                ok = False
                print(f"FAIL in_check at {board.fen()}")
                break
            checked += 1
            m = rng.choice(list(board.legal_moves))
            nm = ne.uci_to_move(pos, m.uci())
            nxt = np.zeros(ne.POS_LEN, dtype=np.uint64)
            assert ne.make_move(pos, nxt, nm)
            pos = nxt
            board.push(m)
        if not ok:
            break
    print(f"fuzz {'ok' if ok else 'FAIL'}: {checked} positions compared over {games} games")
    return ok


def run_eval_symmetry(seed: int) -> bool:
    """Evaluation must be colour-symmetric: mirroring the board flips the sign."""
    rng = random.Random(seed)
    ok = True
    for _ in range(300):
        board = chess.Board()
        for _ in range(rng.randint(0, 60)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        pos = ne.pos_from_fen(board.fen())
        mirrored = ne.pos_from_fen(board.mirror().fen())
        a = ne.evaluate(pos)
        b = ne.evaluate(mirrored)
        if a != b:
            ok = False
            print(f"FAIL eval symmetry {a} vs {b} at {board.fen()}")
            break
    print(f"eval symmetry {'ok' if ok else 'FAIL'}")
    return ok


def run_search_sanity() -> bool:
    eng = ne.Engine()
    ok = True
    # Mate in 1 for white: Qh5xf7#? Use a classic: scholar's mate position.
    tests = [
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),
        ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),  # back rank mate
        ("k7/8/8/8/8/8/8/K6R w - - 0 1", None),  # just must return legal move
    ]
    for fen, expected in tests:
        pos = ne.pos_from_fen(fen)
        eng.new_game()
        eng.record(pos)
        uci, score, depth = eng.think(pos, 1.0)
        b = chess.Board(fen)
        legal = {m.uci() for m in b.legal_moves}
        if uci not in legal or (expected and uci != expected):
            ok = False
            print(f"FAIL search {fen}: got {uci} score {score} d{depth} expected {expected}")
        else:
            print(f"search ok {uci} score {score} depth {depth} nodes {int(eng.ctx[0])}")
    return ok


if __name__ == "__main__" and "--nn" not in sys.argv:
    t0 = time.perf_counter()
    ne.Engine()  # trigger nothing yet; compile happens on first calls below
    results = [run_perft(), run_fuzz(40, 7), run_eval_symmetry(3), run_search_sanity()]
    print(f"total {time.perf_counter() - t0:.1f}s")
    sys.exit(0 if all(results) else 1)


def run_nn_consistency(seed: int) -> bool:
    """Incremental accumulator updates must match a from-scratch refresh, and the numba
    evaluation must match a numpy reference of the same quantised network."""
    rng = np.random.default_rng(seed)
    h = 32
    ft_w = rng.integers(-300, 300, size=(ne.N_BUCKETS * 768, h), dtype=np.int16)
    ft_b = rng.integers(-500, 500, size=h, dtype=np.int16)
    out_w = rng.integers(-200, 200, size=2 * h, dtype=np.int16)
    meta = np.array([255, 64, 400, 1234], dtype=np.int64)
    nn = (ft_w, ft_b, out_w, meta)

    def ref_eval(board: chess.Board) -> int:
        accs = []
        for persp in range(2):
            a = ft_b.astype(np.int64).copy()
            ksq = board.king(chess.WHITE if persp == 0 else chess.BLACK)
            kb = int(ne.KING_BUCKET[ksq ^ (56 * persp)])
            for sq, piece in board.piece_map().items():
                colour = 0 if piece.color == chess.WHITE else 1
                p = piece.piece_type - 1
                f = kb * 768 + (p + 6 * (colour ^ persp)) * 64 + (sq ^ (56 * persp))
                a += ft_w[f]
            accs.append(np.clip(a, 0, 255))
        stm = 0 if board.turn == chess.WHITE else 1
        total = 1234 + int(accs[stm] @ out_w[:h].astype(np.int64)) + int(
            accs[1 - stm] @ out_w[h:].astype(np.int64)
        )
        return max(-3000, min(3000, (total * 400) // (255 * 64)))

    ok = True
    r = random.Random(seed)
    acc = np.zeros((2, 2, h), dtype=np.int32)
    checked = 0
    for _ in range(30):
        board = chess.Board()
        pos = ne.pos_from_fen(board.fen())
        ne.acc_refresh(pos, acc[0], nn)
        for _ in range(120):
            if board.is_game_over():
                break
            got = int(ne.evaluate_nn(pos, acc[0], nn))
            want = ref_eval(board)
            if got != want:
                ok = False
                print(f"FAIL nn eval {got} vs {want} at {board.fen()}")
                break
            m = r.choice(list(board.legal_moves))
            nm = ne.uci_to_move(pos, m.uci())
            nxt = np.zeros(ne.POS_LEN, dtype=np.uint64)
            ne.make_move(pos, nxt, nm)
            ne.acc_apply_move(pos, nxt, nm, acc[0], acc[1], nn)
            fresh = np.zeros((2, h), dtype=np.int32)
            ne.acc_refresh(nxt, fresh, nn)
            if not np.array_equal(fresh, acc[1]):
                ok = False
                print(f"FAIL incremental acc after {m.uci()} at {board.fen()}")
                break
            acc[0] = acc[1]
            pos = nxt
            board.push(m)
            checked += 1
        if not ok:
            break
    print(f"nn consistency {'ok' if ok else 'FAIL'}: {checked} positions")
    return ok


def run_nn_search() -> bool:
    """Search with a random net must still return legal moves and find forced mates."""
    rng = np.random.default_rng(1)
    h = 32
    eng = ne.Engine()
    eng.nn = (
        rng.integers(-50, 50, size=(ne.N_BUCKETS * 768, h), dtype=np.int16),
        rng.integers(-50, 50, size=h, dtype=np.int16),
        rng.integers(-50, 50, size=2 * h, dtype=np.int16),
        np.array([255, 64, 400, 0], dtype=np.int64),
    )
    eng.hidden = h
    eng.acc = np.zeros((ne.MAX_PLY + 2, 2, h), dtype=np.int32)
    eng.use_nn = 1
    ok = True
    for fen, expected in [
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),
        ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),
        (chess.STARTING_FEN, None),
    ]:
        pos = ne.pos_from_fen(fen)
        eng.new_game()
        eng.record(pos)
        uci, score, depth = eng.think(pos, 0.5)
        legal = {m.uci() for m in chess.Board(fen).legal_moves}
        if uci not in legal or (expected and uci != expected):
            ok = False
            print(f"FAIL nn search {fen}: {uci} {score} d{depth}")
        else:
            print(f"nn search ok {uci} score {score} depth {depth} nodes {int(eng.ctx[0])}")
    return ok


if __name__ == "__main__" and "--nn" in sys.argv:
    sys.exit(0 if run_nn_consistency(5) and run_nn_search() else 1)
