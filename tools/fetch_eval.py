"""Stream the Lichess evaluation database into a compact fen,cp file of quiet positions.

Usage: zstd -d -c data/eval_slice.zst | python tools/fetch_eval.py data/eval_q.csv

Keeps the deepest evaluation per position. Positions whose best move is a capture or a
promotion are dropped: a static evaluator should be trained on positions where nothing is
hanging, since the search's quiescence handles the rest. Mate scores clamp to +-3000 cp.
Training data only: nothing from this file ships in the submission.
"""

import json
import sys

FILES = "abcdefgh"


def occupied(board_fen: str) -> set[int]:
    squares = set()
    for r, row in enumerate(board_fen.split("/")):
        rank = 7 - r
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                squares.add(rank * 8 + f)
                f += 1
    return squares


def main() -> None:
    n = kept = 0
    with open(sys.argv[1], "w") as out:
        for line in sys.stdin:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            evals = rec.get("evals")
            if not evals:
                continue
            best = max(evals, key=lambda e: e.get("depth", 0))
            if best.get("depth", 0) < 12:
                continue
            pvs = best.get("pvs")
            if not pvs:
                continue
            pv = pvs[0]
            if "cp" in pv:
                cp = int(pv["cp"])
            elif "mate" in pv:
                cp = 3000 if pv["mate"] > 0 else -3000
            else:
                continue
            n += 1
            first = pv.get("line", "").split(" ")[0]
            if len(first) != 4:
                continue  # promotion or malformed
            fen = rec["fen"]
            to_sq = FILES.index(first[2]) + 8 * (int(first[3]) - 1)
            occ = occupied(fen.split(" ")[0])
            if to_sq in occ:
                continue  # capture
            # En passant capture: pawn moving diagonally to an empty square.
            from_sq = FILES.index(first[0]) + 8 * (int(first[1]) - 1)
            if (from_sq & 7) != (to_sq & 7) and abs(from_sq - to_sq) in (7, 9):
                board = fen.split(" ")[0]
                # Cheap check that the mover is a pawn: look at the square's piece letter.
                rows = board.split("/")
                rank = 7 - (from_sq >> 3)
                f = 0
                piece = ""
                for ch in rows[rank]:
                    if ch.isdigit():
                        f += int(ch)
                    else:
                        if f == (from_sq & 7):
                            piece = ch
                        f += 1
                if piece in ("P", "p"):
                    continue
            cp = max(-3000, min(3000, cp))
            out.write(f"{fen},{cp}\n")
            kept += 1
            if kept % 1_000_000 == 0:
                print(f"{kept} kept of {n}", file=sys.stderr, flush=True)
    print(f"done: kept {kept} of {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
