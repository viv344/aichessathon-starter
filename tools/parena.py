"""Parallel arena: many games between two agent directories, spread over several processes.

Openings come from a text file of FENs (one per line) so tests are not tuned to the eight
sample openings. Each opening is played twice, once with each colour.

    uv run python tools/parena.py --agent . --opponent versions/v2 --games 64 --workers 8 \
        --base-ms 10000 --increment-ms 100 --openings data/openings.txt
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.referee import play_match
from harness.rules import OPENINGS, PLY_CAP
from harness.sandbox import local


def play_one(args: tuple[int, str, str, str, int, int, int]) -> tuple[int, bool, str, str]:
    index, fen, agent, opponent, base_ms, inc_ms, ply_cap = args
    plays_white = index % 2 == 0
    white, black = (agent, opponent) if plays_white else (opponent, agent)
    outcome = play_match(
        local(Path(white), index), local(Path(black), index), base_ms, inc_ms,
        ply_cap=ply_cap, start_fen=fen,
    )
    return index, plays_white, outcome.result, outcome.termination


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=".")
    ap.add_argument("--opponent", default="versions/v2")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-ms", type=int, default=10_000)
    ap.add_argument("--increment-ms", type=int, default=100)
    ap.add_argument("--ply-cap", type=int, default=PLY_CAP)
    ap.add_argument("--openings", default="data/openings.txt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if Path(a.openings).exists():
        with open(a.openings) as fh:
            fens = [ln.strip() for ln in fh if ln.strip()]
    else:
        fens = [fen for _, fen in OPENINGS]
    rng = random.Random(a.seed)
    rng.shuffle(fens)
    n = a.games - a.games % 2
    jobs = []
    for i in range(n):
        fen = fens[(i // 2) % len(fens)]
        jobs.append((i, fen, str(Path(a.agent).resolve()), str(Path(a.opponent).resolve()),
                     a.base_ms, a.increment_ms, a.ply_cap))

    wins = draws = losses = 0
    terms: dict[str, int] = {}
    fails: list[str] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(play_one, j) for j in jobs]
        for fut in as_completed(futures):
            index, plays_white, result, term = fut.result()
            terms[term] = terms.get(term, 0) + 1
            if result == "draw":
                draws += 1
            elif result == "void":
                continue
            elif (result == "white") == plays_white:
                wins += 1
            else:
                losses += 1
                if term in ("illegal", "crash", "flag", "init"):
                    fails.append(f"game {index} lost by {term}")
            done = wins + draws + losses
            if done % 8 == 0 or done == n:
                print(f"  {done}/{n}: +{wins} ={draws} -{losses}", flush=True)

    played = wins + draws + losses
    if played == 0:
        print("no games")
        return
    score = (wins + 0.5 * draws) / played
    p = score
    interval = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / played)
    elo = -400 * math.log10(1 / max(min(score, 0.999), 0.001) - 1)
    print(f"{a.agent} vs {a.opponent}: +{wins} ={draws} -{losses}, "
          f"score {score * 100:.1f}% +- {interval * 100:.1f}%, elo {elo:+.0f}")
    print("terminations", terms)
    for f in fails:
        print("  ", f)


if __name__ == "__main__":
    main()
