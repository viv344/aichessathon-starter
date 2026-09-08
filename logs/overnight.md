# Overnight report (night of 7 to 8 September)

Live upload: v2 (numba engine, hand evaluation). Nothing uploaded overnight.

## Results so far

| Test | Games | Control | Result |
|---|---|---|---|
| 768-feature net (first try) vs v2 | 48 | 10s+0.1 | 24.0%, lost badly (noisy eval, side-to-move bias) |
| king-bucket 256 net vs v2 | 64 | 10s+0.1 | 54.7% +- 12 |
| king-bucket 256 net vs v2 | 160 | 20s+0.2 | 50.3% +- 7.7, equal |
| new search (hand eval) vs v2 | 100 | 10s+0.1 | pending |
| king-bucket 512 net + new search vs v2 | 100 | 10s+0.1 | pending |

Compile time (clean, local): v2 9.1 s, neural build 15.2 s. Platform ran v2 in 24 to 35 s,
so the neural build projects to about 40 to 60 s of the 90 s init budget.

## In progress
- Training 512-hidden net on all 45M quiet positions, 10 epochs (weights/nnue_kb512_full.npz).

## Update, morning of 8 September

- Both 100-game arenas were killed overnight by memory pressure from the 45M-position encoding.
  Partial results at 72 games: new search (hand eval) vs v2 +16 =37 -19 with an abnormal 51%
  draw rate; 512 net + new search vs v2 +28 =3 -41. Both confounded by the new search.
- Suspect in the new search: quiescence stores depth-0 exact entries that overwrite deeper
  transposition-table entries for the same position. Fix pending: never let a shallower entry
  replace a deeper one for the same key.
- Full-data 512 net finished: weights/nnue_kb512_full.npz, validation loss 0.0111 (best so far).
  Candidate cand/kb512f_old pairs it with the pre-change search for a clean evaluation test.
- Jobs stopped at 06:37 for travel. Resume: fix TT replacement, arena "." vs versions/v2,
  arena cand/kb512f_old vs versions/v2 (60+ games each), then package the winner.

## 8 September, daytime results (all vs versions/v2)

| Candidate | Games | Control | Result |
|---|---|---|---|
| new search + TT fix, hand eval | 60 | 10s+0.1 | 56.7% +- 12.5 (24 threefolds: verified legitimate, engines share eval) |
| full-data 512 net, old search | 60 | 10s+0.1 | 55.0% +- 12.6 |
| combo: new search + 512 net | 60 | 10s+0.1 | 49.2% +- 12.6 |
| combo | 40 | 60s+0.5 | 58.8% +- 15.3 |
| new search, hand eval | 300 | 10s+0.1 | running |
| combo | 100 | 60s+0.5 | running |

Lesson: 60-game runs cannot resolve 30-50 Elo. Use 300 games / 10 workers for fast screening.
The net looks more valuable at longer control, where depth is higher and eval quality matters more.
dev/ holds int16 accumulators + skip static eval in check (tests pass, speed not yet measured idle).
