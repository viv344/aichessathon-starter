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
