"""Train a small perspective network (NNUE-style) on engine-labelled positions.

Input features per perspective: king_bucket * 768 + (piece type + 6 * relative colour) * 64 +
relative square, where the black perspective mirrors the board vertically and swaps colours and
king_bucket is one of 12 zones for the perspective's own king. Two accumulators (side to move,
opponent) of size H pass through a clipped ReLU and an antisymmetric linear output, so swapping
the perspectives exactly negates the score and the net carries no side-to-move bias.

    o = W_out . (crelu(acc_stm) - crelu(acc_nstm)),   eval_cp ~= 400 * o

Loss is MSE between sigmoid(o) and sigmoid(cp / 400), the usual win-probability space.

Usage: uv run python tools/train_nnue.py data/eval.csv weights/nnue.npz --epochs 6 --hidden 256
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
from torch import nn

PIECE_IDX = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
N_BUCKETS = 12
N_FEATURES = N_BUCKETS * 768
PAD = N_FEATURES
MAX_PIECES = 32


def king_bucket(sq: int) -> int:
    """Zone of the perspective's own king (relative square, own back rank = rank 0)."""
    rank, file = divmod(sq, 8)
    if rank == 0:
        return [0, 0, 1, 2, 3, 4, 5, 5][file]
    if rank == 1:
        return 6 if file < 3 else (7 if file < 5 else 8)
    if rank <= 3:
        return 9 if file < 4 else 10
    return 11


def encode(fen: str) -> tuple[list[int], list[int], bool] | None:
    parts = fen.split()
    rows = parts[0].split("/")
    stm_black = parts[1] == "b"
    pieces: list[tuple[int, int, int]] = []
    wk = bk = -1
    for r, row in enumerate(rows):
        rank = 7 - r
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
                continue
            colour = 0 if ch.isupper() else 1
            p = PIECE_IDX[ch.upper()]
            sq = rank * 8 + f
            pieces.append((p, colour, sq))
            if p == 5:
                if colour == 0:
                    wk = sq
                else:
                    bk = sq
            f += 1
    if len(pieces) > MAX_PIECES or wk < 0 or bk < 0:
        return None
    wb = king_bucket(wk) * 768
    bb = king_bucket(bk ^ 56) * 768
    white_view = [wb + (p + 6 * colour) * 64 + sq for p, colour, sq in pieces]
    black_view = [bb + (p + 6 * (1 - colour)) * 64 + (sq ^ 56) for p, colour, sq in pieces]
    if stm_black:
        return black_view, white_view, True
    return white_view, black_view, False


def encode_chunk(lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stm = np.full((len(lines), MAX_PIECES), PAD, dtype=np.int16)
    nstm = np.full((len(lines), MAX_PIECES), PAD, dtype=np.int16)
    target = np.zeros(len(lines), dtype=np.float32)
    k = 0
    for line in lines:
        fen, cp_s = line.rsplit(",", 1)
        enc = encode(fen)
        if enc is None:
            continue
        a, b, black = enc
        cp = float(cp_s)
        if black:
            cp = -cp  # labels are white-relative; the net is side-to-move relative
        stm[k, : len(a)] = a
        nstm[k, : len(b)] = b
        target[k] = cp
        k += 1
    return stm[:k], nstm[:k], target[:k]


def load(path: str, limit: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream the csv in chunks through a process pool; memory stays at the encoded arrays."""
    t = time.perf_counter()
    stms: list[np.ndarray] = []
    nstms: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    total = 0

    def chunks():
        nonlocal total
        buf: list[str] = []
        with open(path) as fh:
            for line in fh:
                buf.append(line.rstrip("\n"))
                total += 1
                if limit and total >= limit:
                    break
                if len(buf) >= 200_000:
                    yield buf
                    buf = []
        if buf:
            yield buf

    with Pool() as pool:
        for a, b, c in pool.imap(encode_chunk, chunks(), chunksize=1):
            stms.append(a)
            nstms.append(b)
            targets.append(c)
    stm = np.concatenate(stms)
    nstm = np.concatenate(nstms)
    target = np.concatenate(targets)
    print(f"loaded {len(target)} positions in {time.perf_counter() - t:.0f}s", flush=True)
    return stm, nstm, target


class Net(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.ft = nn.EmbeddingBag(PAD + 1, hidden, mode="sum", padding_idx=PAD)
        self.ft_bias = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(hidden, 1, bias=False)
        nn.init.normal_(self.ft.weight, std=0.01)
        with torch.no_grad():
            self.ft.weight[PAD].zero_()

    def forward(self, stm: torch.Tensor, nstm: torch.Tensor) -> torch.Tensor:
        a = torch.clamp(self.ft(stm) + self.ft_bias, 0.0, 1.0)
        b = torch.clamp(self.ft(nstm) + self.ft_bias, 0.0, 1.0)
        return self.out(a - b).squeeze(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scale", type=float, default=400.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    stm, nstm, target = load(args.data, args.limit)
    n = len(target)
    perm = np.random.default_rng(0).permutation(n)
    n_val = min(200_000, n // 20)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    tgt_t = torch.from_numpy(target)
    scale = args.scale

    def gather(idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(stm[idx].astype(np.int64)),
            torch.from_numpy(nstm[idx].astype(np.int64)),
            tgt_t[torch.from_numpy(idx)],
        )

    net = Net(args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.3)

    def batch_loss(idx: np.ndarray) -> torch.Tensor:
        a, b, t = gather(idx)
        pred = torch.sigmoid(net(a, b))
        want = torch.sigmoid(t / scale)
        return ((pred - want) ** 2).mean()

    for epoch in range(args.epochs):
        t = time.perf_counter()
        net.train()
        rng = np.random.default_rng(epoch)
        order = rng.permutation(train_idx)
        total = 0.0
        steps = 0
        for s in range(0, len(order), args.batch):
            loss = batch_loss(order[s : s + args.batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = 0.0
            vs = 0
            mae = 0.0
            for s in range(0, n_val, 65536):
                a, b, tv = gather(val_idx[s : s + 65536])
                o = net(a, b)
                vl += float(((torch.sigmoid(o) - torch.sigmoid(tv / scale)) ** 2).sum())
                mae += float((o * scale - tv).abs().sum())
                vs += len(tv)
        print(
            f"epoch {epoch + 1} train {total / steps:.5f} val {vl / vs:.5f} "
            f"val_mae {mae / vs:.0f}cp {time.perf_counter() - t:.0f}s",
            flush=True,
        )

    # Quantise. Feature transformer to QA (crelu clips at 1.0 -> QA), output to QB.
    qa, qb = 255, 64
    with torch.no_grad():
        w_ft = net.ft.weight[:PAD].detach().numpy()
        b_ft = net.ft_bias.detach().numpy()
        w_out = net.out.weight[0].detach().numpy()
    ft_w = np.clip(np.round(w_ft * qa), -32767, 32767).astype(np.int16)
    ft_b = np.clip(np.round(b_ft * qa), -32767, 32767).astype(np.int16)
    w_q = np.clip(np.round(w_out * qb), -32767, 32767).astype(np.int16)
    # Engine layout: out_w[:H] applies to the side to move, out_w[H:] to the opponent.
    out_w = np.concatenate([w_q, -w_q])
    out_b = np.int32(0)
    np.savez_compressed(
        args.out, ft_w=ft_w, ft_b=ft_b, out_w=out_w, out_b=out_b,
        qa=np.int32(qa), qb=np.int32(qb), scale=np.int32(int(scale)), hidden=np.int32(args.hidden),
        buckets=np.int32(N_BUCKETS),
    )
    print(f"saved {args.out}: max |ft_w| {np.abs(ft_w).max()}, max |out_w| {np.abs(out_w).max()}")


if __name__ == "__main__":
    sys.exit(main())
