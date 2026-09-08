# mypy: ignore-errors
"""Bitboard chess engine compiled with numba.

Everything here is our own code. Board representation, move generation, evaluation and
search all run inside numba-compiled functions so the search reaches node counts that pure
Python cannot. agent.py drives it and validates its output with python-chess.

Position layout: a uint64 array of length POS_LEN.
    0..11   piece bitboards, index = piece + 6 * colour, piece in P N B R Q K = 0..5
    12      side to move (0 white, 1 black)
    13      castling rights bitmask: 1 WK, 2 WQ, 4 BK, 8 BQ
    14      en passant square, 64 if none
    15      halfmove clock
    16      zobrist key

Move encoding (int64):
    bits 0-5 from, 6-11 to, 12-15 moving piece, 16-19 captured piece (6 = none),
    20-23 promotion piece (0 none, else 1..4 = N B R Q), 24-27 flags (1 ep, 2 castle, 4 double push)
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit, objmode

U = np.uint64
ZERO = U(0)
ONE = U(1)
FULL = U(0xFFFFFFFFFFFFFFFF)

POS_LEN = 17
I_SIDE = 12
I_CASTLE = 13
I_EP = 14
I_HALF = 15
I_KEY = 16

WHITE = 0
BLACK = 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
NO_PIECE = 6

FLAG_EP = 1
FLAG_CASTLE = 2
FLAG_DOUBLE = 4

MAX_PLY = 100
MAX_MOVES = 256

MATE = 100_000
MATE_BOUND = MATE - 2000
INF = 1_000_000

# Late move reduction table: LMR_TABLE[depth][move number].
LMR_TABLE = np.zeros((64, 64), dtype=np.int64)
for _d in range(1, 64):
    for _m in range(1, 64):
        LMR_TABLE[_d, _m] = int(0.75 + np.log(_d) * np.log(_m) / 2.25)

# aux (int64 [2, MAX_PLY+2]) rows
A_STATIC = 0  # static eval at each ply
A_MOVE = 1  # move played to reach each ply

TT_EXACT = 1
TT_LOWER = 2
TT_UPPER = 3
TT_BITS = 22
TT_SIZE = 1 << TT_BITS
TT_MASK = U(TT_SIZE - 1)

# --------------------------------------------------------------------------------------------
# Precomputed tables (built in Python at import, read as constants by numba).
# --------------------------------------------------------------------------------------------


def _bit(sq: int) -> int:
    return 1 << sq


def _build_tables() -> dict[str, np.ndarray]:
    knight = np.zeros(64, dtype=np.uint64)
    king = np.zeros(64, dtype=np.uint64)
    pawn_att = np.zeros((2, 64), dtype=np.uint64)
    file_ex = np.zeros(64, dtype=np.uint64)
    diag_ex = np.zeros(64, dtype=np.uint64)
    anti_ex = np.zeros(64, dtype=np.uint64)
    rank_att = np.zeros((64, 8), dtype=np.uint64)  # [inner occupancy 6 bits][file] -> rank mask
    for sq in range(64):
        r, f = divmod(sq, 8)
        for dr, df in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
            rr, ff = r + dr, f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                knight[sq] |= U(_bit(rr * 8 + ff))
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                rr, ff = r + dr, f + df
                if 0 <= rr < 8 and 0 <= ff < 8:
                    king[sq] |= U(_bit(rr * 8 + ff))
        for df in (-1, 1):
            ff = f + df
            if 0 <= ff < 8:
                if r + 1 < 8:
                    pawn_att[WHITE, sq] |= U(_bit((r + 1) * 8 + ff))
                if r - 1 >= 0:
                    pawn_att[BLACK, sq] |= U(_bit((r - 1) * 8 + ff))
        for rr in range(8):
            if rr != r:
                file_ex[sq] |= U(_bit(rr * 8 + f))
        for d in range(-7, 8):
            if d == 0:
                continue
            rr, ff = r + d, f + d
            if 0 <= rr < 8 and 0 <= ff < 8:
                diag_ex[sq] |= U(_bit(rr * 8 + ff))
            rr, ff = r + d, f - d
            if 0 <= rr < 8 and 0 <= ff < 8:
                anti_ex[sq] |= U(_bit(rr * 8 + ff))
    for occ6 in range(64):
        occ = occ6 << 1
        for f in range(8):
            att = 0
            for ff in range(f + 1, 8):
                att |= 1 << ff
                if occ & (1 << ff):
                    break
            for ff in range(f - 1, -1, -1):
                att |= 1 << ff
                if occ & (1 << ff):
                    break
            rank_att[occ6, f] = U(att)

    rng = np.random.default_rng(20240907)
    zob_piece = rng.integers(0, 2**63 - 1, size=(12, 64), dtype=np.int64).astype(np.uint64)
    zob_piece ^= rng.integers(0, 2**63 - 1, size=(12, 64), dtype=np.int64).astype(np.uint64) << U(1)
    zob_castle = rng.integers(0, 2**63 - 1, size=16, dtype=np.int64).astype(np.uint64)
    zob_ep = rng.integers(0, 2**63 - 1, size=65, dtype=np.int64).astype(np.uint64)
    zob_ep[64] = ZERO
    zob_side = U(rng.integers(0, 2**63 - 1, dtype=np.int64))

    # Passed-pawn front spans (own + adjacent files, ahead of the pawn) for both colours.
    front = np.zeros((2, 64), dtype=np.uint64)
    for sq in range(64):
        r, f = divmod(sq, 8)
        for ff in (f - 1, f, f + 1):
            if not 0 <= ff < 8:
                continue
            for rr in range(r + 1, 8):
                front[WHITE, sq] |= U(_bit(rr * 8 + ff))
            for rr in range(0, r):
                front[BLACK, sq] |= U(_bit(rr * 8 + ff))
    files = np.array([sum(_bit(r * 8 + f) for r in range(8)) for f in range(8)], dtype=np.uint64)
    return {
        "knight": knight,
        "king": king,
        "pawn_att": pawn_att,
        "file_ex": file_ex,
        "diag_ex": diag_ex,
        "anti_ex": anti_ex,
        "rank_att": rank_att,
        "zob_piece": zob_piece,
        "zob_castle": zob_castle,
        "zob_ep": zob_ep,
        "zob_side": np.array([zob_side], dtype=np.uint64),
        "front": front,
        "files": files,
    }


_T = _build_tables()
KNIGHT_ATT = _T["knight"]
KING_ATT = _T["king"]
PAWN_ATT = _T["pawn_att"]
FILE_EX = _T["file_ex"]
DIAG_EX = _T["diag_ex"]
ANTI_EX = _T["anti_ex"]
RANK_ATT = _T["rank_att"]
ZOB_PIECE = _T["zob_piece"]
ZOB_CASTLE = _T["zob_castle"]
ZOB_EP = _T["zob_ep"]
ZOB_SIDE = _T["zob_side"]
FRONT_SPAN = _T["front"]
FILE_MASK = _T["files"]

RANK_1 = U(0x00000000000000FF)
RANK_2 = U(0x000000000000FF00)
RANK_4 = U(0x00000000FF000000)
RANK_5 = U(0x000000FF00000000)
RANK_7 = U(0x00FF000000000000)
RANK_8 = U(0xFF00000000000000)
NOT_A_FILE = U(0xFEFEFEFEFEFEFEFE)
NOT_H_FILE = U(0x7F7F7F7F7F7F7F7F)

# Castling: rights cleared when a piece moves from / to a square.
CASTLE_MASK = np.full(64, 15, dtype=np.int64)
CASTLE_MASK[0] = 15 & ~2  # a1
CASTLE_MASK[4] = 15 & ~3  # e1
CASTLE_MASK[7] = 15 & ~1  # h1
CASTLE_MASK[56] = 15 & ~8  # a8
CASTLE_MASK[60] = 15 & ~12  # e8
CASTLE_MASK[63] = 15 & ~4  # h8

# --------------------------------------------------------------------------------------------
# Evaluation tables: material folded in, [12][64] (piece + 6*colour), a1 = 0.
# --------------------------------------------------------------------------------------------

MG_VALUE = [82, 337, 365, 477, 1025, 0]
EG_VALUE = [94, 281, 297, 512, 936, 0]
PHASE_INC = np.array([0, 1, 1, 2, 4, 0], dtype=np.int64)

# Rank 8 first, White's view, as in agent v1.
_PST_MG = [
    [  # pawn
        0, 0, 0, 0, 0, 0, 0, 0,
        60, 70, 60, 70, 65, 60, 40, 30,
        10, 15, 25, 30, 35, 40, 20, 5,
        -5, 5, 5, 20, 22, 10, 5, -10,
        -15, -5, 0, 15, 18, 5, -5, -20,
        -15, -5, -5, -5, 5, 0, 10, -15,
        -20, 0, -15, -20, -15, 15, 20, -20,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [  # knight
        -110, -60, -30, -30, -30, -30, -60, -110,
        -50, -25, 20, 10, 10, 20, -25, -50,
        -30, 20, 25, 40, 40, 25, 20, -30,
        -20, 10, 20, 35, 35, 20, 10, -20,
        -20, 5, 20, 25, 25, 20, 5, -20,
        -30, -5, 10, 15, 15, 10, -5, -30,
        -50, -25, -5, 5, 5, -5, -25, -50,
        -80, -40, -35, -25, -25, -35, -40, -80,
    ],
    [  # bishop
        -30, 0, -60, -40, -25, -40, 5, -10,
        -25, 15, -15, -10, 30, 55, 20, -45,
        -15, 35, 40, 40, 35, 50, 35, 0,
        -5, 5, 20, 50, 35, 35, 5, 0,
        -5, 15, 15, 25, 35, 10, 10, 5,
        0, 15, 15, 15, 15, 25, 20, 10,
        5, 15, 15, 0, 5, 20, 30, 0,
        -35, -5, -15, -20, -15, -10, -40, -20,
    ],
    [  # rook
        30, 40, 30, 50, 60, 10, 30, 45,
        25, 30, 60, 60, 80, 65, 25, 45,
        -5, 20, 25, 35, 15, 45, 60, 15,
        -25, -10, 5, 25, 25, 35, -10, -20,
        -35, -25, -10, 0, 10, -5, 5, -25,
        -45, -25, -15, -15, 5, 0, -5, -35,
        -45, -15, -20, -10, 0, 10, -5, -70,
        -20, -15, 0, 15, 15, 5, -35, -25,
    ],
    [  # queen
        -30, 0, 30, 10, 60, 45, 45, 45,
        -25, -40, -5, 0, -15, 55, 30, 55,
        -15, -15, 5, 10, 30, 55, 45, 55,
        -25, -25, -15, -15, 0, 15, 0, 0,
        -10, -25, -10, -10, 0, -5, 5, 0,
        -15, 0, -10, 0, -5, 0, 15, 5,
        -35, -10, 10, 0, 10, 15, 0, 0,
        0, -20, -10, 10, -15, -25, -30, -50,
    ],
    [  # king
        -65, 25, 15, -15, -55, -35, 0, 15,
        30, 0, -20, -5, -10, -5, -40, -30,
        -10, 25, 0, -15, -20, 5, 20, -20,
        -15, -20, -10, -25, -30, -25, -15, -35,
        -50, 0, -25, -40, -45, -45, -35, -50,
        -15, -15, -20, -45, -45, -30, -15, -25,
        0, 5, -10, -65, -45, -15, 10, 10,
        -15, 35, 10, -55, 5, -30, 25, 15,
    ],
]
_PST_EG = [
    [
        0, 0, 0, 0, 0, 0, 0, 0,
        120, 115, 105, 90, 95, 90, 110, 125,
        65, 70, 55, 40, 35, 35, 60, 60,
        20, 15, 5, -5, -5, 0, 10, 15,
        5, 5, -10, -10, -10, -10, 0, 0,
        0, 0, -5, 0, 0, -5, -5, -5,
        5, 5, 5, 5, 8, 0, 0, -5,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        -60, -40, -15, -30, -30, -15, -40, -60,
        -25, -10, -25, 0, 0, -25, -10, -25,
        -25, -20, 10, 10, 10, 10, -20, -25,
        -20, 5, 20, 20, 20, 20, 5, -20,
        -20, -5, 15, 25, 25, 15, -5, -20,
        -25, -5, 0, 15, 15, 0, -5, -25,
        -40, -20, -10, -5, -5, -10, -20, -40,
        -60, -50, -25, -15, -15, -25, -50, -60,
    ],
    [
        -15, -20, -10, -10, -5, -10, -15, -25,
        -10, -5, 5, -10, -5, -10, -5, -15,
        0, -10, 0, 0, 0, 5, 0, 5,
        -5, 10, 10, 10, 15, 10, 5, 0,
        -5, 5, 15, 20, 5, 10, -5, -10,
        -10, -5, 10, 10, 15, 5, -5, -15,
        -15, -20, -5, 0, 5, -10, -15, -25,
        -25, -10, -25, -5, -10, -15, -5, -15,
    ],
    [
        15, 10, 20, 15, 10, 10, 10, 5,
        10, 15, 15, 10, -5, 5, 10, 5,
        5, 5, 5, 5, 5, -5, -5, -5,
        5, 5, 15, 0, 0, 0, 0, 0,
        5, 5, 10, 5, -5, -5, -10, -10,
        -5, 0, -5, 0, -5, -10, -10, -15,
        -5, -5, 0, 0, -10, -10, -10, -5,
        -10, 0, 5, 0, -5, -15, 5, -20,
    ],
    [
        -10, 20, 20, 25, 25, 20, 10, 20,
        -15, 20, 30, 40, 60, 25, 30, 0,
        -20, 5, 10, 50, 45, 35, 20, 10,
        5, 20, 25, 45, 55, 40, 55, 35,
        -20, 30, 20, 45, 30, 35, 40, 25,
        -15, -25, 15, 5, 10, 15, 10, 5,
        -20, -25, -30, -15, -15, -25, -35, -30,
        -35, -30, -20, -45, -5, -30, -20, -40,
    ],
    [
        -75, -35, -20, -20, -10, 15, 5, -15,
        -10, 15, 15, 15, 15, 40, 25, 10,
        10, 15, 25, 15, 20, 45, 45, 15,
        -10, 20, 25, 25, 25, 35, 25, 5,
        -20, -5, 20, 25, 25, 25, 10, -10,
        -20, -5, 10, 20, 25, 15, 5, -10,
        -25, -10, 5, 15, 15, 5, -5, -15,
        -55, -35, -20, -10, -30, -15, -25, -45,
    ],
]


def _build_pst() -> tuple[np.ndarray, np.ndarray]:
    mg = np.zeros((12, 64), dtype=np.int64)
    eg = np.zeros((12, 64), dtype=np.int64)
    for p in range(6):
        for i in range(64):
            rank = 7 - i // 8
            file = i % 8
            sq = rank * 8 + file
            mg[p, sq] = _PST_MG[p][i] + MG_VALUE[p]
            eg[p, sq] = _PST_EG[p][i] + EG_VALUE[p]
            # Black mirrors vertically.
            mg[p + 6, sq ^ 56] = mg[p, sq]
            eg[p + 6, sq ^ 56] = eg[p, sq]
    return mg, eg


PST_MG, PST_EG = _build_pst()
PASSED_MG = np.array([0, 5, 10, 15, 30, 50, 80, 0], dtype=np.int64)
PASSED_EG = np.array([0, 10, 20, 35, 60, 100, 150, 0], dtype=np.int64)
MVV_VALUE = np.array([100, 300, 300, 500, 900, 2000, 0], dtype=np.int64)

# --------------------------------------------------------------------------------------------
# Bit helpers
# --------------------------------------------------------------------------------------------

M1 = U(0x5555555555555555)
M2 = U(0x3333333333333333)
M4 = U(0x0F0F0F0F0F0F0F0F)
H01 = U(0x0101010101010101)
BS1 = U(0x00FF00FF00FF00FF)
BS2 = U(0x0000FFFF0000FFFF)


@njit(cache=False)
def popcount(x):
    x = x - ((x >> ONE) & M1)
    x = (x & M2) + ((x >> U(2)) & M2)
    x = (x + (x >> U(4))) & M4
    return np.int64((x * H01) >> U(56))


@njit(cache=False)
def lsb(x):
    # Index of least significant set bit; x must be non-zero.
    return popcount((x & (~x + ONE)) - ONE)


@njit(cache=False)
def bswap(x):
    x = ((x >> U(8)) & BS1) | ((x & BS1) << U(8))
    x = ((x >> U(16)) & BS2) | ((x & BS2) << U(16))
    x = (x >> U(32)) | (x << U(32))
    return x


@njit(cache=False)
def line_attacks(occ, mask, sq_bit):
    # Hyperbola quintessence along a file / diagonal / anti-diagonal.
    o = occ & mask
    fwd = o - (sq_bit << ONE)
    rev = bswap(bswap(o) - (bswap(sq_bit) << ONE))
    return (fwd ^ rev) & mask


@njit(cache=False)
def rank_attacks(occ, sq):
    r = sq >> 3
    f = sq & 7
    occ6 = np.int64((occ >> U(r * 8 + 1)) & U(63))
    return RANK_ATT[occ6, f] << U(r * 8)


@njit(cache=False)
def bishop_attacks(occ, sq):
    b = ONE << U(sq)
    return line_attacks(occ, DIAG_EX[sq], b) | line_attacks(occ, ANTI_EX[sq], b)


@njit(cache=False)
def rook_attacks(occ, sq):
    b = ONE << U(sq)
    return line_attacks(occ, FILE_EX[sq], b) | rank_attacks(occ, sq)


@njit(cache=False)
def occupancy(pos, colour):
    base = colour * 6
    return (
        pos[base] | pos[base + 1] | pos[base + 2] | pos[base + 3] | pos[base + 4] | pos[base + 5]
    )


@njit(cache=False)
def is_attacked(pos, sq, by, occ):
    # Is square sq attacked by side `by`, given total occupancy occ.
    base = by * 6
    if PAWN_ATT[1 - by, sq] & pos[base + PAWN]:
        return True
    if KNIGHT_ATT[sq] & pos[base + KNIGHT]:
        return True
    if KING_ATT[sq] & pos[base + KING]:
        return True
    bq = pos[base + BISHOP] | pos[base + QUEEN]
    if bq and (bishop_attacks(occ, sq) & bq):
        return True
    rq = pos[base + ROOK] | pos[base + QUEEN]
    return bool(rq and rook_attacks(occ, sq) & rq)


@njit(cache=False)
def in_check(pos):
    side = np.int64(pos[I_SIDE])
    ksq = lsb(pos[side * 6 + KING])
    occ = occupancy(pos, 0) | occupancy(pos, 1)
    return is_attacked(pos, ksq, 1 - side, occ)


@njit(cache=False)
def piece_on(pos, sq, colour):
    b = ONE << U(sq)
    base = colour * 6
    for p in range(6):
        if pos[base + p] & b:
            return p
    return NO_PIECE


# --------------------------------------------------------------------------------------------
# Move generation
# --------------------------------------------------------------------------------------------


@njit(cache=False)
def encode(frm, to, piece, captured, promo, flags):
    return frm | (to << 6) | (piece << 12) | (captured << 16) | (promo << 20) | (flags << 24)


@njit(cache=False)
def mv_from(m):
    return m & 63


@njit(cache=False)
def mv_to(m):
    return (m >> 6) & 63


@njit(cache=False)
def mv_piece(m):
    return (m >> 12) & 15


@njit(cache=False)
def mv_captured(m):
    return (m >> 16) & 15


@njit(cache=False)
def mv_promo(m):
    return (m >> 20) & 15


@njit(cache=False)
def mv_flags(m):
    return (m >> 24) & 15


@njit(cache=False)
def gen_moves(pos, buf, captures_only):
    """Pseudo-legal moves into buf; returns count. Legality is checked in make_move."""
    n = 0
    side = np.int64(pos[I_SIDE])
    them = 1 - side
    us_base = side * 6
    them * 6
    own = occupancy(pos, side)
    opp = occupancy(pos, them)
    occ = own | opp
    empty = ~occ
    ep = np.int64(pos[I_EP])

    # Pawns
    pawns = pos[us_base + PAWN]
    if side == WHITE:
        promo_rank = RANK_8
        single = (pawns << U(8)) & empty
        double = ((single & U(0xFF0000)) << U(8)) & empty
        shift = 8
    else:
        promo_rank = RANK_1
        single = (pawns >> U(8)) & empty
        double = ((single & U(0xFF0000000000)) >> U(8)) & empty
        shift = -8

    if not captures_only:
        bb = single & ~promo_rank
        while bb:
            to = lsb(bb)
            bb &= bb - ONE
            buf[n] = encode(to - shift, to, PAWN, NO_PIECE, 0, 0)
            n += 1
        bb = double
        while bb:
            to = lsb(bb)
            bb &= bb - ONE
            buf[n] = encode(to - 2 * shift, to, PAWN, NO_PIECE, 0, FLAG_DOUBLE)
            n += 1
    # Promotions by push (always generated: they are tactical).
    bb = single & promo_rank
    while bb:
        to = lsb(bb)
        bb &= bb - ONE
        frm = to - shift
        buf[n] = encode(frm, to, PAWN, NO_PIECE, 4, 0)
        n += 1
        if not captures_only:
            buf[n] = encode(frm, to, PAWN, NO_PIECE, 1, 0)
            buf[n + 1] = encode(frm, to, PAWN, NO_PIECE, 3, 0)
            buf[n + 2] = encode(frm, to, PAWN, NO_PIECE, 2, 0)
            n += 3
    # Pawn captures
    bb = pawns
    while bb:
        frm = lsb(bb)
        bb &= bb - ONE
        att = PAWN_ATT[side, frm] & opp
        while att:
            to = lsb(att)
            att &= att - ONE
            cap = piece_on(pos, to, them)
            if (ONE << U(to)) & promo_rank:
                buf[n] = encode(frm, to, PAWN, cap, 4, 0)
                n += 1
                if not captures_only:
                    buf[n] = encode(frm, to, PAWN, cap, 1, 0)
                    buf[n + 1] = encode(frm, to, PAWN, cap, 3, 0)
                    buf[n + 2] = encode(frm, to, PAWN, cap, 2, 0)
                    n += 3
            else:
                buf[n] = encode(frm, to, PAWN, cap, 0, 0)
                n += 1
        if ep < 64 and (PAWN_ATT[side, frm] & (ONE << U(ep))):
            buf[n] = encode(frm, ep, PAWN, PAWN, 0, FLAG_EP)
            n += 1

    # Knights
    bb = pos[us_base + KNIGHT]
    while bb:
        frm = lsb(bb)
        bb &= bb - ONE
        att = KNIGHT_ATT[frm] & ~own
        if captures_only:
            att &= opp
        while att:
            to = lsb(att)
            att &= att - ONE
            cap = piece_on(pos, to, them) if (ONE << U(to)) & opp else NO_PIECE
            buf[n] = encode(frm, to, KNIGHT, cap, 0, 0)
            n += 1

    # Bishops, rooks, queens
    for piece in range(BISHOP, QUEEN + 1):
        bb = pos[us_base + piece]
        while bb:
            frm = lsb(bb)
            bb &= bb - ONE
            if piece == BISHOP:
                att = bishop_attacks(occ, frm)
            elif piece == ROOK:
                att = rook_attacks(occ, frm)
            else:
                att = bishop_attacks(occ, frm) | rook_attacks(occ, frm)
            att &= ~own
            if captures_only:
                att &= opp
            while att:
                to = lsb(att)
                att &= att - ONE
                cap = piece_on(pos, to, them) if (ONE << U(to)) & opp else NO_PIECE
                buf[n] = encode(frm, to, piece, cap, 0, 0)
                n += 1

    # King
    ksq = lsb(pos[us_base + KING])
    att = KING_ATT[ksq] & ~own
    if captures_only:
        att &= opp
    while att:
        to = lsb(att)
        att &= att - ONE
        cap = piece_on(pos, to, them) if (ONE << U(to)) & opp else NO_PIECE
        buf[n] = encode(ksq, to, KING, cap, 0, 0)
        n += 1

    # Castling
    if not captures_only:
        rights = np.int64(pos[I_CASTLE])
        if side == WHITE:
            if (rights & 1) and not (occ & U(0x60)):  # f1 g1 empty  # noqa: SIM102
                if (
                    not is_attacked(pos, 4, them, occ)
                    and not is_attacked(pos, 5, them, occ)
                    and not is_attacked(pos, 6, them, occ)
                ):
                    buf[n] = encode(4, 6, KING, NO_PIECE, 0, FLAG_CASTLE)
                    n += 1
            if (rights & 2) and not (occ & U(0x0E)):  # b1 c1 d1 empty  # noqa: SIM102
                if (
                    not is_attacked(pos, 4, them, occ)
                    and not is_attacked(pos, 3, them, occ)
                    and not is_attacked(pos, 2, them, occ)
                ):
                    buf[n] = encode(4, 2, KING, NO_PIECE, 0, FLAG_CASTLE)
                    n += 1
        else:
            if (rights & 4) and not (occ & U(0x6000000000000000)) and (
                not is_attacked(pos, 60, them, occ)
                and not is_attacked(pos, 61, them, occ)
                and not is_attacked(pos, 62, them, occ)
            ):
                buf[n] = encode(60, 62, KING, NO_PIECE, 0, FLAG_CASTLE)
                n += 1
            if (rights & 8) and not (occ & U(0x0E00000000000000)) and (
                not is_attacked(pos, 60, them, occ)
                and not is_attacked(pos, 59, them, occ)
                and not is_attacked(pos, 58, them, occ)
            ):
                buf[n] = encode(60, 58, KING, NO_PIECE, 0, FLAG_CASTLE)
                n += 1
    return n


@njit(cache=False)
def make_move(src, dst, m):
    """Apply m to src writing dst. Returns False if the move leaves own king in check."""
    for i in range(POS_LEN):
        dst[i] = src[i]
    side = np.int64(src[I_SIDE])
    them = 1 - side
    us_base = side * 6
    them_base = them * 6
    frm = mv_from(m)
    to = mv_to(m)
    piece = mv_piece(m)
    cap = mv_captured(m)
    promo = mv_promo(m)
    flags = mv_flags(m)
    fb = ONE << U(frm)
    tb = ONE << U(to)
    key = src[I_KEY]

    # Move the piece.
    dst[us_base + piece] ^= fb | tb
    key ^= ZOB_PIECE[us_base + piece, frm] ^ ZOB_PIECE[us_base + piece, to]

    half = np.int64(src[I_HALF]) + 1
    if piece == PAWN:
        half = 0

    if cap != NO_PIECE:
        half = 0
        if flags & FLAG_EP:
            cap_sq = to - 8 if side == WHITE else to + 8
            dst[them_base + PAWN] ^= ONE << U(cap_sq)
            key ^= ZOB_PIECE[them_base + PAWN, cap_sq]
        else:
            dst[them_base + cap] ^= tb
            key ^= ZOB_PIECE[them_base + cap, to]

    if promo:
        dst[us_base + PAWN] ^= tb
        dst[us_base + promo] ^= tb
        key ^= ZOB_PIECE[us_base + PAWN, to] ^ ZOB_PIECE[us_base + promo, to]

    if flags & FLAG_CASTLE:
        if to == 6:
            rf, rt = 7, 5
        elif to == 2:
            rf, rt = 0, 3
        elif to == 62:
            rf, rt = 63, 61
        else:
            rf, rt = 56, 59
        dst[us_base + ROOK] ^= (ONE << U(rf)) | (ONE << U(rt))
        key ^= ZOB_PIECE[us_base + ROOK, rf] ^ ZOB_PIECE[us_base + ROOK, rt]

    # Castling rights.
    rights = np.int64(src[I_CASTLE])
    new_rights = rights & CASTLE_MASK[frm] & CASTLE_MASK[to]
    if new_rights != rights:
        key ^= ZOB_CASTLE[rights] ^ ZOB_CASTLE[new_rights]
    dst[I_CASTLE] = U(new_rights)

    # En passant square.
    old_ep = np.int64(src[I_EP])
    new_ep = 64
    if flags & FLAG_DOUBLE:
        cand = (frm + to) >> 1
        # Only record the square when an enemy pawn could capture there, matching the
        # FEN convention the platform uses, so keys stay comparable across the game.
        if PAWN_ATT[side, cand] & dst[them_base + PAWN]:
            new_ep = cand
    if old_ep != new_ep:
        key ^= ZOB_EP[old_ep] ^ ZOB_EP[new_ep]
    dst[I_EP] = U(new_ep)

    dst[I_HALF] = U(half)
    dst[I_SIDE] = U(them)
    key ^= ZOB_SIDE[0]
    dst[I_KEY] = key

    # Legality: own king must not be attacked.
    occ = occupancy(dst, 0) | occupancy(dst, 1)
    ksq = lsb(dst[us_base + KING])
    return not is_attacked(dst, ksq, them, occ)


@njit(cache=False)
def make_null(src, dst):
    for i in range(POS_LEN):
        dst[i] = src[i]
    side = np.int64(src[I_SIDE])
    key = src[I_KEY] ^ ZOB_SIDE[0]
    old_ep = np.int64(src[I_EP])
    if old_ep != 64:
        key ^= ZOB_EP[old_ep]
    dst[I_EP] = U(64)
    dst[I_SIDE] = U(1 - side)
    dst[I_KEY] = key


@njit(cache=False)
def compute_key(pos):
    key = ZERO
    for i in range(12):
        bb = pos[i]
        while bb:
            sq = lsb(bb)
            bb &= bb - ONE
            key ^= ZOB_PIECE[i, sq]
    key ^= ZOB_CASTLE[np.int64(pos[I_CASTLE])]
    key ^= ZOB_EP[np.int64(pos[I_EP])]
    if pos[I_SIDE] == ONE:
        key ^= ZOB_SIDE[0]
    return key


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------

BISHOP_PAIR = 30
TEMPO = 10


@njit(cache=False)
def evaluate(pos):
    """Static evaluation from the side to move's point of view, in centipawns."""
    mg = 0
    eg = 0
    phase = 0
    for i in range(12):
        bb = pos[i]
        p = i if i < 6 else i - 6
        sign = 1 if i < 6 else -1
        while bb:
            sq = lsb(bb)
            bb &= bb - ONE
            mg += sign * PST_MG[i, sq]
            eg += sign * PST_EG[i, sq]
            phase += PHASE_INC[p]

    if popcount(pos[BISHOP]) >= 2:
        mg += BISHOP_PAIR
        eg += BISHOP_PAIR
    if popcount(pos[6 + BISHOP]) >= 2:
        mg -= BISHOP_PAIR
        eg -= BISHOP_PAIR

    wp = pos[PAWN]
    bp = pos[6 + PAWN]
    for f in range(8):
        wc = popcount(wp & FILE_MASK[f])
        bc = popcount(bp & FILE_MASK[f])
        if wc > 1:
            mg -= 10 * (wc - 1)
            eg -= 20 * (wc - 1)
        if bc > 1:
            mg += 10 * (bc - 1)
            eg += 20 * (bc - 1)
    bb = wp
    while bb:
        sq = lsb(bb)
        bb &= bb - ONE
        if not (FRONT_SPAN[WHITE, sq] & bp):
            r = sq >> 3
            mg += PASSED_MG[r]
            eg += PASSED_EG[r]
    bb = bp
    while bb:
        sq = lsb(bb)
        bb &= bb - ONE
        if not (FRONT_SPAN[BLACK, sq] & wp):
            r = 7 - (sq >> 3)
            mg -= PASSED_MG[r]
            eg -= PASSED_EG[r]

    if phase > 24:
        phase = 24
    total = mg * phase + eg * (24 - phase)
    # Round toward zero so the evaluation is exactly colour-symmetric.
    score = total // 24 if total >= 0 else -((-total) // 24)
    if pos[I_SIDE] == ONE:
        score = -score
    return score + TEMPO


# --------------------------------------------------------------------------------------------
# Neural evaluation (NNUE-style perspective net, see tools/train_nnue.py)
# --------------------------------------------------------------------------------------------
# nn = (ft_w int16[768, H], ft_b int16[H], out_w int16[2H], out_meta int64[4] = qa, qb, scale, b)


# King zone of the perspective's own king, indexed by relative square (own back rank = 0).
KING_BUCKET = np.zeros(64, dtype=np.int64)
for _sq in range(64):
    _r, _f = divmod(_sq, 8)
    if _r == 0:
        KING_BUCKET[_sq] = [0, 0, 1, 2, 3, 4, 5, 5][_f]
    elif _r == 1:
        KING_BUCKET[_sq] = 6 if _f < 3 else (7 if _f < 5 else 8)
    elif _r <= 3:
        KING_BUCKET[_sq] = 9 if _f < 4 else 10
    else:
        KING_BUCKET[_sq] = 11
N_BUCKETS = 12


@njit(cache=False)
def feat(piece, colour, sq, persp, kb):
    return kb * 768 + (piece + 6 * (colour ^ persp)) * 64 + (sq ^ (56 * persp))


@njit(cache=False)
def king_bucket_of(pos, persp):
    ksq = lsb(pos[persp * 6 + KING])
    return KING_BUCKET[ksq ^ (56 * persp)]


@njit(cache=False)
def acc_refresh_persp(pos, acc, nn, persp):
    ft_w = nn[0]
    ft_b = nn[1]
    h = ft_b.shape[0]
    kb = king_bucket_of(pos, persp)
    for j in range(h):
        acc[persp, j] = ft_b[j]
    for i in range(12):
        p = i if i < 6 else i - 6
        colour = 0 if i < 6 else 1
        bb = pos[i]
        while bb:
            sq = lsb(bb)
            bb &= bb - ONE
            f = feat(p, colour, sq, persp, kb)
            for j in range(h):
                acc[persp, j] += ft_w[f, j]


@njit(cache=False)
def acc_refresh(pos, acc, nn):
    acc_refresh_persp(pos, acc, nn, 0)
    acc_refresh_persp(pos, acc, nn, 1)


@njit(cache=False)
def acc_add_sub(acc, nn, persp, kb, p_add, c_add, sq_add, p_sub, c_sub, sq_sub):
    ft_w = nn[0]
    h = ft_w.shape[1]
    fa = feat(p_add, c_add, sq_add, persp, kb)
    fs = feat(p_sub, c_sub, sq_sub, persp, kb)
    for j in range(h):
        acc[persp, j] += ft_w[fa, j] - ft_w[fs, j]


@njit(cache=False)
def acc_sub(acc, nn, persp, kb, p, c, sq):
    ft_w = nn[0]
    h = ft_w.shape[1]
    fs = feat(p, c, sq, persp, kb)
    for j in range(h):
        acc[persp, j] -= ft_w[fs, j]


@njit(cache=False)
def acc_apply_move(src, dst, m, acc_src, acc_dst, nn):
    """acc_dst = acc_src updated for move m taking position src to dst.

    The mover's own perspective is rebuilt from scratch when its king moves, because the king
    bucket changes every feature index; the other perspective is always updated incrementally.
    """
    h = acc_src.shape[1]
    side = np.int64(src[I_SIDE])
    them = 1 - side
    frm = mv_from(m)
    to = mv_to(m)
    piece = mv_piece(m)
    cap = mv_captured(m)
    promo = mv_promo(m)
    flags = mv_flags(m)
    for persp in range(2):
        if persp == side and piece == KING:
            acc_refresh_persp(dst, acc_dst, nn, persp)
            continue
        for j in range(h):
            acc_dst[persp, j] = acc_src[persp, j]
        kb = king_bucket_of(src, persp)
        if promo:
            acc_add_sub(acc_dst, nn, persp, kb, promo, side, to, PAWN, side, frm)
        else:
            acc_add_sub(acc_dst, nn, persp, kb, piece, side, to, piece, side, frm)
        if cap != NO_PIECE:
            if flags & FLAG_EP:
                cap_sq = to - 8 if side == WHITE else to + 8
                acc_sub(acc_dst, nn, persp, kb, PAWN, them, cap_sq)
            else:
                acc_sub(acc_dst, nn, persp, kb, cap, them, to)
        if flags & FLAG_CASTLE:
            if to == 6:
                acc_add_sub(acc_dst, nn, persp, kb, ROOK, side, 5, ROOK, side, 7)
            elif to == 2:
                acc_add_sub(acc_dst, nn, persp, kb, ROOK, side, 3, ROOK, side, 0)
            elif to == 62:
                acc_add_sub(acc_dst, nn, persp, kb, ROOK, side, 61, ROOK, side, 63)
            else:
                acc_add_sub(acc_dst, nn, persp, kb, ROOK, side, 59, ROOK, side, 56)


@njit(cache=False)
def acc_copy(acc_src, acc_dst):
    h = acc_src.shape[1]
    for persp in range(2):
        for j in range(h):
            acc_dst[persp, j] = acc_src[persp, j]


@njit(cache=False)
def evaluate_nn(pos, acc, nn):
    out_w = nn[2]
    meta = nn[3]
    qa = meta[0]
    qb = meta[1]
    scale = meta[2]
    h = acc.shape[1]
    stm = np.int64(pos[I_SIDE])
    nstm = 1 - stm
    total = meta[3]
    for j in range(h):
        v = np.int64(acc[stm, j])
        if v < 0:
            v = 0
        elif v > qa:
            v = qa
        total += v * out_w[j]
        v = np.int64(acc[nstm, j])
        if v < 0:
            v = 0
        elif v > qa:
            v = qa
        total += v * out_w[h + j]
    score = (total * scale) // (qa * qb)
    if score > 3000:
        score = 3000
    elif score < -3000:
        score = -3000
    return score


@njit(cache=False)
def eval_pos(pos, acc, nn, use_nn):
    if use_nn:
        return evaluate_nn(pos, acc, nn)
    return evaluate(pos)


# --------------------------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------------------------

# ctx (int64 array) indices
C_NODES = 0
C_ABORT = 1
C_NODE_LIMIT = 2
C_GAME_KEYS = 3  # number of game history keys
C_ROOT_DEPTH = 4
# ctxf (float64 array): [0] deadline (perf_counter seconds)


@njit(cache=False)
def tt_probe(tt_keys, tt_data, key):
    idx = np.int64(key & TT_MASK)
    if tt_keys[idx] == key:
        return tt_data[idx]
    return np.int64(0)


@njit(cache=False)
def tt_store(tt_keys, tt_data, key, depth, flag, score, move):
    idx = np.int64(key & TT_MASK)
    # Same position: never let a shallower search overwrite a deeper result. A different
    # position always replaces (the table is large and always-replace ages entries for free).
    if tt_keys[idx] == key:
        old_depth = (tt_data[idx] >> 28) & 255
        if old_depth > depth:
            return
    tt_keys[idx] = key
    tt_data[idx] = (
        (move & 0xFFFFFFF) | (depth << 28) | (flag << 36) | ((score + (1 << 20)) << 40)
    )


@njit(cache=False)
def tt_move(data):
    return data & 0xFFFFFFF


@njit(cache=False)
def tt_depth(data):
    return (data >> 28) & 255


@njit(cache=False)
def tt_flag(data):
    return (data >> 36) & 15


@njit(cache=False)
def tt_score(data):
    return ((data >> 40) & 0x1FFFFF) - (1 << 20)


@njit(cache=False)
def check_time(ctx, ctxf):
    ctx[C_NODES] += 1
    if (ctx[C_NODES] & 2047) == 0:
        if ctx[C_NODES] >= ctx[C_NODE_LIMIT]:
            ctx[C_ABORT] = 1
        else:
            with objmode(now="float64"):
                now = time.perf_counter()
            if now >= ctxf[0]:
                ctx[C_ABORT] = 1


@njit(cache=False)
def score_move(m, tt_m, killers, history, ply, side, counter_m):
    if m == tt_m:
        return 10_000_000
    cap = mv_captured(m)
    promo = mv_promo(m)
    if cap != NO_PIECE:
        return 1_000_000 + MVV_VALUE[cap] * 10 - MVV_VALUE[mv_piece(m)] // 10 + promo * 100
    if promo:
        return 900_000 + promo * 1000
    if m == killers[ply, 0]:
        return 800_000
    if m == killers[ply, 1]:
        return 700_000
    if m == counter_m:
        return 600_000
    return history[side, mv_from(m), mv_to(m)]


@njit(cache=False)
def pick_next(moves, scores, n, i):
    # Selection: swap the best remaining move into slot i.
    best = i
    for j in range(i + 1, n):
        if scores[j] > scores[best]:
            best = j
    if best != i:
        moves[i], moves[best] = moves[best], moves[i]
        scores[i], scores[best] = scores[best], scores[i]
    return moves[i]


@njit(cache=False)
def is_repetition(stack, ply, game_keys, n_game):
    """True if the position at stack[ply] occurred earlier on the search path or in the game.

    stack[0] is the current game position and equals game_keys[n_game - 1]; earlier game
    positions continue the same sequence backwards. Only positions with the same side to move
    can repeat, so we step two plies at a time, no further back than the halfmove clock.
    """
    key = stack[ply, I_KEY]
    half = np.int64(stack[ply, I_HALF])
    if half < 4:
        return False
    back = 2
    p = ply - 2
    while back <= half:
        if p >= 0:
            if stack[p, I_KEY] == key:
                return True
        else:
            g = n_game - 1 + p
            if g < 0:
                return False
            if game_keys[g] == key:
                return True
        p -= 2
        back += 2
    return False


@njit(cache=False)
def quiesce(
    stack, ply, alpha, beta, ctx, ctxf, movebuf, scorebuf, acc, nn, use_nn, tt_keys, tt_data
):
    check_time(ctx, ctxf)
    if ctx[C_ABORT]:
        return 0
    pos = stack[ply]
    key = pos[I_KEY]
    data = tt_probe(tt_keys, tt_data, key)
    if data != 0:
        s = tt_score(data)
        f = tt_flag(data)
        if f == TT_EXACT or (f == TT_LOWER and s >= beta) or (f == TT_UPPER and s <= alpha):
            return s
    stand = eval_pos(pos, acc[ply], nn, use_nn)
    if stand >= beta:
        return stand
    if stand > alpha:
        alpha = stand
    if ply >= MAX_PLY - 1:
        return stand
    orig_alpha = alpha

    moves = movebuf[ply]
    scores = scorebuf[ply]
    n = gen_moves(pos, moves, True)
    for i in range(n):
        m = moves[i]
        scores[i] = (
            MVV_VALUE[mv_captured(m)] * 10 - MVV_VALUE[mv_piece(m)] // 10 + mv_promo(m) * 100
        )
    best = stand
    best_move = 0
    for i in range(n):
        m = pick_next(moves, scores, n, i)
        cap = mv_captured(m)
        gain = MVV_VALUE[cap] + (900 if mv_promo(m) else 0)
        if stand + gain + 200 < alpha:
            continue
        if not make_move(pos, stack[ply + 1], m):
            continue
        if use_nn:
            acc_apply_move(pos, stack[ply + 1], m, acc[ply], acc[ply + 1], nn)
        score = -quiesce(
            stack, ply + 1, -beta, -alpha, ctx, ctxf, movebuf, scorebuf, acc, nn, use_nn,
            tt_keys, tt_data,
        )
        if ctx[C_ABORT]:
            return 0
        if score > best:
            best = score
            best_move = m
        if score > alpha:
            alpha = score
            if alpha >= beta:
                break
    if best >= beta:
        flag = TT_LOWER
    elif best <= orig_alpha:
        flag = TT_UPPER
    else:
        flag = TT_EXACT
    tt_store(tt_keys, tt_data, key, 0, flag, best, best_move)
    return best


@njit(cache=False)
def insufficient_material(pos):
    # No pawns, rooks or queens and at most one minor piece on the board: nobody can mate.
    if pos[PAWN] | pos[6 + PAWN] | pos[ROOK] | pos[6 + ROOK] | pos[QUEEN] | pos[6 + QUEEN]:
        return False
    minors = pos[KNIGHT] | pos[6 + KNIGHT] | pos[BISHOP] | pos[6 + BISHOP]
    return popcount(minors) <= 1


@njit(cache=False)
def has_non_pawn(pos, side):
    base = side * 6
    return (pos[base + 1] | pos[base + 2] | pos[base + 3] | pos[base + 4]) != ZERO


@njit(cache=False)
def search(
    stack, ply, depth, alpha, beta, allow_null, ctx, ctxf, tt_keys, tt_data, killers, history,
    movebuf, scorebuf, game_keys, acc, nn, use_nn, aux, counters,
):
    check_time(ctx, ctxf)
    if ctx[C_ABORT]:
        return 0
    pos = stack[ply]
    side = np.int64(pos[I_SIDE])

    checked = in_check(pos)
    if checked:
        depth += 1

    if depth <= 0:
        return quiesce(
            stack, ply, alpha, beta, ctx, ctxf, movebuf, scorebuf, acc, nn, use_nn, tt_keys, tt_data
        )
    if ply >= MAX_PLY - 1:
        return eval_pos(pos, acc[ply], nn, use_nn)

    # Draws.
    if np.int64(pos[I_HALF]) >= 100:
        return 0
    if is_repetition(stack, ply, game_keys, ctx[C_GAME_KEYS]):
        return 0
    if insufficient_material(pos):
        return 0

    # Mate distance pruning.
    if alpha < -MATE + ply:
        alpha = -MATE + ply
    if beta > MATE - ply - 1:
        beta = MATE - ply - 1
    if alpha >= beta:
        return alpha

    key = pos[I_KEY]
    data = tt_probe(tt_keys, tt_data, key)
    tt_m = 0
    if data != 0:
        tt_m = tt_move(data)
        if tt_depth(data) >= depth:
            s = tt_score(data)
            f = tt_flag(data)
            if f == TT_EXACT:
                return s
            if f == TT_LOWER and s >= beta:
                return s
            if f == TT_UPPER and s <= alpha:
                return s

    pv_node = beta - alpha > 1

    # In check the static eval is not used for pruning, so skip the evaluation.
    static = 0 if checked else eval_pos(pos, acc[ply], nn, use_nn)
    aux[A_STATIC, ply] = static if not checked else aux[A_STATIC, ply - 2] if ply >= 2 else 0
    # Improving: our static eval is better than it was two plies ago (same side to move).
    improving = ply < 2 or checked or static > aux[A_STATIC, ply - 2]

    # Reverse futility pruning: far above beta at low depth.
    if not pv_node and not checked and depth <= 3:
        margin = 120 * depth if improving else 80 * depth
        if static - margin >= beta:
            return static

    # Null-move pruning.
    if (
        allow_null
        and not pv_node
        and not checked
        and depth >= 3
        and static >= beta
        and has_non_pawn(pos, side)
    ):
        make_null(pos, stack[ply + 1])
        if use_nn:
            acc_copy(acc[ply], acc[ply + 1])
        r = 2 + depth // 4
        aux[A_MOVE, ply + 1] = 0
        score = -search(
            stack, ply + 1, depth - 1 - r, -beta, -beta + 1, False, ctx, ctxf, tt_keys,
            tt_data, killers, history, movebuf, scorebuf, game_keys, acc, nn, use_nn, aux,
            counters,
        )
        if ctx[C_ABORT]:
            return 0
        if score >= beta and score < MATE_BOUND:
            return beta

    moves = movebuf[ply]
    scores = scorebuf[ply]
    n = gen_moves(pos, moves, False)
    prev = aux[A_MOVE, ply]
    counter_m = counters[side, mv_from(prev), mv_to(prev)] if prev != 0 else 0
    for i in range(n):
        scores[i] = score_move(moves[i], tt_m, killers, history, ply, side, counter_m)

    best_score = -INF
    best_move = 0
    orig_alpha = alpha
    legal = 0
    fut_margin = 150 * depth if improving else 100 * depth
    futile = not pv_node and not checked and depth <= 2 and static + fut_margin <= alpha

    for i in range(n):
        m = pick_next(moves, scores, n, i)
        if not make_move(pos, stack[ply + 1], m):
            continue
        if use_nn:
            acc_apply_move(pos, stack[ply + 1], m, acc[ply], acc[ply + 1], nn)
        legal += 1
        is_cap = mv_captured(m) != NO_PIECE
        is_promo = mv_promo(m) != 0
        gives_check = in_check(stack[ply + 1])

        # Futility pruning of quiet moves that cannot raise alpha.
        if futile and legal > 1 and not is_cap and not is_promo and not gives_check:
            continue

        reduce = 0
        if depth >= 3 and legal > 2 and not is_cap and not is_promo and not gives_check:
            reduce = LMR_TABLE[min(depth, 63), min(legal, 63)]
            if pv_node:
                reduce -= 1
            if not improving:
                reduce += 1
            if m == killers[ply, 0] or m == killers[ply, 1] or m == counter_m:
                reduce -= 1
            if history[side, mv_from(m), mv_to(m)] > 2000:
                reduce -= 1
            if reduce < 0:
                reduce = 0
            if reduce > depth - 2:
                reduce = depth - 2

        aux[A_MOVE, ply + 1] = m
        if legal == 1:
            score = -search(
                stack, ply + 1, depth - 1, -beta, -alpha, True, ctx, ctxf, tt_keys, tt_data,
                killers, history, movebuf, scorebuf, game_keys, acc, nn, use_nn, aux, counters,
            )
        else:
            score = -search(
                stack, ply + 1, depth - 1 - reduce, -alpha - 1, -alpha, True, ctx, ctxf,
                tt_keys, tt_data, killers, history, movebuf, scorebuf, game_keys, acc, nn, use_nn,
                aux, counters,
            )
            if not ctx[C_ABORT] and score > alpha and (reduce > 0 or score < beta):
                score = -search(
                    stack, ply + 1, depth - 1, -beta, -alpha, True, ctx, ctxf, tt_keys,
                    tt_data, killers, history, movebuf, scorebuf, game_keys, acc, nn, use_nn,
                    aux, counters,
                )
        if ctx[C_ABORT]:
            return 0

        if score > best_score:
            best_score = score
            best_move = m
        if score > alpha:
            alpha = score
            if alpha >= beta:
                if not is_cap:
                    if killers[ply, 0] != m:
                        killers[ply, 1] = killers[ply, 0]
                        killers[ply, 0] = m
                    history[side, mv_from(m), mv_to(m)] += depth * depth
                    if prev != 0:
                        counters[side, mv_from(prev), mv_to(prev)] = m
                    if history[side, mv_from(m), mv_to(m)] > 1_000_000:
                        for a in range(64):
                            for b in range(64):
                                history[side, a, b] //= 2
                break

    if legal == 0:
        return -MATE + ply if checked else 0

    if best_score <= orig_alpha:
        flag = TT_UPPER
    elif best_score >= beta:
        flag = TT_LOWER
    else:
        flag = TT_EXACT
    tt_store(tt_keys, tt_data, key, depth, flag, best_score, best_move)
    return best_score


@njit(cache=False)
def search_root(
    stack, depth, alpha, beta, ctx, ctxf, tt_keys, tt_data, killers, history, movebuf,
    scorebuf, game_keys, root_moves, n_root, root_scores, prev_best, acc, nn, use_nn, aux,
    counters,
):
    """One iteration at the root. Returns (best_move, score). Aborts leave ctx[C_ABORT] set."""
    pos = stack[0]
    side = np.int64(pos[I_SIDE])
    for i in range(n_root):
        root_scores[i] = score_move(root_moves[i], prev_best, killers, history, 0, side, 0)
    aux[A_STATIC, 0] = eval_pos(pos, acc[0], nn, use_nn)
    aux[A_MOVE, 0] = 0
    best_move = root_moves[0]
    best_score = -INF
    for i in range(n_root):
        m = pick_next(root_moves, root_scores, n_root, i)
        make_move(pos, stack[1], m)  # root moves are pre-filtered legal
        if use_nn:
            acc_apply_move(pos, stack[1], m, acc[0], acc[1], nn)
        aux[A_MOVE, 1] = m
        if i == 0:
            score = -search(
                stack, 1, depth - 1, -beta, -alpha, True, ctx, ctxf, tt_keys, tt_data, killers,
                history, movebuf, scorebuf, game_keys, acc, nn, use_nn, aux, counters,
            )
        else:
            score = -search(
                stack, 1, depth - 1, -alpha - 1, -alpha, True, ctx, ctxf, tt_keys, tt_data,
                killers, history, movebuf, scorebuf, game_keys, acc, nn, use_nn, aux, counters,
            )
            if not ctx[C_ABORT] and score > alpha and score < beta:
                score = -search(
                    stack, 1, depth - 1, -beta, -alpha, True, ctx, ctxf, tt_keys, tt_data,
                    killers, history, movebuf, scorebuf, game_keys, acc, nn, use_nn, aux, counters,
                )
        if ctx[C_ABORT]:
            return best_move, best_score
        if score > best_score:
            best_score = score
            best_move = m
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    tt_store(tt_keys, tt_data, pos[I_KEY], depth, TT_EXACT, best_score, best_move)
    return best_move, best_score


@njit(cache=False)
def legal_moves(pos, buf, tmp):
    """All legal moves into buf; returns count."""
    scratch = np.zeros(MAX_MOVES, dtype=np.int64)
    n = gen_moves(pos, scratch, False)
    k = 0
    for i in range(n):
        if make_move(pos, tmp, scratch[i]):
            buf[k] = scratch[i]
            k += 1
    return k


@njit(cache=False)
def perft(pos, depth, stack, ply):
    if depth == 0:
        return 1
    buf = np.zeros(MAX_MOVES, dtype=np.int64)
    n = gen_moves(pos, buf, False)
    total = 0
    for i in range(n):
        if make_move(pos, stack[ply + 1], buf[i]):
            total += perft(stack[ply + 1], depth - 1, stack, ply + 1)
    return total


# --------------------------------------------------------------------------------------------
# Python-side helpers
# --------------------------------------------------------------------------------------------

PIECE_CHARS = "PNBRQK"
PROMO_CHARS = " nbrq"


def pos_from_fen(fen: str) -> np.ndarray:
    pos = np.zeros(POS_LEN, dtype=np.uint64)
    parts = fen.split()
    rows = parts[0].split("/")
    for r, row in enumerate(rows):
        rank = 7 - r
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
                continue
            colour = WHITE if ch.isupper() else BLACK
            p = PIECE_CHARS.index(ch.upper())
            pos[p + 6 * colour] |= U(1 << (rank * 8 + f))
            f += 1
    pos[I_SIDE] = U(0 if parts[1] == "w" else 1)
    rights = 0
    if len(parts) > 2:
        for ch in parts[2]:
            rights |= {"K": 1, "Q": 2, "k": 4, "q": 8}.get(ch, 0)
    pos[I_CASTLE] = U(rights)
    ep = 64
    if len(parts) > 3 and parts[3] != "-":
        ep = (ord(parts[3][0]) - ord("a")) + 8 * (int(parts[3][1]) - 1)
        side = int(pos[I_SIDE])
        if not (int(PAWN_ATT[1 - side, ep]) & int(pos[PAWN + 6 * side])):
            ep = 64
    pos[I_EP] = U(ep)
    pos[I_HALF] = U(int(parts[4]) if len(parts) > 4 else 0)
    pos[I_KEY] = compute_key(pos)
    return pos


def sq_name(sq: int) -> str:
    return "abcdefgh"[sq & 7] + str((sq >> 3) + 1)


def move_to_uci(m: int) -> str:
    s = sq_name(mv_from(m)) + sq_name(mv_to(m))
    promo = mv_promo(m)
    if promo:
        s += PROMO_CHARS[promo]
    return s


def uci_to_move(pos: np.ndarray, uci: str) -> int:
    buf = np.zeros(MAX_MOVES, dtype=np.int64)
    tmp = np.zeros(POS_LEN, dtype=np.uint64)
    n = legal_moves(pos, buf, tmp)
    for i in range(n):
        if move_to_uci(int(buf[i])) == uci:
            return int(buf[i])
    raise ValueError(f"no legal move {uci}")


class Engine:
    """Owns the search memory and drives iterative deepening under a clock."""

    def __init__(self) -> None:
        self.stack = np.zeros((MAX_PLY + 2, POS_LEN), dtype=np.uint64)
        self.tt_keys = np.zeros(TT_SIZE, dtype=np.uint64)
        self.tt_data = np.zeros(TT_SIZE, dtype=np.int64)
        self.killers = np.zeros((MAX_PLY + 2, 2), dtype=np.int64)
        self.history = np.zeros((2, 64, 64), dtype=np.int64)
        self.movebuf = np.zeros((MAX_PLY + 2, MAX_MOVES), dtype=np.int64)
        self.scorebuf = np.zeros((MAX_PLY + 2, MAX_MOVES), dtype=np.int64)
        self.game_keys = np.zeros(1024, dtype=np.uint64)
        self.n_game = 0
        self.ctx = np.zeros(8, dtype=np.int64)
        self.ctxf = np.zeros(2, dtype=np.float64)
        self.root_moves = np.zeros(MAX_MOVES, dtype=np.int64)
        self.root_scores = np.zeros(MAX_MOVES, dtype=np.int64)
        self.tmp = np.zeros(POS_LEN, dtype=np.uint64)
        self.aux = np.zeros((2, MAX_PLY + 2), dtype=np.int64)
        self.counters = np.zeros((2, 64, 64), dtype=np.int64)
        self.log: list[str] = []
        # Neural evaluation: off until load_nn() succeeds.
        self.use_nn = 0
        self.hidden = 8
        self.nn = (
            np.zeros((N_BUCKETS * 768, self.hidden), dtype=np.int16),
            np.zeros(self.hidden, dtype=np.int16),
            np.zeros(2 * self.hidden, dtype=np.int16),
            np.array([255, 64, 400, 0], dtype=np.int64),
        )
        self.acc = np.zeros((MAX_PLY + 2, 2, self.hidden), dtype=np.int16)

    def load_nn(self, path: str) -> None:
        d = np.load(path)
        ft_w = np.ascontiguousarray(d["ft_w"], dtype=np.int16)
        if ft_w.shape[0] != N_BUCKETS * 768:
            raise ValueError(f"weights have {ft_w.shape[0]} features, want {N_BUCKETS * 768}")
        ft_b = np.ascontiguousarray(d["ft_b"], dtype=np.int16)
        out_w = np.ascontiguousarray(d["out_w"], dtype=np.int16)
        meta = np.array(
            [int(d["qa"]), int(d["qb"]), int(d["scale"]), int(d["out_b"])], dtype=np.int64
        )
        self.hidden = ft_w.shape[1]
        self.nn = (ft_w, ft_b, out_w, meta)
        self.acc = np.zeros((MAX_PLY + 2, 2, self.hidden), dtype=np.int16)
        self.use_nn = 1

    def evaluate(self, pos: np.ndarray) -> int:
        """Static evaluation of a position with whichever evaluator is active."""
        if self.use_nn:
            acc_refresh(pos, self.acc[0], self.nn)
            return int(evaluate_nn(pos, self.acc[0], self.nn))
        return int(evaluate(pos))

    def new_game(self) -> None:
        self.tt_keys[:] = 0
        self.tt_data[:] = 0
        self.killers[:] = 0
        self.history[:] = 0
        self.counters[:] = 0
        self.n_game = 0

    def record(self, pos: np.ndarray) -> None:
        if self.n_game < len(self.game_keys):
            self.game_keys[self.n_game] = pos[I_KEY]
            self.n_game += 1

    def think(
        self, pos: np.ndarray, budget_s: float, max_depth: int = 60, hard_s: float | None = None
    ) -> tuple[str, int, int]:
        """Return (uci, score, depth) for the best move found.

        budget_s is the soft target: no new iteration starts once a fraction of it is spent,
        the fraction growing when the best move just changed or the score fell. hard_s is the
        deadline at which the search is aborted wherever it is.
        """
        start = time.perf_counter()
        self.stack[0] = pos
        if self.use_nn:
            acc_refresh(pos, self.acc[0], self.nn)
        n_root = legal_moves(pos, self.root_moves, self.tmp)
        if n_root == 0:
            raise ValueError("no legal moves")
        if n_root == 1:
            return move_to_uci(int(self.root_moves[0])), 0, 0

        self.killers[:] = 0
        self.history //= 8
        self.ctx[:] = 0
        self.ctx[C_NODE_LIMIT] = 1 << 62
        self.ctx[C_GAME_KEYS] = self.n_game
        self.ctxf[0] = start + (hard_s if hard_s is not None else budget_s)

        best = int(self.root_moves[0])
        best_score = 0
        completed = 0
        depth = 1
        window = 25
        alpha, beta = -INF, INF
        while depth <= max_depth:
            m, score = search_root(
                self.stack, depth, alpha, beta, self.ctx, self.ctxf, self.tt_keys, self.tt_data,
                self.killers, self.history, self.movebuf, self.scorebuf, self.game_keys,
                self.root_moves, n_root, self.root_scores, best, self.acc, self.nn, self.use_nn,
                self.aux, self.counters,
            )
            if self.ctx[C_ABORT]:
                self.log.append(f"time up in depth {depth} nodes {int(self.ctx[C_NODES])}")
                break
            m = int(m)
            score = int(score)
            # Aspiration window handling.
            if score <= alpha:
                alpha = -INF
                continue
            if score >= beta:
                beta = INF
                continue
            unstable = (m != best and completed > 0) or score < best_score - 30
            best, best_score, completed = m, score, depth
            elapsed = time.perf_counter() - start
            self.log.append(
                f"depth {depth} score {score} nodes {int(self.ctx[C_NODES])} "
                f"time {elapsed * 1000:.0f}ms best {move_to_uci(best)}"
            )
            if abs(score) >= MATE_BOUND:
                break
            if elapsed > budget_s * (0.8 if unstable else 0.4):
                break
            depth += 1
            alpha, beta = score - window, score + window
        return move_to_uci(best), best_score, completed
