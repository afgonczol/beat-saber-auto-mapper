"""Group maps by song, so the train/val/test split never puts one song in two splits.

Two maps are the same song when their audio fingerprints match, or (the fallback)
when their normalized title + artist are equal and their durations are close.
Groups are the connected components of both relations.

The fingerprint follows Haitsma & Kalker (2002): per frame, 32 bits saying whether
the energy difference between neighbouring mel bands grew or shrank since the
previous frame. It survives re-encoding, gain changes and trimming. Matching
works like Shazam-style lookup: an exact 32-bit word shared by two songs votes
for their time offset, and a pair with enough votes at one offset is confirmed by
the bit error rate (BER) over the aligned overlap. Unrelated audio has a BER
near 0.5; the same recording, re-encoded, stays well under 0.3.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

FP_RATE_DIVISOR = 4  # 100 frames/s mel -> 25 fingerprint words/s
FP_SMOOTH_FRAMES = 8  # 80 ms box filter before decimating: trims rarely land on a frame
FP_BANDS = 33  # 33 bands -> 32 band differences -> one uint32 per frame
FP_MEL_LO, FP_MEL_HI = 6, 72  # mel bins used (about 300 Hz .. 6 kHz at 80 bins)


def fingerprint(mel_db: np.ndarray) -> np.ndarray:
    """uint32 words from a (frames, n_mels) log-mel, at 1/FP_RATE_DIVISOR its rate."""
    if mel_db.shape[0] < 4 * FP_RATE_DIVISOR:
        return np.zeros(0, dtype=np.uint32)
    power = 10.0 ** (mel_db[:, FP_MEL_LO:FP_MEL_HI].astype(np.float64) / 10.0)
    c = np.cumsum(np.pad(power, ((1, 0), (0, 0))), axis=0)
    w = FP_SMOOTH_FRAMES
    smooth = (c[w:] - c[:-w]) / w  # row i: mean of frames i .. i + w - 1
    power = smooth[::FP_RATE_DIVISOR]
    edges = np.linspace(0, power.shape[1], FP_BANDS + 1).astype(int)
    bands = np.stack([power[:, a:b].sum(axis=1) for a, b in zip(edges, edges[1:], strict=False)])
    energy = np.log10(bands.T + 1e-10)  # (n, 33)
    d = energy[:, :-1] - energy[:, 1:]  # (n, 32)
    bits = (d[1:] - d[:-1]) > 0  # (n - 1, 32)
    packed = np.ascontiguousarray(np.packbits(bits, axis=1, bitorder="little"))
    return packed.view("<u4").ravel().astype(np.uint32)


def _popcount(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.astype("<u4").view(np.uint8)).reshape(len(x), 32).sum(axis=1)


def bit_error_rate(a: np.ndarray, b: np.ndarray, offset: int) -> tuple[float, int]:
    """BER between ``a[i]`` and ``b[i - offset]`` over their overlap, and its length."""
    lo, hi = max(0, offset), min(len(a), len(b) + offset)
    if hi <= lo:
        return 1.0, 0
    x = np.bitwise_xor(a[lo:hi], b[lo - offset : hi - offset])
    return float(_popcount(x).mean() / 32.0), hi - lo


@dataclass(frozen=True, slots=True)
class FingerprintMatch:
    a: int
    b: int
    offset: int  # frames of b relative to a
    votes: int
    ber: float
    overlap_s: float


def match_fingerprints(
    fps: Sequence[np.ndarray],
    min_votes: int = 8,
    max_ber: float = 0.3,
    min_overlap_s: float = 20.0,
    max_word_maps: int = 20,
) -> list[FingerprintMatch]:
    """Confirmed same-audio pairs among ``fps`` (indices into the list).

    Words shared by more than ``max_word_maps`` songs (silence, steady tones) carry
    no information and are ignored.
    """
    rate = 100.0 / FP_RATE_DIVISOR
    words, owner, frame = [], [], []
    for i, fp in enumerate(fps):
        if len(fp) == 0:
            continue
        w, first = np.unique(fp, return_index=True)  # one vote per word per song
        words.append(w)
        owner.append(np.full(len(w), i, dtype=np.int64))
        frame.append(first.astype(np.int64))
    if not words:
        return []
    w, o, f = np.concatenate(words), np.concatenate(owner), np.concatenate(frame)
    order = np.argsort(w, kind="stable")
    w, o, f = w[order], o[order], f[order]
    starts = np.flatnonzero(np.r_[True, w[1:] != w[:-1]])
    sizes = np.diff(np.r_[starts, len(w)])
    keep = (sizes >= 2) & (sizes <= max_word_maps)
    pa, pb, po = [], [], []
    for size in np.unique(sizes[keep]):
        grp = starts[keep & (sizes == size)]
        idx = grp[:, None] + np.arange(size)[None, :]  # (groups, size)
        ii, jj = np.triu_indices(size, k=1)
        a, b = o[idx[:, ii]].ravel(), o[idx[:, jj]].ravel()
        fa, fb = f[idx[:, ii]].ravel(), f[idx[:, jj]].ravel()
        swap = a > b
        a, b = np.where(swap, b, a), np.where(swap, a, b)
        off = np.where(swap, fb - fa, fa - fb)  # frame in lower index minus frame in higher
        pa.append(a)
        pb.append(b)
        po.append(off)
    if not pa:
        return []
    a, b, off = np.concatenate(pa), np.concatenate(pb), np.concatenate(po)
    # Tolerate one frame of jitter: vote into offset bins of 3 frames. Pack
    # (a, b, offset bin) into one int64 so np.unique stays one-dimensional.
    n, span = len(fps), 1 << 20
    key = (a * n + b) * span + (np.floor_divide(off + 1, 3) + span // 2)
    uniq, counts = np.unique(key, return_counts=True)
    hit = counts >= min_votes
    uniq, counts = uniq[hit], counts[hit]
    best: dict[tuple[int, int], tuple[int, int]] = {}
    for k, c in zip(uniq.tolist(), counts.tolist(), strict=True):
        pair, ob = divmod(k, span)
        ia, ib = divmod(pair, n)
        if c > best.get((ia, ib), (0, 0))[0]:
            best[(ia, ib)] = (c, ob - span // 2)
    out = []
    for (ia, ib), (votes, ob) in sorted(best.items()):
        candidates = [
            (*bit_error_rate(fps[ia], fps[ib], 3 * ob + d), 3 * ob + d) for d in (-1, 0, 1)
        ]
        ber, overlap, offset = min(candidates)
        if ber <= max_ber and overlap / rate >= min_overlap_s:
            out.append(FingerprintMatch(ia, ib, offset, votes, ber, overlap / rate))
    return out


_BRACKETS = re.compile(r"[\(\[\{].*?[\)\]\}]")
_FEAT = re.compile(r"\b(feat|ft|featuring)\b.*$")
_NON_WORD = re.compile(r"[\W_]+")


def normalize_text(s: str) -> str:
    """Case-fold, drop bracketed parts and 'feat. ...', keep only words."""
    s = unicodedata.normalize("NFKC", s or "").casefold()
    s = _BRACKETS.sub(" ", s)
    s = _FEAT.sub(" ", s)
    return _NON_WORD.sub(" ", s).strip()


def metadata_key(title: str, artist: str) -> str | None:
    """Fallback identity of a song; None when the title normalizes to nothing."""
    t = normalize_text(title)
    return f"{t}|{normalize_text(artist)}" if t else None


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


@dataclass(slots=True)
class SongGroups:
    group: list[str]  # per input map: the group id (smallest key in the group)
    fingerprint_links: int  # unions made by fingerprint matches
    metadata_links: int  # extra unions made by the title/artist/duration fallback
    matches: list[FingerprintMatch]


def group_songs(
    keys: Sequence[str],
    fingerprints: Sequence[np.ndarray | None],
    titles: Sequence[str],
    artists: Sequence[str],
    durations_s: Sequence[float],
    max_duration_diff_s: float = 3.0,
    **match_kwargs,
) -> SongGroups:
    n = len(keys)
    uf = _UnionFind(n)
    fps = [fp if fp is not None else np.zeros(0, dtype=np.uint32) for fp in fingerprints]
    matches = match_fingerprints(fps, **match_kwargs)
    fp_links = sum(uf.union(m.a, m.b) for m in matches)
    meta_links = 0
    by_meta: dict[str, list[int]] = {}
    for i in range(n):
        k = metadata_key(titles[i], artists[i])
        if k is not None:
            by_meta.setdefault(k, []).append(i)
    for members in by_meta.values():
        for x, i in enumerate(members):
            for j in members[x + 1 :]:
                if abs(durations_s[i] - durations_s[j]) <= max_duration_diff_s:
                    meta_links += uf.union(i, j)
    roots: dict[int, str] = {}
    for i in range(n):
        r = uf.find(i)
        roots[r] = min(roots.get(r, keys[i]), keys[i])
    return SongGroups(
        group=[roots[uf.find(i)] for i in range(n)],
        fingerprint_links=fp_links,
        metadata_links=meta_links,
        matches=matches,
    )


SPLITS = ("train", "val", "test")


def split_for(group_id: str, val: float = 0.05, test: float = 0.05) -> str:
    """Deterministic split of a song group from a hash of its id."""
    h = int.from_bytes(hashlib.sha1(group_id.encode()).digest()[:8], "big") / 2**64
    if h < 1.0 - val - test:
        return "train"
    return "val" if h < 1.0 - test else "test"
