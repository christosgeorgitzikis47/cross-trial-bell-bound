"""
DJ-scratch test: does a low-description-length reordering reveal hidden structure?

Formal question: min over pi in P of [ |compress(pi(x))| + K(pi) ]  vs  |compress(x)|
where P is a family of permutations each describable in few bits AND
independent of the outcome VALUES (only functions of the index/time label).

The value-independence constraint is essential: sorting is a permutation
that compresses ANY sequence, but it requires reading the outcomes.
"""
import numpy as np
import zlib, bz2, lzma, math
from math import gcd

# ---------- compression / entropy measures ----------

def pack(bits):
    return np.packbits(bits.astype(np.uint8)).tobytes()

def comp_bits(bits):
    """Best compressed length in bits across standard compressors."""
    b = pack(bits)
    cands = [len(zlib.compress(b, 9)),
             len(bz2.compress(b, 9)),
             len(lzma.compress(b, preset=9))]
    return 8 * min(cands)

def block_entropy_rate(bits, k):
    """Order-k conditional empirical entropy rate (bits/symbol), Miller-Madow corrected.
    Far more sensitive than zlib for binary streams with short-range structure."""
    n = len(bits)
    if n <= k + 1:
        return 1.0
    # build k-mer context indices
    ctx = np.zeros(n - k, dtype=np.int64)
    for j in range(k):
        ctx = (ctx << 1) | bits[j:n - k + j]
    nxt = bits[k:]
    M = 1 << k
    idx = ctx * 2 + nxt
    counts = np.bincount(idx, minlength=2 * M).reshape(M, 2).astype(float)
    tot = counts.sum(axis=1)
    nz = tot > 0
    p = np.zeros_like(counts)
    p[nz] = counts[nz] / tot[nz, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        lg = np.where(p > 0, np.log2(p), 0.0)
    H = -(counts * lg).sum() / tot.sum()
    # Miller-Madow bias correction
    nonzero_cells = (counts > 0).sum()
    H += (nonzero_cells - M) / (2 * np.log(2) * tot.sum())
    return H

def best_entropy_rate(bits, kmax=12):
    """Lowest entropy rate found over orders 1..kmax, with an MDL penalty for
    the model size (2^k parameters) so high k can't win by overfitting."""
    n = len(bits)
    best = 1.0
    bestk = 0
    for k in range(1, kmax + 1):
        if (1 << k) * 8 > n:   # need >=8 samples per context on average
            break
        H = block_entropy_rate(bits, k)
        # MDL: model costs (2^k)*0.5*log2(n) bits, amortised per symbol
        pen = (1 << k) * 0.5 * math.log2(n) / n
        if H + pen < best:
            best, bestk = H + pen, k
    return best, bestk

def comp_bits_fast(bits):
    b = pack(bits)
    return 8 * min(len(zlib.compress(b, 9)), len(bz2.compress(b, 9)))

def score(bits, fast=False):
    """Composite: bits-per-symbol, lower = more structure. Uses whichever
    of (compressor, Markov-MDL) finds more structure."""
    n = len(bits)
    c = comp_bits_fast(bits) if fast else comp_bits(bits)
    return min(c / n, best_entropy_rate(bits)[0])

# ---------- permutation families (value-independent, low description length) ----------

def perm_identity(n):
    yield ("identity", 0.0, np.arange(n))

def perm_stride(n, samples=None):
    """y[i] = x[(a*i) mod n], a coprime to n.  Cost: log2(#valid a) bits."""
    valid = [a for a in range(3, n, 2) if gcd(a, n) == 1]
    if samples is not None and len(valid) > samples:
        rng = np.random.default_rng(0)
        valid = list(rng.choice(valid, samples, replace=False))
    cost = math.log2(max(len(valid), 1))
    i = np.arange(n)
    for a in valid:
        yield (f"stride a={a}", cost, (a * i) % n)

def perm_blockrev(n):
    """Reverse each block of size b (b a power of 2).

    Αν το b ΔΕΝ διαιρεί το n, το τελευταίο ημιτελές μπλοκ μένει ταυτοτικό.
    Χωρίς αυτό η μετάθεση δείχνει εκτός ορίων (index >= n) και δεν είναι
    αμφιμονοσήμαντη. Για n δύναμη του 2 δεν αλλάζει τίποτα: full == n.
    """
    bs = [1 << e for e in range(1, int(math.log2(n)))]
    cost = math.log2(len(bs))
    i = np.arange(n)
    for b in bs:
        full = (n // b) * b
        p = i.copy()
        j = i[:full]
        p[:full] = (j // b) * b + (b - 1 - (j % b))
        yield (f"blockrev b={b}", cost, p)

def perm_deinterleave(n):
    """Split into d interleaved streams and concatenate: classic 'scratch' undo."""
    ds = [d for d in range(2, 65) if n % d == 0]
    cost = math.log2(max(len(ds), 1))
    for d in ds:
        p = np.concatenate([np.arange(r, n, d) for r in range(d)])
        yield (f"deinterleave d={d}", cost, p)

def perm_bitrev(n):
    """Bit-reversal permutation (n a power of 2). Cost: ~0 bits."""
    k = int(math.log2(n))
    if (1 << k) != n:
        return
    i = np.arange(n)
    r = np.zeros(n, dtype=np.int64)
    for b in range(k):
        r |= ((i >> b) & 1) << (k - 1 - b)
    yield ("bitreverse", 0.0, r)

def perm_reverse(n):
    yield ("time-reverse", 0.0, np.arange(n)[::-1])

def perm_random(n, count=200, seed=1):
    """High-K(pi) controls: shows what an UNCONSTRAINED search finds by luck."""
    rng = np.random.default_rng(seed)
    cost = math.lgamma(n + 1) / math.log(2)   # log2(n!)
    for j in range(count):
        yield (f"random#{j}", cost, rng.permutation(n))

def family(n, stride_samples=3000):
    yield from perm_identity(n)
    yield from perm_reverse(n)
    yield from perm_bitrev(n)
    yield from perm_blockrev(n)
    yield from perm_deinterleave(n)
    yield from perm_stride(n, stride_samples)

# ---------- the search ----------

def search(bits, include_random=False, stride_samples=3000):
    n = len(bits)
    base = score(bits) * n            # baseline description length in bits
    rows = []
    gen = family(n, stride_samples)
    if include_random:
        import itertools
        gen = itertools.chain(gen, perm_random(n))
    for name, cost, p in gen:
        s = score(bits[p], fast=True) * n
        rows.append((name, s, cost, s + cost, p))
    rows.sort(key=lambda r: r[3])
    # rescore finalists with the full (slower) compressor set
    out = []
    for name, s, cost, tot, p in rows[:15]:
        s2 = score(bits[p]) * n
        out.append((name, s2, cost, s2 + cost))
    out.sort(key=lambda r: r[3])
    out += [(r[0], r[1], r[2], r[3]) for r in rows[15:]]
    return base, out
