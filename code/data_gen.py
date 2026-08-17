import numpy as np, os, hashlib
from math import gcd

N = 32768

def bits_from_bytes(b, n):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))[:n].astype(np.int8)

# ---- positive controls: genuinely low-complexity sequences ----

def melody(n=N):
    """The literal DJ case: a repeating musical phrase, 8 bits/note."""
    phrase = [60,62,64,65,67,65,64,62, 60,64,67,72,67,64,62,60]   # a tune
    notes = []
    rng = np.random.default_rng(7)
    while len(notes)*8 < n:
        v = rng.integers(-1, 2)          # small variation between repeats
        notes += [p + int(v) for p in phrase]
    arr = np.array(notes[: n//8], dtype=np.uint8)
    return np.unpackbits(arr)[:n].astype(np.int8)

def thue_morse(n=N):
    i = np.arange(n)
    out = np.zeros(n, dtype=np.int8)
    for b in range(20):
        out ^= ((i >> b) & 1).astype(np.int8)
    return out

def logistic(n=N):
    x = 0.4213
    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        x = 3.9 * x * (1 - x)
        out[i] = 1 if x > 0.5 else 0
    return out

# ---- negative control: cryptographically random ----

def crypto(n=N, seed=None):
    if seed is None:
        return bits_from_bytes(os.urandom(n // 8 + 1), n)
    b = b''.join(hashlib.sha512(f"{seed}:{i}".encode()).digest()
                 for i in range(n // 512 + 1))
    return bits_from_bytes(b, n)

# ---- the scramble (known ground truth) ----

def scramble(bits, a):
    n = len(bits)
    assert gcd(a, n) == 1
    return bits[(a * np.arange(n)) % n]

def unscramble_exponent(a, n):
    """The stride that undoes stride a is a^{-1} mod n."""
    return pow(a, -1, n)

# ---- quantum eraser simulation ----

def eraser(n=N, seed=3):
    """Signal photon position (binarised) + idler which-path/erase label.
    Marginal is flat; sorting BY THE LABEL reveals fringes.
    Returns (outcomes, labels)."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)                  # R+ / R- (random, from idler)
    x = rng.uniform(0, 2*np.pi, n)                  # detector position phase
    # P(click|label) = (1 + s*cos x)/2  with s=+1 for R+, -1 for R-
    s = np.where(labels == 1, 1.0, -1.0)
    p = (1 + s*np.cos(x)) / 2
    out = (rng.uniform(size=n) < p).astype(np.int8)
    order = np.argsort(x)                           # detector position ordering
    return out[order], labels[order]
