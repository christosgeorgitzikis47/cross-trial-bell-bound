import numpy as np, scratch_test as T
from math import gcd

def tm(n):
    i = np.arange(n); out = np.zeros(n, dtype=np.int8)
    for b in range(24): out ^= ((i >> b) & 1).astype(np.int8)
    return out

def h2(p):
    return 1.0 if p>=0.5 else (0.0 if p<=0 else -p*np.log2(p)-(1-p)*np.log2(1-p))

def coprime_a(n):
    a = n//8 + 1
    while gcd(a, n) != 1: a += 2
    return a

print(f"{'N (trials)':>12} {'threshold p':>12} {'min detectable h':>18}")
print("-"*46)
rows=[]
for N in [4096, 8192, 16384, 32768, 65536, 131072]:
    seq = tm(N); A = coprime_a(N); rng = np.random.default_rng(5)
    thr = 0.0
    for p in [0.20,0.25,0.30,0.325,0.35,0.375,0.40,0.425,0.45,0.475]:
        s = (seq ^ (rng.uniform(size=N) < p).astype(np.int8)).astype(np.int8)
        sc = s[(A*np.arange(N)) % N]
        base, r = T.search(sc, stride_samples=150)
        if base - r[0][3] > 50: thr = p
        else: break
    print(f"{N:12,} {thr:12.3f} {h2(thr):18.4f}")
    rows.append((N, thr, h2(thr)))
np.save("scaling.npy", np.array(rows))
