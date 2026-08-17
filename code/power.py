import numpy as np, data_gen as D, scratch_test as T

A = 4097
rng = np.random.default_rng(11)

def noisy(seq, p, rng):
    """XOR the structured sequence with Bernoulli(p) noise.
    p=0 -> pure structure, p=0.5 -> pure randomness."""
    return (seq ^ (rng.uniform(size=len(seq)) < p).astype(np.int8)).astype(np.int8)

def h2(p):
    if p <= 0 or p >= 1: return 0.0 if p<=0 else 1.0
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

base_seq = D.thue_morse()          # the easiest possible structure to find
print(f"{'noise p':>8} {'entropy rate':>13} {'search gain (bits)':>20} {'detected?':>11}")
print("-"*56)
results = []
for p in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    s  = noisy(base_seq, p, rng)
    sc = D.scramble(s, A)
    base, rows = T.search(sc, stride_samples=400)
    gain = base - rows[0][3]
    det = "YES" if gain > 50 else "no"
    print(f"{p:8.2f} {h2(p):13.4f} {gain:20.1f} {det:>11}")
    results.append((p, h2(p), gain))
np.save("power_tm.npy", np.array(results))
