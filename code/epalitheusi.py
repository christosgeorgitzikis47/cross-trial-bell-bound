"""
Verification of the downloaded CURBy pulses.

Checks: base64 -> zlib -> the right dtype -> a plausible trial count,
and cross-checks against the metadata in the manifest.
"""
import base64, json, os, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
man = json.load(open(os.path.join(HERE, "manifest.json")))


def load_trials(path):
    raw = open(path, "rb").read()
    b = base64.b64decode(raw, validate=False)
    d = zlib.decompress(b)
    if len(d) % 4:
        raise ValueError(f"length {len(d)} is not divisible by 4")
    return np.frombuffer(d, dtype=np.uint8).reshape(-1, 4), len(raw), len(d)


print(f"{'round':>7} {'downl. MB':>10} {'decomp. MB':>12} {'TRIALS':>12} "
      f"{'nTrialsNeeded':>14} {'stopCriteria':>13}")
print("-" * 76)

all_cols = []
for p in sorted(man["pulses"], key=lambda x: x["round"]):
    path = os.path.join(HERE, p["file"])
    arr, nraw, ndec = load_trials(path)
    all_cols.append((p["round"], arr))
    print(f"{p['round']:>7} {nraw/1e6:>10.2f} {ndec/1e6:>12.2f} {len(arr):>12,} "
          f"{p['nTrialsNeeded']:>14,} {p['stoppingCriteria']:>13,}")

print("\n=== RECORD STRUCTURE (4 bytes per trial) ===")
rnd, arr = all_cols[-1]
print(f"sample from round {rnd}, first 8 records:")
for row in arr[:8]:
    print("   ", list(row))
print()
for c in range(4):
    vals, counts = np.unique(arr[:, c], return_counts=True)
    frac = counts / len(arr)
    desc = "  ".join(f"{v}:{f*100:.2f}%" for v, f in zip(vals, frac))
    print(f"column {c}: values {list(vals)}   distribution -> {desc}")

print("\n=== CONSISTENCY ACROSS PULSES ===")
for rnd, arr in all_cols:
    cols = [tuple(np.unique(arr[:, c]).tolist()) for c in range(4)]
    print(f"  round {rnd}: {len(arr):>11,} trials   values per column: {cols}")

print("\n=== INTEGRITY CHECK (sha256 from the manifest) ===")
import hashlib
for p in sorted(man["pulses"], key=lambda x: x["round"]):
    path = os.path.join(HERE, p["file"])
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    ok = h == p["sha256"]
    print(f"  round {p['round']}: {'OK' if ok else 'FAILED'}  {h[:16]}...")
