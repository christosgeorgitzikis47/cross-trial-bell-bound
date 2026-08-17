"""
Επαλήθευση των κατεβασμένων παλμών CURBy.

Ελέγχει: base64 -> zlib -> σωστό dtype -> λογικό πλήθος trials,
και διασταυρώνει με τα μεταδεδομένα του manifest.
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
        raise ValueError(f"το μήκος {len(d)} δεν διαιρείται με 4")
    return np.frombuffer(d, dtype=np.uint8).reshape(-1, 4), len(raw), len(d)


print(f"{'γύρος':>7} {'κατέβ. MB':>10} {'αποσυμπ. MB':>12} {'TRIALS':>12} "
      f"{'nTrialsNeeded':>14} {'stopCriteria':>13}")
print("-" * 76)

all_cols = []
for p in sorted(man["pulses"], key=lambda x: x["round"]):
    path = os.path.join(HERE, p["file"])
    arr, nraw, ndec = load_trials(path)
    all_cols.append((p["round"], arr))
    print(f"{p['round']:>7} {nraw/1e6:>10.2f} {ndec/1e6:>12.2f} {len(arr):>12,} "
          f"{p['nTrialsNeeded']:>14,} {p['stoppingCriteria']:>13,}")

print("\n=== ΔΟΜΗ ΕΓΓΡΑΦΗΣ (4 bytes ανά trial) ===")
rnd, arr = all_cols[-1]
print(f"δείγμα από γύρο {rnd}, πρώτες 8 εγγραφές:")
for row in arr[:8]:
    print("   ", list(row))
print()
for c in range(4):
    vals, counts = np.unique(arr[:, c], return_counts=True)
    frac = counts / len(arr)
    desc = "  ".join(f"{v}:{f*100:.2f}%" for v, f in zip(vals, frac))
    print(f"στήλη {c}: τιμές {list(vals)}   κατανομή -> {desc}")

print("\n=== ΣΥΝΕΠΕΙΑ ΜΕΤΑΞΥ ΠΑΛΜΩΝ ===")
for rnd, arr in all_cols:
    cols = [tuple(np.unique(arr[:, c]).tolist()) for c in range(4)]
    print(f"  γύρος {rnd}: {len(arr):>11,} trials   τιμές ανά στήλη: {cols}")

print("\n=== ΕΛΕΓΧΟΣ ΑΚΕΡΑΙΟΤΗΤΑΣ (sha256 από manifest) ===")
import hashlib
for p in sorted(man["pulses"], key=lambda x: x["round"]):
    path = os.path.join(HERE, p["file"])
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    ok = h == p["sha256"]
    print(f"  γύρος {p['round']}: {'ΟΚ' if ok else 'ΑΠΟΤΥΧΙΑ'}  {h[:16]}…")
