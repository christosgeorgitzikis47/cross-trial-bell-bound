"""
ΑΝΩ ΟΡΙΟ σε παραβίαση measurement independence.

Μέθοδος: ένεση τεχνητής συσχέτισης αποτελέσματος–ΜΕΛΛΟΝΤΙΚΗΣ ρύθμισης.
Με πιθανότητα ε, το OA[i] αντιγράφει το SB[i+k] (ως 0/1). Σαρώνουμε το ε
και βρίσκουμε το μικρότερο που ανιχνεύεται πάνω από το null max.

Αποτέλεσμα: «αποκλείουμε παραβίαση measurement independence πάνω από ε_min».

ΠΡΟΣΟΧΗ: η ένεση αλλάζει το περιθώριο του OA (από 0,69% προς 50% κατά ε),
οπότε το null ΞΑΝΑΥΠΟΛΟΓΙΖΕΤΑΙ για κάθε ε. Αλλιώς θα μετρούσαμε τη
μετατόπιση του περιθωρίου, όχι τη συσχέτιση.
"""
import argparse, json
import numpy as np
from lag_test import mi, align

N_NULL = 200


def inject(OA, SB, k, eps, rng):
    """Με πιθανότητα eps: OA[i] <- (SB[i+k]==2). Επιστρέφει ευθυγραμμισμένα o,s."""
    o, s = align(OA, SB, k)
    o = o.copy()
    hit = rng.random(len(o)) < eps
    o[hit] = (s[hit] == 2).astype(o.dtype)
    return o, s


def null_max(o, s, n_null, seed):
    rng = np.random.default_rng(seed)
    sh = s.copy()
    v = np.empty(n_null)
    for i in range(n_null):
        rng.shuffle(sh)
        v[i] = mi(o, sh)
    return v


def detect(OA, SB, k, eps, n_null, seed):
    rng = np.random.default_rng(seed)
    o, s = inject(OA, SB, k, eps, rng)
    m = mi(o, s)
    nd = null_max(o, s, n_null, seed + 1)
    return m, nd.mean(), nd.max(), nd.std(ddof=1), bool(m > nd.max()), float(o.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--nulls", type=int, default=N_NULL)
    a = ap.parse_args()

    d = np.load(a.path)
    OA, SB = d['OA'], d['SB']
    print(f"Trials: {len(OA):,}   ρυθμός click OA: {OA.mean()*100:.4f}%")
    print(f"null ανά σημείο: {a.nulls} ανακατέματα\n")

    results = {}
    for k in (1, 10, 100):
        print("=" * 74)
        print(f"k = {k}   (OA[i] αντιγράφει SB[i+{k}] με πιθανότητα ε)")
        print("=" * 74)
        print(f"{'ε':>10} {'MI':>12} {'null max':>12} {'σ':>9} {'ρυθμός OA':>10} {'ανιχν.':>8}")
        rows = []
        # χονδρική σάρωση
        grid = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
        first = None
        for eps in grid:
            m, mu, mx, sd, ok, rate = detect(OA, SB, k, eps, a.nulls, 1000 + k)
            sig = (m - mu) / sd
            print(f"{eps:>10.1e} {m:12.4e} {mx:12.4e} {sig:>+9.1f} "
                  f"{rate*100:>9.4f}% {('ΝΑΙ' if ok else '-'):>8}")
            rows.append({"eps": eps, "mi": float(m), "null_max": float(mx),
                         "sigma": float(sig), "detected": ok, "oa_rate": rate})
            if ok and first is None:
                first = eps
                break

        # εκλέπτυνση με διχοτόμηση ανάμεσα στο τελευταίο «όχι» και το πρώτο «ναι»
        if first is not None:
            lo = max([r["eps"] for r in rows if not r["detected"]], default=first / 10)
            hi = first
            print(f"\n  εκλέπτυνση ανάμεσα σε {lo:.2e} και {hi:.2e}:")
            for _ in range(5):
                mid = (lo * hi) ** 0.5          # γεωμετρικό μέσο
                m, mu, mx, sd, ok, rate = detect(OA, SB, k, mid, a.nulls, 2000 + k)
                sig = (m - mu) / sd
                print(f"{mid:>10.2e} {m:12.4e} {mx:12.4e} {sig:>+9.1f} "
                      f"{rate*100:>9.4f}% {('ΝΑΙ' if ok else '-'):>8}")
                rows.append({"eps": mid, "mi": float(m), "null_max": float(mx),
                             "sigma": float(sig), "detected": ok, "oa_rate": rate})
                if ok:
                    hi = mid
                else:
                    lo = mid
            eps_min = hi
            print(f"\n  ε_min ≈ {eps_min:.2e}   (ανάμεσα σε {lo:.2e} και {hi:.2e})")
        else:
            eps_min = None
            print(f"\n  ΚΑΜΙΑ ανίχνευση μέχρι ε = {grid[-1]:.1e}")
        results[f"k={k}"] = {"rows": rows, "eps_min": eps_min}
        print()

    print("=" * 74)
    print("ΣΥΝΟΨΗ — ΑΝΩ ΟΡΙΟ")
    print("=" * 74)
    for k in (1, 10, 100):
        e = results[f"k={k}"]["eps_min"]
        s = f"{e:.2e}" if e else "δεν βρέθηκε"
        print(f"  k={k:<4} ε_min = {s}")
    print("\nΕρμηνεία: αποκλείεται παραβίαση measurement independence με ένταση")
    print("πάνω από το ε_min, στο αντίστοιχο lag.")

    with open("lag_limit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nΑποθηκεύτηκε: lag_limit_results.json")


if __name__ == "__main__":
    main()
