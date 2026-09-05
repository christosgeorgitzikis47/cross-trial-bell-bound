"""
UPPER BOUND on a violation of measurement independence.

Method: inject an artificial outcome-to-FUTURE-setting correlation. With
probability eps, OA[i] copies SB[i+k] (as 0/1). We scan eps and find the
smallest value detected above the null max.

Result: "we exclude a violation of measurement independence above eps_min".

NOTE: the injection changes the margin of OA (from 0.69% towards 50% as eps
grows), so the null is RECOMPUTED for every eps. Otherwise we would be
measuring the shift of the margin, not the correlation.
"""
import argparse, json
import numpy as np
from lag_test import mi, align

N_NULL = 200


def inject(OA, SB, k, eps, rng):
    """With probability eps: OA[i] <- (SB[i+k]==2). Returns aligned o,s."""
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
    print(f"Trials: {len(OA):,}   OA click rate: {OA.mean()*100:.4f}%")
    print(f"null per point: {a.nulls} shuffles\n")

    results = {}
    for k in (1, 10, 100):
        print("=" * 74)
        print(f"k = {k}   (OA[i] copies SB[i+{k}] with probability eps)")
        print("=" * 74)
        print(f"{'eps':>10} {'MI':>12} {'null max':>12} {'sig':>9} {'OA rate':>10} {'detect':>8}")
        rows = []
        # coarse scan
        grid = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
        first = None
        for eps in grid:
            m, mu, mx, sd, ok, rate = detect(OA, SB, k, eps, a.nulls, 1000 + k)
            sig = (m - mu) / sd
            print(f"{eps:>10.1e} {m:12.4e} {mx:12.4e} {sig:>+9.1f} "
                  f"{rate*100:>9.4f}% {('YES' if ok else '-'):>8}")
            rows.append({"eps": eps, "mi": float(m), "null_max": float(mx),
                         "sigma": float(sig), "detected": ok, "oa_rate": rate})
            if ok and first is None:
                first = eps
                break

        # refine by bisection between the last "no" and the first "yes"
        if first is not None:
            lo = max([r["eps"] for r in rows if not r["detected"]], default=first / 10)
            hi = first
            print(f"\n  refining between {lo:.2e} and {hi:.2e}:")
            for _ in range(5):
                mid = (lo * hi) ** 0.5          # geometric mean
                m, mu, mx, sd, ok, rate = detect(OA, SB, k, mid, a.nulls, 2000 + k)
                sig = (m - mu) / sd
                print(f"{mid:>10.2e} {m:12.4e} {mx:12.4e} {sig:>+9.1f} "
                      f"{rate*100:>9.4f}% {('YES' if ok else '-'):>8}")
                rows.append({"eps": mid, "mi": float(m), "null_max": float(mx),
                             "sigma": float(sig), "detected": ok, "oa_rate": rate})
                if ok:
                    hi = mid
                else:
                    lo = mid
            eps_min = hi
            print(f"\n  eps_min ~ {eps_min:.2e}   (between {lo:.2e} and {hi:.2e})")
        else:
            eps_min = None
            print(f"\n  NO detection up to eps = {grid[-1]:.1e}")
        results[f"k={k}"] = {"rows": rows, "eps_min": eps_min}
        print()

    print("=" * 74)
    print("SUMMARY - THE UPPER BOUND")
    print("=" * 74)
    for k in (1, 10, 100):
        e = results[f"k={k}"]["eps_min"]
        s = f"{e:.2e}" if e else "not found"
        print(f"  k={k:<4} eps_min = {s}")
    print("\nReading: a violation of measurement independence stronger than")
    print("eps_min is excluded, at the corresponding lag.")

    with open("lag_limit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: lag_limit_results.json")


if __name__ == "__main__":
    main()
