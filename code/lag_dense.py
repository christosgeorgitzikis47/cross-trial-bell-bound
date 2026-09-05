"""
DENSE lag test: every lag from -50 to +50, plus a few large ones.

Two design decisions, both verified:

1. THE NULL IS REUSED. Shuffling the settings destroys every temporal
   structure, so the null distribution depends only on the margins and on n,
   not on the lag. Verified empirically at lags 0/50/1000/10000: identical
   distributions. n changes by <= 0.07% for |lag| <= 10,000 out of 15,000,000
   trials.

2. MULTIPLE COMPARISONS. 111 lags x 2 pairs = 222 hypotheses are tested. With
   a null max from 100 shuffles we would expect ~2 false positives by chance
   alone. Hence: 2000 shuffles and a Bonferroni threshold.
"""
import argparse, json
import numpy as np
from lag_test import mi, align

N_SHUFFLE = 2000
LAGS = (list(range(-50, 51))
        + [-10000, -3000, -1000, -300, -100, 100, 300, 1000, 3000, 10000])


def build_null(o, s, n_shuffle, seed=777):
    rng = np.random.default_rng(seed)
    sh = s.copy()
    v = np.empty(n_shuffle)
    for i in range(n_shuffle):
        rng.shuffle(sh)
        v[i] = mi(o, sh)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--shuffles", type=int, default=N_SHUFFLE)
    a = ap.parse_args()

    d = np.load(a.path)
    SA, SB, OA, OB = d['SA'], d['SB'], d['OA'], d['OB']
    lags = sorted(LAGS)
    n_tests = len(lags) * 2
    print(f"Trials: {len(SA):,}   lags: {len(lags)}   comparisons: {n_tests}")
    print(f"Null shuffles: {a.shuffles}\n")

    out = {}
    for label, o_full, s_full in [("OA vs SB", OA, SB), ("OB vs SA", OB, SA)]:
        print("=" * 70)
        print(label)
        print("=" * 70)
        nd = build_null(o_full, s_full, a.shuffles)
        mu, sd, mx = nd.mean(), nd.std(ddof=1), nd.max()
        # Bonferroni: threshold at the 1 - 0.05/n_tests quantile of the null
        q = 100 * (1 - 0.05 / n_tests)
        bonf = float(np.percentile(nd, q))
        print(f"null: mean {mu:.4e}  sd {sd:.4e}  max {mx:.4e}")
        print(f"Bonferroni threshold ({q:.3f}th percentile): {bonf:.4e}\n")

        rows = []
        for k in lags:
            oo, ss = align(o_full, s_full, k)
            m = mi(oo, ss)
            rows.append({"lag": k, "mi": float(m),
                         "sigma": float((m - mu) / sd),
                         "above_max": bool(m > mx),
                         "above_bonf": bool(m > bonf)})

        sig = np.array([r["sigma"] for r in rows])
        ks = np.array([r["lag"] for r in rows])
        i_max = int(np.argmax(sig))
        fut = sig[ks > 0]
        pas = sig[ks < 0]
        n_above = sum(r["above_max"] for r in rows)
        n_bonf = sum(r["above_bonf"] for r in rows)

        print(f"  largest sigma overall     : {sig[i_max]:+.2f}  at lag {ks[i_max]}")
        print(f"  smallest sigma            : {sig.min():+.2f}  at lag {ks[int(np.argmin(sig))]}")
        print(f"  LARGEST |sigma|           : {np.abs(sig).max():.2f}  "
              f"at lag {ks[int(np.argmax(np.abs(sig)))]}")
        print(f"  FUTURE (lag>0) max sigma  : {fut.max():+.2f}  at lag {ks[ks>0][int(np.argmax(fut))]}")
        print(f"  PAST   (lag<0) max sigma  : {pas.max():+.2f}  at lag {ks[ks<0][int(np.argmax(pas))]}")
        print(f"  lags above the null max   : {n_above} / {len(rows)}")
        print(f"  lags above Bonferroni     : {n_bonf} / {len(rows)}")
        if n_above:
            hits = [(r["lag"], round(r["sigma"], 1)) for r in rows if r["above_max"]]
            print(f"     -> {hits}")
        print()

        out[label] = {"null": {"mean": float(mu), "sd": float(sd), "max": float(mx),
                               "bonferroni": bonf, "n_shuffle": a.shuffles},
                      "rows": rows,
                      "max_sigma": float(sig.max()), "max_sigma_lag": int(ks[i_max]),
                      "max_abs_sigma": float(np.abs(sig).max()),
                      "future_max_sigma": float(fut.max()),
                      "past_max_sigma": float(pas.max()),
                      "n_above_max": int(n_above), "n_above_bonf": int(n_bonf)}

    with open("lag_dense_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: lag_dense_results.json")


if __name__ == "__main__":
    main()
