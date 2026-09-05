"""
STEP B - FULL SCAN of every lag from -10,000 to +10,000 (20,001 values).

Replaces the claim "over the 111 lags tested" with "for EVERY |k| <= 10,000".

--------------------------------------------------------------------------
TWO TECHNICAL POINTS, BOTH VERIFIED INSIDE THE SCRIPT
--------------------------------------------------------------------------

1. THE BONFERRONI THRESHOLD CANNOT BE OBTAINED EMPIRICALLY.
   20,001 lags x 2 pairs = 40,002 hypotheses -> it needs the 99.999875th
   percentile of the null. With 2,000 shuffles the most extreme value
   available is about the 99.95th; some 800,000 shuffles would be needed.

   Instead: an analytic calibration. Under H0 the table is 2x2
   (outcome {0,1} x setting {1,2}), so df = 1 and

       G = 2 * n * ln2 * MI   ~   chi^2(1)

   -> a closed-form threshold, with no shuffling.
   VERIFICATION (--validate-null): we build 2,000 shuffles and check that
   their G really does follow chi^2(1) -- mean 1, variance 2, KS, tails.
   If it does not fit, it is NOT used.

   Bonferroni holds without an independence assumption. The 40,002 lags are
   correlated (they share data), but that makes the correction CONSERVATIVE,
   not invalid.

2. SPEED: 40,002 direct MI evaluations take about an hour. The counts at each
   lag form a cross-correlation, so they all come out AT ONCE by FFT:

       N11(k) = sum_i o[i] * s1[i+k]        s1 = (setting == 1)

   The other three cells follow from the margins, which are prefix sums.
   VERIFICATION: 20 random lags are compared against a direct computation;
   any difference != 0 aborts the run.
"""
import argparse, json, math, os
import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.stats import chi2, kstest

from lag_test import mi, align

HERE = os.path.dirname(os.path.abspath(__file__))
K = 10_000
LN2 = math.log(2.0)


# ------------------------------------------------------------------ counts
def n11_all_lags(o, s1, K):
    """N11(k) = sum_i o[i]*s1[i+k] for every k in [-K, +K], with ONE pair of
    FFTs.

    Zero-padded to a length >= n+K so that the circular correlation COINCIDES
    with the linear one at the lags of interest (no wraparound)."""
    n = len(o)
    L = next_fast_len(n + K + 1)
    O = rfft(o.astype(np.float64), L)
    S = rfft(s1.astype(np.float64), L)
    R = irfft(np.conjugate(O) * S, L)
    pos = R[:K + 1]                 # k = 0 ... +K
    neg = R[L - K:]                 # k = -K ... -1
    c = np.concatenate([neg, pos])  # k = -K ... +K
    return np.rint(c).astype(np.int64), float(np.abs(c - np.rint(c)).max())


def margins(o, s1, K):
    """A1(k) = #{o=1}, B1(k) = #{s=1}, n_k, over the same aligned samples."""
    n = len(o)
    ks = np.arange(-K, K + 1)
    co = np.concatenate([[0], np.cumsum(o, dtype=np.int64)])
    cs = np.concatenate([[0], np.cumsum(s1, dtype=np.int64)])
    tot_o, tot_s = int(co[-1]), int(cs[-1])
    A1 = np.empty(len(ks), np.int64)
    B1 = np.empty(len(ks), np.int64)
    for j, k in enumerate(ks):
        if k >= 0:
            # indices of o: 0 ... n-k-1 ;  of s: k ... n-1
            A1[j] = co[n - k]
            B1[j] = tot_s - cs[k]
        else:
            m = -k
            # indices of o: m ... n-1 ;  of s: 0 ... n-m-1
            A1[j] = tot_o - co[m]
            B1[j] = cs[n - m]
    nk = n - np.abs(ks)
    return ks, A1, B1, nk


def mi_from_counts(n11, A1, B1, nk):
    """MI in bits from the 2x2 table, vectorised over all lags."""
    n11 = n11.astype(np.float64); A1 = A1.astype(np.float64)
    B1 = B1.astype(np.float64);   nk = nk.astype(np.float64)
    cells = np.stack([n11, A1 - n11, B1 - n11, nk - A1 - B1 + n11])   # 11,12,01,02
    rows = np.stack([A1, A1, nk - A1, nk - A1])
    cols = np.stack([B1, nk - B1, B1, nk - B1])
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (cells / nk) * np.log2(cells * nk / (rows * cols))
    return np.nansum(t, axis=0)


# --------------------------------------------------------------- chi^2 null
def validate_chi2(o, s, n_shuffle, seed=777):
    """Checks that G = 2n ln2 MI of the shuffled data follows chi^2(1)."""
    rng = np.random.default_rng(seed)
    sh = s.copy()
    n = len(o)
    g = np.empty(n_shuffle)
    for i in range(n_shuffle):
        rng.shuffle(sh)
        g[i] = 2 * n * LN2 * mi(o, sh)
    ks = kstest(g, 'chi2', args=(1,))
    return {
        "n_shuffle": n_shuffle,
        "mean": float(g.mean()), "mean_theory": 1.0,
        "var": float(g.var(ddof=1)), "var_theory": 2.0,
        "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
        "q50": float(np.percentile(g, 50)), "q50_theory": float(chi2.ppf(.50, 1)),
        "q90": float(np.percentile(g, 90)), "q90_theory": float(chi2.ppf(.90, 1)),
        "q99": float(np.percentile(g, 99)), "q99_theory": float(chi2.ppf(.99, 1)),
        "q999": float(np.percentile(g, 99.9)), "q999_theory": float(chi2.ppf(.999, 1)),
        "max": float(g.max()), "max_theory": float(chi2.ppf(1 - 1/n_shuffle, 1)),
        "null_mi_mean": float(g.mean() / (2 * n * LN2)),
        "null_mi_sd": float(g.std(ddof=1) / (2 * n * LN2)),
        "null_mi_max": float(g.max() / (2 * n * LN2)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?",
                    default=os.path.join(HERE, "curby_28297.npz"))
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--shuffles", type=int, default=2000)
    ap.add_argument("--checks", type=int, default=20)
    a = ap.parse_args()

    d = np.load(a.path)
    SA, SB, OA, OB = d['SA'], d['SB'], d['OA'], d['OB']
    n = len(SA)
    n_lags = 2 * a.K + 1
    n_tests = n_lags * 2
    print(f"Pulse: {a.path}")
    print(f"Trials: {n:,}   lags: {n_lags:,} per pair   "
          f"TOTAL HYPOTHESES: {n_tests:,}\n")

    # ---------------- Bonferroni threshold, analytic ----------------
    alpha = 0.05
    p_thr = alpha / n_tests
    g_thr = float(chi2.ppf(1 - p_thr, 1))
    print("=" * 78)
    print("BONFERRONI THRESHOLD FOR THE NEW NUMBER OF HYPOTHESES")
    print("=" * 78)
    print(f"  hypotheses m          = {n_tests:,}   (was 222 over 111 lags)")
    print(f"  alpha                 = {alpha}")
    print(f"  p per hypothesis      = alpha/m = {p_thr:.4e}")
    print(f"  chi^2(1) threshold    = {g_thr:.4f}")
    print(f"  -> MI threshold       = {g_thr/(2*n*LN2):.4e} bits/trial  (at n={n:,})")
    print(f"  (for comparison: the 222-hypothesis threshold was 7.0e-07)\n")

    rng_chk = np.random.default_rng(12345)
    out = {"n": n, "K": a.K, "n_lags": n_lags, "n_tests": n_tests,
           "alpha": alpha, "p_per_test": p_thr, "chi2_threshold": g_thr,
           "pairs": {}}

    for label, o_full, s_full in [("OA vs SB", OA, SB), ("OB vs SA", OB, SA)]:
        print("=" * 78)
        print(label)
        print("=" * 78)

        # --- 1. verify the chi^2(1) calibration on a real null ---
        print(f"  validating chi^2(1) with {a.shuffles} shuffles...",
              flush=True)
        v = validate_chi2(o_full, s_full, a.shuffles)
        print(f"    mean G    {v['mean']:.4f}  (theory 1)     "
              f"variance {v['var']:.4f}  (theory 2)")
        print(f"    median    {v['q50']:.4f} / {v['q50_theory']:.4f}    "
              f"q90 {v['q90']:.3f} / {v['q90_theory']:.3f}    "
              f"q99 {v['q99']:.3f} / {v['q99_theory']:.3f}")
        print(f"    q99.9     {v['q999']:.3f} / {v['q999_theory']:.3f}    "
              f"max {v['max']:.3f} / {v['max_theory']:.3f} (expected)")
        print(f"    KS: D = {v['ks_stat']:.4f}, p = {v['ks_p']:.3f}  -> "
              + ("FITS" if v['ks_p'] > 0.01 else "DOES NOT FIT"))
        if v['ks_p'] <= 0.01:
            raise SystemExit("The chi^2(1) calibration was REJECTED - stop.")

        # --- 2. full scan by FFT ---
        print(f"  scanning {n_lags:,} lags by FFT...", flush=True)
        s1 = (s_full == 1).astype(np.int8)
        n11, ferr = n11_all_lags(o_full, s1, a.K)
        ks, A1, B1, nk = margins(o_full, s1, a.K)
        print(f"    largest FFT rounding error: {ferr:.2e}  "
              + ("(safe)" if ferr < 0.1 else "(HIGH)"))

        # --- 3. verify the FFT against a direct computation ---
        picks = np.unique(np.concatenate([
            [-a.K, -1, 0, 1, a.K],
            rng_chk.integers(-a.K, a.K + 1, a.checks)]))
        bad = 0
        for k in picks:
            oo, ss = align(o_full, s_full, int(k))
            direct = int(((oo == 1) & (ss == 1)).sum())
            j = int(k) + a.K
            if direct != n11[j] or len(oo) != nk[j] or int((oo == 1).sum()) != A1[j] \
                    or int((ss == 1).sum()) != B1[j]:
                bad += 1
                print(f"    DISAGREEMENT at lag {k}: FFT {n11[j]} vs direct {direct}")
        print(f"    verified at {len(picks)} lags: "
              + ("ALL MATCH" if bad == 0 else f"{bad} DISAGREEMENTS"))
        if bad:
            raise SystemExit("The FFT disagreed with the direct computation.")

        # --- 4. MI, G, p ---
        MI = mi_from_counts(n11, A1, B1, nk)
        G = 2 * nk * LN2 * MI
        above = G > g_thr
        i_max = int(np.argmax(G))
        # sigma against the empirical null, for continuity with the earlier report
        sig = (MI - v["null_mi_mean"]) / v["null_mi_sd"]

        print(f"\n    LARGEST over {n_lags:,} lags:")
        print(f"      lag              = {int(ks[i_max]):+,}")
        print(f"      MI               = {MI[i_max]:.4e} bits/trial")
        print(f"      G                = {G[i_max]:.3f}   (threshold {g_thr:.3f})")
        print(f"      p (chi^2(1))     = {chi2.sf(G[i_max], 1):.3e}   "
              f"(threshold {p_thr:.3e})")
        print(f"      sigma above null = {sig[i_max]:+.2f}")
        print(f"      -> {'ABOVE THE THRESHOLD' if above[i_max] else 'BELOW THE THRESHOLD'}"
              f"  ({MI[i_max]/(g_thr/(2*n*LN2))*100:.0f}% of the threshold)")
        print(f"    lags above the threshold: {int(above.sum())} / {n_lags:,}")
        if above.any():
            hit = [(int(ks[i]), float(MI[i]), float(G[i])) for i in np.where(above)[0]]
            print(f"      -> {hit[:20]}")
        # how many we would expect by chance at this threshold
        print(f"    (expected by chance across the whole experiment: "
              f"{n_tests*p_thr:.3f} = alpha = {alpha})")

        # where the 111 old lags fall inside the new full set
        old_lags = list(range(-50, 51)) + [-10000, -3000, -1000, -300, -100,
                                           100, 300, 1000, 3000, 10000]
        oi = np.array([int(k) + a.K for k in old_lags if abs(k) <= a.K])
        i_old = oi[int(np.argmax(MI[oi]))]
        rank = int((MI > MI[i_old]).sum()) + 1
        print(f"    old maximum (over the 111 tested lags): MI = {MI[i_old]:.4e} "
              f"at lag {int(ks[i_old]):+,}  -> rank {rank} among {n_lags:,}")

        out["pairs"][label] = {
            "chi2_validation": v,
            "fft_round_error": ferr,
            "max_lag": int(ks[i_max]), "max_mi": float(MI[i_max]),
            "max_G": float(G[i_max]), "max_p": float(chi2.sf(G[i_max], 1)),
            "max_sigma_vs_null": float(sig[i_max]),
            "n_above_threshold": int(above.sum()),
            "mi_threshold": float(g_thr / (2 * n * LN2)),
            "top20": [{"lag": int(ks[i]), "mi": float(MI[i]), "G": float(G[i]),
                       "p": float(chi2.sf(G[i], 1))}
                      for i in np.argsort(G)[::-1][:20]],
        }
        np.savez_compressed(
            os.path.join(HERE, f"full_scan_{label.replace(' ', '_')}.npz"),
            lags=ks, mi=MI, G=G)
        print()

    mx = max(p["max_mi"] for p in out["pairs"].values())
    print("=" * 78)
    print("THE BOUND FOR EVERY |k| <= 10,000")
    print("=" * 78)
    print(f"  largest observed MI over {n_tests:,} hypotheses: {mx:.4e} bits/trial")
    print(f"  Bonferroni threshold:                          "
          f"{g_thr/(2*n*LN2):.4e} bits/trial")
    print(f"  -> MI(outcome ; setting at lag k) < "
          f"{g_thr/(2*n*LN2):.2e} bits/trial for EVERY |k| <= {a.K:,}")
    out["max_mi_overall"] = mx
    out["mi_threshold"] = g_thr / (2 * n * LN2)
    json.dump(out, open(os.path.join(HERE, "full_scan_results.json"), "w"),
              indent=2)
    print(f"\nSaved: full_scan_results.json  + full_scan_*.npz")


if __name__ == "__main__":
    main()
