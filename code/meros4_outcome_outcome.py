"""
PART 4 - TEST B: OUTCOME-OUTCOME CORRELATION

A separate question from parts 1-3. If the SEQUENCE were shuffled in time (a
wrong Alice-Bob pairing in the record), the signal would not appear as an
outcome-setting correlation but as an outcome-OUTCOME correlation at lag
k != 0.

    I( OA(i) ; OB(i+k) )  for every |k| <= 10,000, by FFT.

POSITIVE CONTROL: at k=0 it MUST fire strongly (the Bell correlation).
If it does not -> a bug in the loading, and we stop.

The same design as full_scan.py: an analytic chi^2(1) threshold, VERIFIED
against real shuffles before it is used. Bonferroni over 20,001 hypotheses
(ONE pair here, not two).
"""
import argparse, json, math, os
import numpy as np
from scipy.stats import chi2, kstest

from lag_test import mi
from meros2_injection import scan, mi_and_delta

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2.0)


def validate_chi2(oa, ob, n_shuffle, seed=999):
    """Is G = 2n ln2 MI of the shuffled data ~ chi^2(1)? We shuffle OB."""
    rng = np.random.default_rng(seed)
    sh = ob.copy()
    n = len(oa)
    g = np.empty(n_shuffle)
    for i in range(n_shuffle):
        rng.shuffle(sh)
        # mi() expects a "setting" in {1,2} -> map OB into {1,2}
        g[i] = 2 * n * LN2 * mi(oa, sh + 1)
    ks = kstest(g, 'chi2', args=(1,))
    return dict(n_shuffle=n_shuffle, mean=float(g.mean()),
                var=float(g.var(ddof=1)), ks_stat=float(ks.statistic),
                ks_p=float(ks.pvalue),
                q50=float(np.percentile(g, 50)), q50_th=float(chi2.ppf(.50, 1)),
                q90=float(np.percentile(g, 90)), q90_th=float(chi2.ppf(.90, 1)),
                q99=float(np.percentile(g, 99)), q99_th=float(chi2.ppf(.99, 1)),
                q999=float(np.percentile(g, 99.9)), q999_th=float(chi2.ppf(.999, 1)),
                max=float(g.max()),
                null_mi_mean=float(g.mean() / (2 * n * LN2)),
                null_mi_sd=float(g.std(ddof=1) / (2 * n * LN2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10_000)
    ap.add_argument("--shuffles", type=int, default=2000)
    a = ap.parse_args()

    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    OA, OB = d['OA'], d['OB']
    n = len(OA)
    n_lags = 2 * a.K + 1
    n_tests = n_lags                      # ONE pair

    p_thr = 0.05 / n_tests
    g_thr = float(chi2.ppf(1 - p_thr, 1))
    mi_thr = g_thr / (2 * n * LN2)

    print("=" * 78)
    print("PART 4 - I( OA(i) ; OB(i+k) ),  |k| <= 10,000")
    print("=" * 78)
    print(f"  n = {n:,}   click A {OA.mean()*100:.4f}%  B {OB.mean()*100:.4f}%")
    print(f"  hypotheses m = {n_tests:,}   p = {p_thr:.4e}   "
          f"chi^2(1) threshold {g_thr:.3f}")
    print(f"  -> MI threshold = {mi_thr:.4e} bits/trial\n")

    print(f"  validating chi^2(1) with {a.shuffles} shuffles of OB...",
          flush=True)
    v = validate_chi2(OA, OB, a.shuffles)
    print(f"    mean G {v['mean']:.4f} (theory 1)   "
          f"variance {v['var']:.4f} (theory 2)")
    print(f"    q50 {v['q50']:.3f}/{v['q50_th']:.3f}  "
          f"q90 {v['q90']:.3f}/{v['q90_th']:.3f}  "
          f"q99 {v['q99']:.3f}/{v['q99_th']:.3f}  "
          f"q99.9 {v['q999']:.3f}/{v['q999_th']:.3f}")
    print(f"    KS: D = {v['ks_stat']:.4f}, p = {v['ks_p']:.3f}  -> "
          + ("FITS" if v['ks_p'] > 0.01 else "DOES NOT FIT"))
    if v['ks_p'] <= 0.01:
        raise SystemExit("The chi^2(1) calibration was REJECTED.")

    print(f"\n  scanning {n_lags:,} lags by FFT...", flush=True)
    ks, n11, A1, B1, nk, ferr = scan(OA, OB.astype(np.int8), a.K)
    MI, _ = mi_and_delta(n11, A1, B1, nk)
    G = 2 * nk * LN2 * MI
    print(f"    FFT rounding error: {ferr:.2e}")

    # --- verify the FFT against a direct computation ---
    rng = np.random.default_rng(31337)
    picks = np.unique(np.concatenate([[-a.K, -1, 0, 1, a.K],
                                      rng.integers(-a.K, a.K + 1, 20)]))
    bad = 0
    for k in picks:
        kk = int(k)
        if kk > 0:
            oa, ob = OA[:-kk], OB[kk:]
        elif kk < 0:
            oa, ob = OA[-kk:], OB[:kk]
        else:
            oa, ob = OA, OB
        direct = int(((oa == 1) & (ob == 1)).sum())
        j = kk + a.K
        if direct != n11[j] or len(oa) != nk[j]:
            bad += 1
            print(f"    DISAGREEMENT at lag {kk}: FFT {n11[j]} vs {direct}")
    print(f"    verified at {len(picks)} lags: "
          + ("ALL MATCH" if bad == 0 else f"{bad} DISAGREEMENTS"))
    if bad:
        raise SystemExit("The FFT disagreed.")

    # ---------------- POSITIVE CONTROL ----------------
    j0 = a.K
    mi0, G0 = float(MI[j0]), float(G[j0])
    p0 = float(chi2.sf(G0, 1))
    print("\n" + "=" * 78)
    print("POSITIVE CONTROL - k = 0")
    print("=" * 78)
    print(f"  MI(OA;OB) at k=0 = {mi0:.6e} bits/trial")
    print(f"  G = {G0:,.1f}   ({G0/g_thr:,.0f}x the threshold)")
    print(f"  p = {p0:.3e}   equivalent sigma ~ {math.sqrt(G0):.1f}")
    print(f"  2x2 table at k=0: N11 = {n11[j0]:,}  "
          f"N(OA=1) = {A1[j0]:,}  N(OB=1) = {B1[j0]:,}")
    exp11 = A1[j0] * B1[j0] / nk[j0]
    print(f"  expected N11 under independence = {exp11:,.0f}  "
          f"-> ratio {n11[j0]/exp11:.3f}")
    lights = G0 > g_thr
    print(f"  DOES IT FIRE? {'YES' if lights else 'NO - BUG, WE STOP'}")
    if not lights:
        raise SystemExit("The k=0 positive control failed: bug in the loading.")

    # ---------------- the question: k != 0 ----------------
    mask = ks != 0
    MInz, Gnz, ksnz = MI[mask], G[mask], ks[mask]
    i = int(np.argmax(Gnz))
    above = Gnz > g_thr
    sig = (MInz - v["null_mi_mean"]) / v["null_mi_sd"]

    print("\n" + "=" * 78)
    print("THE QUESTION - k != 0")
    print("=" * 78)
    print(f"  largest MI = {MInz[i]:.4e} bits/trial  at lag "
          f"{int(ksnz[i]):+,}")
    print(f"  G = {Gnz[i]:.3f}   (threshold {g_thr:.3f})  "
          f"-> {Gnz[i]/g_thr*100:.0f}% of the threshold")
    print(f"  p = {chi2.sf(Gnz[i],1):.3e}   (threshold {p_thr:.3e})")
    print(f"  sigma above the empirical null = {sig[i]:+.2f}")
    print(f"  lags above the threshold: {int(above.sum())} / {len(ksnz):,}")
    if above.any():
        print(f"    -> {[(int(ksnz[j]), float(MInz[j])) for j in np.where(above)[0][:20]]}")
    print(f"\n  +/-1 (deadtime, expected):")
    for k in (-2, -1, 1, 2):
        j = k + a.K
        print(f"    lag {k:+d}: MI = {MI[j]:.4e}  G = {G[j]:.2f}  "
              f"({'above' if G[j] > g_thr else 'below'} the threshold)")

    top = np.argsort(Gnz)[::-1][:20]
    print(f"\n  top 10 at k != 0:")
    for j in top[:10]:
        print(f"    lag {int(ksnz[j]):+7,}  MI = {MInz[j]:.4e}  "
              f"G = {Gnz[j]:6.2f}  sigma = {sig[j]:+.2f}")

    print("\n" + "=" * 78)
    print(f"  BOUND: I(OA(i);OB(i+k)) < {mi_thr:.3e} bits/trial for every "
          f"0 < |k| <= {a.K:,}")
    print(f"  Scale: k=0 is {mi0:.3e} -> the ratio is "
          f"1 / {mi0/mi_thr:,.0f}")
    print("=" * 78)

    out = dict(n=n, K=a.K, n_tests=n_tests, p_thr=p_thr, g_thr=g_thr,
               mi_threshold=mi_thr, chi2_validation=v, fft_round_error=ferr,
               k0=dict(mi=mi0, G=G0, p=p0, n11=int(n11[j0]),
                       expected_n11=float(exp11),
                       ratio=float(n11[j0] / exp11), lights=bool(lights)),
               max_nonzero=dict(lag=int(ksnz[i]), mi=float(MInz[i]),
                                G=float(Gnz[i]),
                                p=float(chi2.sf(Gnz[i], 1)),
                                sigma=float(sig[i]),
                                pct_of_threshold=float(Gnz[i] / g_thr * 100)),
               n_above=int(above.sum()),
               top20=[dict(lag=int(ksnz[j]), mi=float(MInz[j]),
                           G=float(Gnz[j])) for j in top],
               deadtime={str(k): dict(mi=float(MI[k + a.K]), G=float(G[k + a.K]))
                         for k in (-2, -1, 1, 2)})
    json.dump(out, open(os.path.join(HERE, "meros4_results.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(HERE, "meros4_OA_OB.npz"),
                        lags=ks, mi=MI, G=G)
    print("\nSaved: meros4_results.json + meros4_OA_OB.npz")


if __name__ == "__main__":
    main()
