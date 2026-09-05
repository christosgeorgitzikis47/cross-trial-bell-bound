"""
PART 14 - THE OUTCOME-OUTCOME SCAN ON ALL TEN PULSES

Section 6.5 (I(O_A(i);O_B(i+k)), the check for a temporal re-pairing of the
records) had been run on round 28297 only. Here it is run identically on all
ten.

METHOD IDENTICAL to section 6.5: a family of 20,001 hypotheses (one pair),
g_thr = chi^2(1).ppf(1 - 0.05/20001) = 22.167, MI threshold = 1.066e-6 at
n = 15e6. k = 0 is the positive control (the Bell correlation) and is
excluded from the count.

Anything crossing the threshold at k != 0 is written FIRST and we STOP.
"""
import json, math, os
import numpy as np
from scipy.stats import chi2

from load_curby import read_file
from meros2_injection import scan, mi_and_delta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dedomena_curby")
LN2 = math.log(2.0)
K = 10_000
ROUNDS = [1000, 15000, 22000, 23000, 26000, 28293, 28294, 28295, 28296, 28297]


def main():
    m_family = 2 * K + 1                       # 20,001, as in section 6.5
    g_thr = float(chi2.ppf(1 - 0.05 / m_family, 1))

    print("=" * 78)
    print("PART 14 - I(O_A(i) ; O_B(i+k)) ON THE TEN PULSES")
    print("=" * 78)
    print(f"  |k| <= {K:,}   family {m_family:,}   G threshold = {g_thr:.3f} "
          f"(sqrt(G) = {math.sqrt(g_thr):.3f})\n")
    print(f"  {'round':>7} {'k=0 sqG':>9} {'P11/indep':>9} "
          f"{'max MI (k!=0)':>14} {'at lag':>8} {'sqG':>6} {'>thresh':>9}")

    rows = []
    n_bad = 0
    for rnd in ROUNDS:
        data, _ = read_file(os.path.join(DATA, f"curby_round_{rnd}.bin"))
        OA = (data['OA'] > 0).astype(np.int8)
        OB = (data['OB'] > 0).astype(np.int8)
        del data
        n = len(OA)
        mi_thr = g_thr / (2 * n * LN2)

        ks, n11, A1, B1, nk, ferr = scan(OA, OB, K)
        if ferr > 0.1:
            raise SystemExit(f"FFT rounding error {ferr}")
        MI, _ = mi_and_delta(n11, A1, B1, nk)
        G = 2 * nk * LN2 * MI

        i0 = K
        nz = np.ones(len(ks), bool); nz[i0] = False
        above = (G > g_thr) & nz
        n_above = int(above.sum())
        n_bad += n_above
        jm = int(np.argmax(np.where(nz, G, -np.inf)))
        # coincidence boost at k=0 over independence (positive control)
        boost = (n11[i0] / nk[i0]) / (OA.mean() * OB.mean())
        print(f"  {rnd:>7} {math.sqrt(G[i0]):>9.1f} {boost:>9.1f} "
              f"{MI[jm]:>14.4e} {int(ks[jm]):>+8d} "
              f"{math.sqrt(G[jm]):>6.2f} {n_above:>9d}", flush=True)
        if n_above:
            hits = [(int(ks[i]), float(MI[i]), float(math.sqrt(G[i])))
                    for i in np.where(above)[0]]
            print(f"    *** ABOVE THE THRESHOLD: {hits} ***")
        rows.append(dict(round=rnd, n=n, mi_threshold=float(mi_thr),
                         k0_mi=float(MI[i0]), k0_G=float(G[i0]),
                         k0_sqrtG=float(math.sqrt(G[i0])),
                         k0_coincidence_boost=float(boost),
                         max_mi_nonzero=float(MI[jm]), at_lag=int(ks[jm]),
                         max_G_nonzero=float(G[jm]),
                         max_sqrtG_nonzero=float(math.sqrt(G[jm])),
                         pct_of_threshold=float(MI[jm] / mi_thr * 100),
                         n_above_threshold=n_above,
                         deadtime_max_G=float(G[[i0-2, i0-1, i0+1, i0+2]].max()),
                         fft_err=float(ferr)))
        del OA, OB

    n_tests = len(rows) * (2 * K)
    mx = max(r["max_sqrtG_nonzero"] for r in rows)
    ratio_min = min(r["k0_mi"] / r["max_mi_nonzero"] for r in rows)
    print(f"\n  TOTAL above the threshold: {n_bad} / {n_tests:,} "
          f"(10 pulses x {2*K:,} nonzero lags)")
    print(f"  largest sqrt(G) at k != 0: {mx:.2f} sigma")
    print(f"  Bell/threshold per pulse: "
          f"{min(r['k0_mi']/r['mi_threshold'] for r in rows):,.0f} - "
          f"{max(r['k0_mi']/r['mi_threshold'] for r in rows):,.0f}")
    if n_bad:
        print("\n  *** STOP: outcome-outcome structure found ***")

    out = {"g_thr": g_thr, "K": K, "m_family": m_family,
           "n_tests_nonzero": n_tests, "n_above_total": n_bad,
           "max_sqrtG_nonzero_overall": mx,
           "clean": bool(n_bad == 0), "rows": rows}
    json.dump(out, open(os.path.join(HERE, "meros14_oo10.json"), "w"),
              indent=2)
    print("\nSaved: meros14_oo10.json")


if __name__ == "__main__":
    main()
