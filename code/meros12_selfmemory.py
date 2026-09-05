"""
PART 12 - SINGLE-PARTY MEMORY: I(O_A(i) ; S_A(i+k)) AND I(O_B(i) ; S_B(i+k))

So far I(O ; S) at lag != 0 was measured only ACROSS the wings (O_A vs S_B).
The memory of the SAME device -- Alice's outcome against Alice's setting at
another trial -- had been checked only as deadtime (|k| = 1) and only on
round 28297. Here: a full scan of 20,001 lags, in both same-party channels,
on ALL ten pulses.

METHOD: as in section 6.1 -- G = 2 n ln2 MI ~ chi^2(1), with the same
threshold z_thr = 4.848 (Bonferroni m = 40,002) taken from meros5_asym.json.

k = 0 IS THE POSITIVE CONTROL: ordinary quantum mechanics lives there
(92.2 sigma on round 28297) and it is excluded from the count. Anything that
crosses the threshold at k != 0 is written FIRST and we STOP.
"""
import json, math, os
import numpy as np

from load_curby import read_file
from meros2_injection import scan, mi_and_delta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dedomena_curby")
LN2 = math.log(2.0)
K = 10_000
ROUNDS = [1000, 15000, 22000, 23000, 26000, 28293, 28294, 28295, 28296, 28297]


def main():
    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    z_thr = m5["z_thr"]
    g_thr = z_thr ** 2

    print("=" * 78)
    print("PART 12 - SINGLE-PARTY MEMORY ON ALL TEN PULSES")
    print("=" * 78)
    print(f"  |k| <= {K:,}   threshold z = {z_thr:.4f}  (G = {g_thr:.3f})\n")
    print(f"  {'round':>7} {'channel':<11} {'k=0 sqG':>8} "
          f"{'max MI (k!=0)':>14} {'at lag':>8} {'sqG':>6} {'>thresh':>9}")

    rows = []
    n_bad_total = 0
    for rnd in ROUNDS:
        data, _ = read_file(os.path.join(DATA, f"curby_round_{rnd}.bin"))
        SA = data['SA'].astype(np.int8)
        SB = data['SB'].astype(np.int8)
        OA = (data['OA'] > 0).astype(np.int8)
        OB = (data['OB'] > 0).astype(np.int8)
        del data
        n = len(SA)
        mi_thr = g_thr / (2 * n * LN2)

        for label, o, s in [("OA vs SA", OA, SA), ("OB vs SB", OB, SB)]:
            s1 = (s == 1).astype(np.int8)
            ks, n11, A1, B1, nk, ferr = scan(o, s1, K)
            if ferr > 0.1:
                raise SystemExit(f"FFT rounding error {ferr}")
            MI, _ = mi_and_delta(n11, A1, B1, nk)
            G = 2 * nk * LN2 * MI

            i0 = K                       # k = 0: the positive control
            nz = np.ones(len(ks), bool); nz[i0] = False
            above = (G > g_thr) & nz
            n_above = int(above.sum())
            n_bad_total += n_above
            jm = int(np.argmax(np.where(nz, G, -np.inf)))
            print(f"  {rnd:>7} {label:<11} {math.sqrt(G[i0]):>8.1f} "
                  f"{MI[jm]:>14.4e} {int(ks[jm]):>+8d} "
                  f"{math.sqrt(G[jm]):>6.2f} {n_above:>9d}", flush=True)
            if n_above:
                hits = [(int(ks[i]), float(MI[i]), float(math.sqrt(G[i])))
                        for i in np.where(above)[0]]
                print(f"    *** ABOVE THE THRESHOLD: {hits} ***")
            rows.append(dict(
                round=rnd, pair=label, n=n, mi_threshold=float(mi_thr),
                k0_mi=float(MI[i0]), k0_G=float(G[i0]),
                k0_sqrtG=float(math.sqrt(G[i0])),
                max_mi_nonzero=float(MI[jm]), at_lag=int(ks[jm]),
                max_G_nonzero=float(G[jm]),
                max_sqrtG_nonzero=float(math.sqrt(G[jm])),
                pct_of_threshold=float(MI[jm] / mi_thr * 100),
                n_above_threshold=n_above,
                fft_err=float(ferr)))
        del SA, SB, OA, OB

    n_tests = len(rows) * (2 * K)        # k != 0 per channel
    print(f"\n  TOTAL above the threshold: {n_bad_total} / {n_tests:,} "
          f"({len(rows)} channels x {2*K:,} nonzero lags)")
    mx = max(r["max_sqrtG_nonzero"] for r in rows)
    print(f"  largest sqrt(G) at k != 0 anywhere: {mx:.2f} sigma")
    if n_bad_total:
        print("\n  *** STOP: single-party memory found ***")

    out = {"z_thr": z_thr, "g_thr": g_thr, "K": K,
           "n_tests_nonzero": n_tests, "n_above_total": n_bad_total,
           "max_sqrtG_nonzero_overall": mx,
           "clean": bool(n_bad_total == 0), "rows": rows}
    json.dump(out, open(os.path.join(HERE, "meros12_selfmemory.json"), "w"),
              indent=2)
    print("\nSaved: meros12_selfmemory.json")


if __name__ == "__main__":
    main()
