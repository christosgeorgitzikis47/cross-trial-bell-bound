"""
PART 16 - CORRELATION OF delta-hat_p(k) BETWEEN PULSES (referee objection: the
inverse-variance weighting of section 6.4 presupposes independent pulses, and
the five rounds 28293-28297 were acquired consecutively within 65 minutes).

For every pair of pulses (p,q) and every channel: the Pearson r of
delta-hat_p(k)/sigma_p(k) with delta-hat_q(k)/sigma_q(k) over the 20,000 lags
k != 0. Under independence r ~ N(0, 1/sqrt(20000) = 0.0071). Reported: max |r|
over all 45 pairs, and SEPARATELY over the 10 pairs among the five consecutive
pulses. Correlating the eps-hat_p(tau) across tau would be meaningless with ten
points, whereas the correlation of the delta-hat at lag level is exactly what
would make the sigma_p^2 non-additive, because T = sum W delta-hat is linear.
"""
import json, os, math, itertools, time
import numpy as np
from load_curby import read_file
from meros2_injection import scan, mi_and_delta
from meros3_map import sigma_delta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dedomena_curby")
ROUNDS = [1000, 15000, 22000, 23000, 26000, 28293, 28294, 28295, 28296, 28297]
CONSEC = [28293, 28294, 28295, 28296, 28297]
K = 10_000


def main():
    zs = {"OA vs SB": {}, "OB vs SA": {}}
    t0 = time.time()
    for rnd in ROUNDS:
        data, _ = read_file(os.path.join(DATA, f"curby_round_{rnd}.bin"))
        SA = data['SA'].astype(np.int8); SB = data['SB'].astype(np.int8)
        OA = (data['OA'] > 0).astype(np.int8); OB = (data['OB'] > 0).astype(np.int8)
        del data
        for label, O, S in (("OA vs SB", OA, SB), ("OB vs SA", OB, SA)):
            s1 = (S == 1).astype(np.int8)
            ks, n11, A1, B1, nk, _ = scan(O, s1, K)
            _, delta = mi_and_delta(n11, A1, B1, nk)
            sd = sigma_delta(A1, B1, nk)
            nz = ks != 0
            zs[label][rnd] = (delta[nz] / sd[nz]).astype(np.float64)
        print(f"  round {rnd} scanned ({time.time()-t0:.0f}s)", flush=True)

    n_lag = 2 * K
    se = 1.0 / math.sqrt(n_lag)
    out = {"n_lag": n_lag, "se_r": se, "pairs": {}}
    print("=" * 78)
    print("PART 16 - CROSS-PULSE CORRELATION OF delta-hat(k)")
    print("=" * 78)
    print(f"  {n_lag:,} lags (k != 0) per pair; standard error of r under "
          f"independence: {se:.4f}")
    for label in zs:
        rows = []
        for p, q in itertools.combinations(ROUNDS, 2):
            r = float(np.corrcoef(zs[label][p], zs[label][q])[0, 1])
            rows.append(dict(p=p, q=q, r=r, z=r / se,
                             consecutive=(p in CONSEC and q in CONSEC)))
        rs = np.array([x["r"] for x in rows])
        worst = max(rows, key=lambda x: abs(x["r"]))
        cons = [x for x in rows if x["consecutive"]]
        wc = max(cons, key=lambda x: abs(x["r"]))
        print(f"\n  {label}: {len(rows)} pairs")
        print(f"    max |r| = {abs(worst['r']):.4f} ({worst['z']:+.2f} sigma) "
              f"for the pair ({worst['p']}, {worst['q']})")
        print(f"    mean r = {rs.mean():+.4f}   sd r = {rs.std(ddof=1):.4f} "
              f"(expected {se:.4f})   |r|>3 sigma: "
              f"{int((np.abs(rs)>3*se).sum())}/{len(rs)}")
        print(f"    CONSECUTIVE (28293-28297), {len(cons)} pairs: "
              f"max |r| = {abs(wc['r']):.4f} ({wc['z']:+.2f} sigma) for "
              f"({wc['p']}, {wc['q']})   mean r = "
              f"{np.mean([x['r'] for x in cons]):+.4f}")
        for x in cons:
            print(f"       ({x['p']}, {x['q']})  r = {x['r']:+.4f}  "
                  f"({x['z']:+.2f} sigma)")
        out["pairs"][label] = dict(rows=rows, max_abs_r=abs(worst["r"]),
                                   max_pair=[worst["p"], worst["q"]],
                                   max_abs_r_consecutive=abs(wc["r"]),
                                   max_pair_consecutive=[wc["p"], wc["q"]],
                                   mean_r=float(rs.mean()),
                                   sd_r=float(rs.std(ddof=1)))
    allmax = max(v["max_abs_r"] for v in out["pairs"].values())
    print("\n" + "=" * 78)
    print(f"  OVERALL max |r| = {allmax:.4f}  ->  "
          f"{'PULSES INDEPENDENT, the inverse-variance weighting holds' if allmax < 0.05 else '*** SUBSTANTIAL CORRELATION - STOP ***'}")
    out["max_abs_r_all"] = allmax
    json.dump(out, open(os.path.join(HERE, "meros16_crosspulse.json"), "w"),
              indent=2)
    print("Saved: meros16_crosspulse.json")


if __name__ == "__main__":
    main()
