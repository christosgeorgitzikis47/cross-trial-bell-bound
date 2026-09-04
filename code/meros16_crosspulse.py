"""
ΜΕΡΟΣ 16 — ΣΥΣΧΕΤΙΣΗ ΤΩΝ δ̂_p(k) ΑΝΑΜΕΣΑ ΣΤΟΥΣ ΠΑΛΜΟΥΣ (ένσταση κριτή: το
inverse-variance weighting του §6.4 προϋποθέτει ανεξάρτητους παλμούς· οι πέντε
28293–28297 είναι διαδοχικοί μέσα σε 65 λεπτά).

Για κάθε ζεύγος παλμών (p,q) και κάθε κανάλι: Pearson r των δ̂_p(k)/σ_p(k) με
δ̂_q(k)/σ_q(k) πάνω στα 20.000 lag k ≠ 0. Υπό ανεξαρτησία r ~ N(0, 1/√20000 =
0,0071). Αναφέρονται max|r| σε όλα τα 45 ζεύγη, και ΧΩΡΙΣΤΑ στα 10 ζεύγη των
πέντε διαδοχικών. Επίσης το ίδιο στο T(τ): Corr των ε̂_p(τ) ανά τ δεν έχει
νόημα με 10 σημεία, αλλά η συσχέτιση των δ̂ σε επίπεδο lag είναι ακριβώς ό,τι
θα έκανε τα σ_p² μη προσθετικά, γιατί T = Σ W δ̂ είναι γραμμικό.
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
        print(f"  γύρος {rnd} σαρώθηκε ({time.time()-t0:.0f}s)", flush=True)

    n_lag = 2 * K
    se = 1.0 / math.sqrt(n_lag)
    out = {"n_lag": n_lag, "se_r": se, "pairs": {}}
    print("=" * 78)
    print("ΜΕΡΟΣ 16 — ΔΙΑΠΑΛΜΙΚΗ ΣΥΣΧΕΤΙΣΗ ΤΩΝ δ̂(k)")
    print("=" * 78)
    print(f"  {n_lag:,} lag (k ≠ 0) ανά ζεύγος· τυπικό σφάλμα του r υπό "
          f"ανεξαρτησία: {se:.4f}")
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
        print(f"\n  {label}: {len(rows)} ζεύγη")
        print(f"    max |r| = {abs(worst['r']):.4f} ({worst['z']:+.2f}σ) στο "
              f"ζεύγος ({worst['p']}, {worst['q']})")
        print(f"    μέσος r = {rs.mean():+.4f}   sd r = {rs.std(ddof=1):.4f} "
              f"(αναμ. {se:.4f})   |r|>3σ: {int((np.abs(rs)>3*se).sum())}/{len(rs)}")
        print(f"    ΔΙΑΔΟΧΙΚΟΙ (28293–28297), {len(cons)} ζεύγη: "
              f"max |r| = {abs(wc['r']):.4f} ({wc['z']:+.2f}σ) στο "
              f"({wc['p']}, {wc['q']})   μέσος r = "
              f"{np.mean([x['r'] for x in cons]):+.4f}")
        for x in cons:
            print(f"       ({x['p']}, {x['q']})  r = {x['r']:+.4f}  ({x['z']:+.2f}σ)")
        out["pairs"][label] = dict(rows=rows, max_abs_r=abs(worst["r"]),
                                   max_pair=[worst["p"], worst["q"]],
                                   max_abs_r_consecutive=abs(wc["r"]),
                                   max_pair_consecutive=[wc["p"], wc["q"]],
                                   mean_r=float(rs.mean()),
                                   sd_r=float(rs.std(ddof=1)))
    allmax = max(v["max_abs_r"] for v in out["pairs"].values())
    print("\n" + "=" * 78)
    print(f"  ΣΥΝΟΛΙΚΟ max |r| = {allmax:.4f}  ->  "
          f"{'ΑΝΕΞΑΡΤΗΤΟΙ ΠΑΛΜΟΙ, το inverse-variance στέκει' if allmax < 0.05 else '*** ΟΥΣΙΑΣΤΙΚΗ ΣΥΣΧΕΤΙΣΗ — ΣΤΑΜΑΤΑ ***'}")
    out["max_abs_r_all"] = allmax
    json.dump(out, open(os.path.join(HERE, "meros16_crosspulse.json"), "w"),
              indent=2)
    print("Αποθηκεύτηκε: meros16_crosspulse.json")


if __name__ == "__main__":
    main()
