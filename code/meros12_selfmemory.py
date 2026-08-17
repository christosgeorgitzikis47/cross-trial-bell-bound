"""
ΜΕΡΟΣ 12 — SINGLE-PARTY MEMORY: I(O_A(i) ; S_A(i+k)) ΚΑΙ I(O_B(i) ; S_B(i+k))

Μέχρι τώρα το I(O ; S) σε lag != 0 μετριόταν μόνο ΔΙΑΣΤΑΥΡΟΥΜΕΝΑ
(O_A vs S_B). Η μνήμη της ΙΔΙΑΣ συσκευής — το αποτέλεσμα της Alice έναντι
της ρύθμισης της Alice σε άλλο trial — είχε ελεγχθεί μόνο ως deadtime
(|k| = 1) και μόνο στον 28297. Εδώ: πλήρης σάρωση 20.001 lag, και στα δύο
same-party κανάλια, ΣΕ ΟΛΟΥΣ τους δέκα παλμούς.

ΜΕΘΟΔΟΛΟΓΙΑ: ίδια με το §6.1 — G = 2 n ln2 MI ~ χ²(1), κατώφλι το ίδιο
z_thr = 4,848 (Bonferroni m = 40.002) από το meros5_asym.json.

k = 0 ΕΙΝΑΙ ΘΕΤΙΚΟΣ ΕΛΕΓΧΟΣ: εκεί ζει η συνηθισμένη κβαντομηχανική
(92,2σ στον 28297) και εξαιρείται από την καταμέτρηση. Ό,τι περάσει το
κατώφλι σε k != 0 το γράφουμε ΠΡΩΤΟ και ΣΤΑΜΑΤΑΜΕ.
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
    print("ΜΕΡΟΣ 12 — SINGLE-PARTY MEMORY ΣΕ ΟΛΟΥΣ ΤΟΥΣ ΔΕΚΑ ΠΑΛΜΟΥΣ")
    print("=" * 78)
    print(f"  |k| <= {K:,}   κατώφλι z = {z_thr:.4f}  (G = {g_thr:.3f})\n")
    print(f"  {'γύρος':>7} {'κανάλι':<11} {'k=0 √G':>8} "
          f"{'max MI (k!=0)':>14} {'σε lag':>8} {'√G':>6} {'>κατώφλι':>9}")

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
                raise SystemExit(f"FFT σφάλμα στρογγυλοποίησης {ferr}")
            MI, _ = mi_and_delta(n11, A1, B1, nk)
            G = 2 * nk * LN2 * MI

            i0 = K                       # k = 0: θετικός έλεγχος
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
                print(f"    *** ΠΑΝΩ ΑΠΟ ΤΟ ΚΑΤΩΦΛΙ: {hits} ***")
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

    n_tests = len(rows) * (2 * K)        # k != 0 ανά κανάλι
    print(f"\n  ΣΥΝΟΛΟ πάνω από το κατώφλι: {n_bad_total} / {n_tests:,} "
          f"({len(rows)} κανάλια × {2*K:,} μη μηδενικά lag)")
    mx = max(r["max_sqrtG_nonzero"] for r in rows)
    print(f"  μέγιστο √G σε k != 0 σε όλο το σύνολο: {mx:.2f}σ")
    if n_bad_total:
        print("\n  *** ΣΤΑΜΑΤΑ: βρέθηκε single-party μνήμη ***")

    out = {"z_thr": z_thr, "g_thr": g_thr, "K": K,
           "n_tests_nonzero": n_tests, "n_above_total": n_bad_total,
           "max_sqrtG_nonzero_overall": mx,
           "clean": bool(n_bad_total == 0), "rows": rows}
    json.dump(out, open(os.path.join(HERE, "meros12_selfmemory.json"), "w"),
              indent=2)
    print("\nΑποθηκεύτηκε: meros12_selfmemory.json")


if __name__ == "__main__":
    main()
