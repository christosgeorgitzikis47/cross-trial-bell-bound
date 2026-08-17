"""
ΜΕΡΟΣ 13 — ΣΥΝΑΡΤΗΣΟΕΙΔΗ ΠΟΛΛΩΝ ΡΥΘΜΙΣΕΩΝ: ΤΟ ΤΕΣΤ ΙΣΟΤΙΜΙΑΣ

Κενό εύρους που εντόπισε ο έλεγχος: όλα τα τεστ του paper είναι περιθωριακά
ανά lag — εξάρτηση του O από ΣΥΝΑΡΤΗΣΟΕΙΔΗ πολλών ρυθμίσεων με μηδενικά
πρώτα περιθώρια (π.χ. ισοτιμία S(i+k)·S(i+k+1)) θα ήταν αόρατη.

Εδώ: η ακολουθία-γινόμενο γειτονικών ρυθμίσεων είναι κι αυτή δυαδική,
P(i) = [S(i) == S(i+1)], άρα η ίδια FFT μηχανή σαρώνει
    I(O_A(i) ; P_B(i+k))  και  I(O_B(i) ; P_A(i+k))
για κάθε |k| <= 10.000, ΚΑΙ στους δέκα παλμούς.

ΔΕΝ υπάρχει θετικός έλεγχος στο k=0: η ισοτιμία είναι ανεξάρτητη από το
S(i) μεμονωμένα (το S(i+1) είναι ομοιόμορφο και ανεξάρτητο), οπότε ΟΛΑ τα
lag, και το 0, αναμένονται μηδενικά. Η μηχανή έχει ήδη επαληθευτεί στο
ΜΕΡΟΣ 12· εδώ προστίθεται μικρή βαθμονόμηση χ²(1) με ανακατέματα.

Κατώφλι: το ίδιο z = 4,848 (Bonferroni m = 40.002, δανεικό όπως παντού).
Ό,τι περάσει το κατώφλι γράφεται ΠΡΩΤΟ και ΣΤΑΜΑΤΑΜΕ.
"""
import json, math, os
import numpy as np

from load_curby import read_file
from lag_test import mi
from meros2_injection import scan, mi_and_delta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dedomena_curby")
LN2 = math.log(2.0)
K = 10_000
ROUNDS = [1000, 15000, 22000, 23000, 26000, 28293, 28294, 28295, 28296, 28297]


def parity(S):
    """P(i) = 1 αν S(i) == S(i+1), αλλιώς 0. Μήκος n-1."""
    return (S[:-1] == S[1:]).astype(np.int8)


def main():
    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    z_thr = m5["z_thr"]
    g_thr = z_thr ** 2

    print("=" * 78)
    print("ΜΕΡΟΣ 13 — I(O ; ΙΣΟΤΙΜΙΑ ΡΥΘΜΙΣΕΩΝ) ΣΤΟΥΣ ΔΕΚΑ ΠΑΛΜΟΥΣ")
    print("=" * 78)
    print(f"  |k| <= {K:,}   κατώφλι z = {z_thr:.4f}  (G = {g_thr:.3f})")
    print("  κανένα lag δεν είναι θετικός έλεγχος — αναμένεται παντού μηδέν\n")

    # ---- βαθμονόμηση χ²(1) της ισοτιμίας σε έναν παλμό ----
    data, _ = read_file(os.path.join(DATA, "curby_round_28297.bin"))
    OA0 = (data['OA'] > 0).astype(np.int8)[:-1]
    PB0 = parity(data['SB'].astype(np.int8))
    del data
    rng = np.random.default_rng(1313)
    sh = PB0.copy()
    n0 = len(OA0)
    g = np.empty(200)
    for i in range(200):
        rng.shuffle(sh)
        g[i] = 2 * n0 * LN2 * mi(OA0, sh)
    print(f"  βαθμονόμηση: μέσος G σε 200 ανακατέματα = {g.mean():.3f} "
          f"(θεωρία 1.000), max = {g.max():.2f}")
    if not 0.8 < g.mean() < 1.2:
        raise SystemExit("Η βαθμονόμηση χ²(1) απέτυχε για την ισοτιμία.")
    del OA0, PB0, sh

    print(f"\n  {'γύρος':>7} {'κανάλι':<13} {'P(ισοτιμία)':>11} "
          f"{'max MI':>12} {'σε lag':>8} {'√G':>6} {'>κατώφλι':>9}")

    rows = []
    n_bad = 0
    for rnd in ROUNDS:
        data, _ = read_file(os.path.join(DATA, f"curby_round_{rnd}.bin"))
        SA = data['SA'].astype(np.int8)
        SB = data['SB'].astype(np.int8)
        OA = (data['OA'] > 0).astype(np.int8)
        OB = (data['OB'] > 0).astype(np.int8)
        del data
        n = len(SA) - 1
        mi_thr = g_thr / (2 * n * LN2)

        for label, o, s in [("OA vs par(SB)", OA[:-1], parity(SB)),
                            ("OB vs par(SA)", OB[:-1], parity(SA))]:
            ks, n11, A1, B1, nk, ferr = scan(o, s, K)
            if ferr > 0.1:
                raise SystemExit(f"FFT σφάλμα στρογγυλοποίησης {ferr}")
            MI, _ = mi_and_delta(n11, A1, B1, nk)
            G = 2 * nk * LN2 * MI
            above = G > g_thr
            n_above = int(above.sum())
            n_bad += n_above
            jm = int(np.argmax(G))
            print(f"  {rnd:>7} {label:<13} {float(s.mean()):>11.5f} "
                  f"{MI[jm]:>12.4e} {int(ks[jm]):>+8d} "
                  f"{math.sqrt(G[jm]):>6.2f} {n_above:>9d}", flush=True)
            if n_above:
                hits = [(int(ks[i]), float(MI[i]), float(math.sqrt(G[i])))
                        for i in np.where(above)[0]]
                print(f"    *** ΠΑΝΩ ΑΠΟ ΤΟ ΚΑΤΩΦΛΙ: {hits} ***")
            rows.append(dict(round=rnd, pair=label, n=n,
                             parity_rate=float(s.mean()),
                             mi_threshold=float(mi_thr),
                             max_mi=float(MI[jm]), at_lag=int(ks[jm]),
                             max_G=float(G[jm]),
                             max_sqrtG=float(math.sqrt(G[jm])),
                             pct_of_threshold=float(MI[jm] / mi_thr * 100),
                             n_above_threshold=n_above,
                             fft_err=float(ferr)))
        del SA, SB, OA, OB

    n_tests = len(rows) * (2 * K + 1)
    mx = max(r["max_sqrtG"] for r in rows)
    print(f"\n  ΣΥΝΟΛΟ πάνω από το κατώφλι: {n_bad} / {n_tests:,} "
          f"({len(rows)} κανάλια × {2*K+1:,} lag)")
    print(f"  μέγιστο √G σε όλο το σύνολο: {mx:.2f}σ")
    if n_bad:
        print("\n  *** ΣΤΑΜΑΤΑ: βρέθηκε εξάρτηση από ισοτιμία ***")

    out = {"z_thr": z_thr, "g_thr": g_thr, "K": K,
           "calibration_mean_G": float(g.mean()),
           "n_tests": n_tests, "n_above_total": n_bad,
           "max_sqrtG_overall": mx, "clean": bool(n_bad == 0), "rows": rows}
    json.dump(out, open(os.path.join(HERE, "meros13_parity.json"), "w"),
              indent=2)
    print("\nΑποθηκεύτηκε: meros13_parity.json")


if __name__ == "__main__":
    main()
