"""
ΜΕΡΟΣ 6.4 — ΤΟ α ΣΤΟΥΣ ΔΕΚΑ ΠΑΛΜΟΥΣ (ένσταση peer review #4)

Ο περιορισμός #1 έλεγε «το α βαθμονομήθηκε στον 28297 και δεν είναι σταθερά
του dataset». Σωστό αλλά αόριστο. Εδώ γίνεται αριθμός.

    α = δ(0) = (r₂ − r₁)/2,  r_s = ρυθμός click με ρύθμιση s, ΙΔΙΟ ζεύγος.

Υπολογίζεται από τα ωμά αρχεία και για τους δέκα παλμούς, χωριστά για Alice
(OA vs SA) και Bob (OB vs SB). Δίνεται εύρος, μέσος, sd.

ΓΙΑΤΙ ΜΕΤΡΑΕΙ: ε_excl ∝ 1/α. Παλμός με μικρότερο α δίνει χαλαρότερο όριο
στο ίδιο n. Η διασπορά του α είναι άμεσα η αβεβαιότητα μεταφοράς του χάρτη
σε άλλον παλμό.
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dedomena_curby")
sys.path.insert(0, HERE)
from load_curby import read_file                                  # noqa: E402

ROUNDS = [1000, 15000, 22000, 23000, 26000,
          28293, 28294, 28295, 28296, 28297]
GROUP = {r: ("απλωμένοι" if r < 28000 else "διαδοχικοί") for r in ROUNDS}


def rates(O, S):
    """r₁, r₂, n₁, n₂ για τον 2×2 του lag 0."""
    m1 = (S == 1)
    n1 = int(m1.sum()); n2 = int(len(S) - n1)
    k1 = int(O[m1].sum()); k2 = int(O.sum() - k1)
    return k1 / n1, k2 / n2, n1, n2, k1, k2


def main():
    print("=" * 78)
    print("ΜΕΡΟΣ 6.4 — ΤΟ α ΣΕ ΚΑΘΕΝΑΝ ΑΠΟ ΤΟΥΣ 10 ΠΑΛΜΟΥΣ")
    print("=" * 78)
    print(f"  {'γύρος':>7} {'ομάδα':>11} {'p₀(A)':>9} {'r₁(A)':>10} "
          f"{'r₂(A)':>10} {'α(A)':>11} {'α(B)':>11} {'r₂/r₁(A)':>9}")

    res = []
    for r in ROUNDS:
        path = os.path.join(DATA, f"curby_round_{r}.bin")
        data, n_raw = read_file(path)
        SA = data['SA'].astype(np.int8); SB = data['SB'].astype(np.int8)
        OA = (data['OA'] > 0).astype(np.int8)
        OB = (data['OB'] > 0).astype(np.int8)
        n = len(SA)

        rA1, rA2, nA1, nA2, kA1, kA2 = rates(OA, SA)
        rB1, rB2, nB1, nB2, kB1, kB2 = rates(OB, SB)
        aA = (rA2 - rA1) / 2.0
        aB = (rB2 - rB1) / 2.0
        p0A = (kA1 + kA2) / n
        p0B = (kB1 + kB2) / n
        # τυπικό σφάλμα του α από τη διωνυμική
        seA = 0.5 * math.sqrt(rA1 * (1 - rA1) / nA1 + rA2 * (1 - rA2) / nA2)
        seB = 0.5 * math.sqrt(rB1 * (1 - rB1) / nB1 + rB2 * (1 - rB2) / nB2)

        print(f"  {r:>7} {GROUP[r]:>11} {p0A:>9.5f} {rA1:>10.6f} "
              f"{rA2:>10.6f} {aA:>11.4e} {aB:>11.4e} {rA2/rA1:>9.4f}")
        res.append(dict(round=r, group=GROUP[r], n=n, n_raw=n_raw,
                        r1_A=rA1, r2_A=rA2, alpha_A=aA, se_alpha_A=seA,
                        p0_A=p0A, r1_B=rB1, r2_B=rB2, alpha_B=aB,
                        se_alpha_B=seB, p0_B=p0B, ratio_A=rA2 / rA1))
        del data, SA, SB, OA, OB

    aA = np.array([x["alpha_A"] for x in res])
    aB = np.array([x["alpha_B"] for x in res])
    seA = np.array([x["se_alpha_A"] for x in res])
    p0A = np.array([x["p0_A"] for x in res])

    print("\n" + "-" * 78)
    for nm, v in (("α Alice", aA), ("α Bob", aB)):
        print(f"  {nm}: μέσος {v.mean():.4e}  sd {v.std(ddof=1):.4e} "
              f"({100*v.std(ddof=1)/v.mean():.1f}%)   "
              f"εύρος [{v.min():.4e}, {v.max():.4e}]  "
              f"max/min = {v.max()/v.min():.3f}")
    print(f"  τυπικό σφάλμα ΜΕΤΡΗΣΗΣ ανά παλμό: ~{seA.mean():.2e} "
          f"({100*seA.mean()/aA.mean():.2f}%) -> η διασπορά είναι "
          f"{aA.std(ddof=1)/seA.mean():.0f}× μεγαλύτερη, άρα ΠΡΑΓΜΑΤΙΚΗ")
    print(f"  p₀ Alice: εύρος [{p0A.min():.5f}, {p0A.max():.5f}]  "
          f"({100*p0A.min():.3f}% – {100*p0A.max():.3f}%)")

    a28297 = [x for x in res if x["round"] == 28297][0]["alpha_A"]
    print(f"\n  Ο χάρτης χρησιμοποιεί α(28297) = {a28297:.4e}.")
    print(f"  Επειδή ε_excl ∝ 1/α, μεταφορά του χάρτη σε άλλον παλμό αλλάζει")
    print(f"  το όριο κατά τον λόγο α(28297)/α(παλμού):")
    print(f"    {'γύρος':>7} {'α(A)':>11} {'α(28297)/α':>11}")
    for x in res:
        print(f"    {x['round']:>7} {x['alpha_A']:>11.4e} "
              f"{a28297/x['alpha_A']:>11.3f}")
    fac = a28297 / aA
    print(f"\n  -> το ε_excl σε άλλον παλμό θα ήταν {fac.min():.2f}× έως "
          f"{fac.max():.2f}× αυτού του χάρτη (μόνο λόγω του α, με ίδιο n).")

    json.dump(dict(rounds=res,
                   alpha_A_mean=float(aA.mean()), alpha_A_sd=float(aA.std(ddof=1)),
                   alpha_A_min=float(aA.min()), alpha_A_max=float(aA.max()),
                   alpha_B_mean=float(aB.mean()), alpha_B_sd=float(aB.std(ddof=1)),
                   alpha_B_min=float(aB.min()), alpha_B_max=float(aB.max()),
                   alpha_28297_A=a28297,
                   transfer_factor_min=float(fac.min()),
                   transfer_factor_max=float(fac.max())),
              open(os.path.join(HERE, "meros6_alpha10.json"), "w"), indent=2)
    print("\nΑποθηκεύτηκε: meros6_alpha10.json")


if __name__ == "__main__":
    main()
