"""
ΜΕΡΟΣ 6.1 — ΤΟ p₀ ΣΕ ΑΣΥΜΜΕΤΡΗ ΔΙΑΤΑΞΗ (ένσταση peer review #1)

Η ένσταση: «σε διάταξη Eberhard οι δύο ρυθμίσεις έχουν ρυθμούς click που
διαφέρουν κατά 1,79×. Ποιο p₀ μπαίνει στον τύπο; Μέσος όρος ως πρόχειρη
προσέγγιση;»

ΤΡΙΑ ΠΡΑΓΜΑΤΑ ΕΛΕΓΧΟΝΤΑΙ ΕΔΩ

(α) Το p₀ είναι η ΠΕΡΙΘΩΡΙΑ πιθανότητα click: p₀ = (k₁+k₂)/n, δηλαδή το
    περιθώριο του 2×2 πίνακα. ΔΕΝ είναι «μέσος όρος ως προσέγγιση».
    Επειδή οι ρυθμίσεις είναι 50/50, ισχύει p₀ = (r₁+r₂)/2 — ταυτοτικά αν
    n₁ = n₂ ακριβώς. Μετριέται πόσο αποκλίνει το n₁/n₂ από το 1/2 και πόσο
    κοστίζει αυτό στην ταυτότητα.

(β) Το ΑΚΡΙΒΕΣ MI του 2×2 (άθροισμα 4 όρων p log p, καμία ανάπτυξη) δίπλα
    στο προσεγγιστικό δ²/(2 ln2 p₀(1−p₀)).

(γ) ΤΟ ΚΥΡΙΟ: πού ΧΡΗΣΙΜΟΠΟΙΕΙΤΑΙ η προσέγγιση. Ο χάρτης ε_excl
    ανακατασκευάζεται από το JSON χρησιμοποιώντας ΜΟΝΟ (T, σ_T, α, Q) και
    συγκρίνεται με τον δημοσιευμένο. Αν ταυτίζεται στο τελευταίο ψηφίο, η
    προσέγγιση (και άρα το p₀) ΔΕΝ μπαίνει πουθενά στο όριο.
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2.0)


def exact_mi_2x2(k1, k2, n1, n2):
    """ΑΚΡΙΒΕΣ MI (bits) του 2×2 από τα ωμά πλήθη. Καμία ανάπτυξη."""
    n = n1 + n2
    cells = np.array([[n1 - k1, k1], [n2 - k2, k2]], dtype=np.float64)
    rows = cells.sum(1, keepdims=True)
    cols = cells.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (cells / n) * np.log2(cells * n / (rows * cols))
    return float(np.nansum(t))


def main():
    cal = json.load(open(os.path.join(HERE, "meros1_alpha.json")))
    print("=" * 78)
    print("ΜΕΡΟΣ 6.1 — p₀: ΠΕΡΙΘΩΡΙΟ, ΟΧΙ ΠΡΟΣΕΓΓΙΣΗ")
    print("=" * 78)

    for label in ("OA vs SA", "OB vs SB"):
        d = cal[label]
        n1, n2 = d["n1"], d["n2"]
        k1, k2 = int(d["counts"][1][0]), int(d["counts"][1][1])
        n = n1 + n2
        r1, r2 = k1 / n1, k2 / n2

        p0_marginal = (k1 + k2) / n              # ΟΡΙΣΜΟΣ: περιθώριο
        p0_mean = (r1 + r2) / 2                  # ταυτότητα αν n₁ = n₂
        frac1 = n1 / n

        print(f"\n--- {label} ---")
        print(f"  n₁ = {n1:,}  n₂ = {n2:,}   n₁/n = {frac1:.9f}  "
              f"(απόκλιση από 1/2: {frac1-0.5:+.2e})")
        print(f"  r₁ = {r1:.9e}   r₂ = {r2:.9e}   r₂/r₁ = {r2/r1:.4f}")
        print(f"  p₀ (περιθώριο)  = {p0_marginal:.12e}")
        print(f"  (r₁+r₂)/2       = {p0_mean:.12e}")
        print(f"  σχετική διαφορά = {abs(p0_mean-p0_marginal)/p0_marginal:.3e}"
              f"   [ταυτοτικά 0 αν n₁ = n₂]")
        print(f"  αναλυτικά: p₀ = (n₁r₁+n₂r₂)/n, άρα p₀−(r₁+r₂)/2 = "
              f"(n₁−n₂)/(2n)·(r₁−r₂) = "
              f"{(n1-n2)/(2*n)*(r1-r2):+.3e}")

        mi_exact = exact_mi_2x2(k1, k2, n1, n2)
        delta = (r2 - r1) / 2
        mi_approx = delta ** 2 / (2 * LN2 * p0_marginal * (1 - p0_marginal))
        # η ίδια προσέγγιση αν κάποιος έβαζε λάθος p₀ (γεωμετρικό/min/max)
        alts = {"√(r₁r₂)": math.sqrt(r1 * r2), "r₁": r1, "r₂": r2}

        print(f"\n  MI ΑΚΡΙΒΕΣ  (4 όροι p log p, χωρίς ανάπτυξη) = "
              f"{mi_exact:.9e} bits/trial")
        print(f"  MI ΠΡΟΣΕΓΓΙΣΤΙΚΟ  δ²/(2 ln2 p₀(1−p₀))        = "
              f"{mi_approx:.9e} bits/trial")
        print(f"  λόγος προσεγγιστικό/ακριβές = {mi_approx/mi_exact:.6f} "
              f"({100*(mi_approx/mi_exact-1):+.2f}%)")
        print(f"  (το JSON έγραφε: exact {d['mi_measured_exact']:.9e}, "
              f"predicted {d['mi_predicted']:.9e}) -> "
              f"{'ΣΥΜΦΩΝΟΥΝ' if abs(mi_exact-d['mi_measured_exact'])<1e-15 else 'ΔΙΑΦΩΝΟΥΝ'}")
        for nm, v in alts.items():
            m = delta ** 2 / (2 * LN2 * v * (1 - v))
            print(f"    αν κάποιος έβαζε p₀ = {nm:>7}: "
                  f"λόγος {m/mi_exact:.4f}")

    # ---------- (γ) ο χάρτης ΔΕΝ περνά από την προσέγγιση ----------
    print("\n" + "=" * 78)
    print("ΕΞΑΡΤΗΣΕΙΣ ΤΟΥ ΧΑΡΤΗ ε_excl — ΑΝΑΚΑΤΑΣΚΕΥΗ ΑΠΟ ΤΑ ΩΜΑ ΜΕΓΕΘΗ")
    print("=" * 78)
    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    z_thr = m5["z_thr"]
    worst = 0.0
    for pair, alpha in (("OA vs SB", m5["alpha_A"]), ("OB vs SA", m5["alpha_B"])):
        for kn in m5["kernels"]:
            P = m5["pairs"][pair][kn]
            T = np.array(P["T"]); sT = np.array(P["sigma_T"])
            Q = np.array(P["Q"])
            # ΜΟΝΟ αυτά: T, σ_T, α, Q, z_thr. Κανένα p₀, κανένα C, κανένα MI.
            rebuilt = np.abs(T / (alpha * Q)) + z_thr * sT / (alpha * Q)
            published = np.array(P["eps_excl"])
            worst = max(worst, float(np.abs(rebuilt / published - 1).max()))
    print(f"  ε_excl = |T|/(αQ) + z·σ_T/(αQ)")
    print(f"  μέγιστη σχετική διαφορά ανακατασκευής vs δημοσιευμένου σε "
          f"{2*4*len(m5['taus'])} σημεία: {worst:.2e}")
    print(f"  -> {'ΤΑΥΤΙΖΟΝΤΑΙ' if worst < 1e-12 else 'ΔΕΝ ταυτίζονται'}")
    print("\n  Τα συστατικά και η προέλευσή τους:")
    print("    T(τ)  = Σ W(k)·δ̂(k)      — μετρημένοι ρυθμοί click, 2×2 ανά lag")
    print("    σ_T   = max(εμπειρικό από ανακατέματα, διωνυμικό)")
    print("    α     = δ(0) = (r₂−r₁)/2  — μετρημένη διαφορά ρυθμών στο lag 0")
    print("    Q(τ)  = Σ W(k)²           — καθαρά γεωμετρικό")
    print("  ΚΑΝΕΝΑ από αυτά δεν περιέχει p₀ ή την ανάπτυξη 2ης τάξης.")
    print("  Η προσέγγιση χρησιμοποιείται ΜΟΝΟ ως έλεγχος συνέπειας (μέρος 1)")
    print("  και στη ΔΕΥΤΕΡΕΥΟΥΣΑ στήλη ε_excl(I) της αναφοράς #6, που δεν")
    print("  μπαίνει στον χάρτη ούτε στο PNG.")

    out = dict(z_thr=z_thr, rebuild_max_rel_diff=worst)
    for label in ("OA vs SA", "OB vs SB"):
        d = cal[label]
        n1, n2 = d["n1"], d["n2"]
        k1, k2 = int(d["counts"][1][0]), int(d["counts"][1][1])
        r1, r2 = k1 / n1, k2 / n2
        p0m = (k1 + k2) / (n1 + n2)
        mi_e = exact_mi_2x2(k1, k2, n1, n2)
        mi_a = ((r2 - r1) / 2) ** 2 / (2 * LN2 * p0m * (1 - p0m))
        out[label] = dict(n1=n1, n2=n2, r1=r1, r2=r2, p0_marginal=p0m,
                          p0_mean=(r1 + r2) / 2, mi_exact=mi_e,
                          mi_approx=mi_a, ratio=mi_a / mi_e)
    json.dump(out, open(os.path.join(HERE, "meros6_p0.json"), "w"), indent=2)
    print("\nΑποθηκεύτηκε: meros6_p0.json")


if __name__ == "__main__":
    main()
