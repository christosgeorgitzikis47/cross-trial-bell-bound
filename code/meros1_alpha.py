"""
ΜΕΡΟΣ 1 — ΒΑΘΜΟΝΟΜΗΣΗ ΤΟΥ α ΑΠΟ ΤΑ ΔΕΔΟΜΕΝΑ

Μοντέλο (Α+Γ):
    λ(i) = λ0(i) + ε · Σ_k W_τ(k) · S(i+k),      W_τ(k) = exp(-k²/2τ²)
με S κωδικοποιημένο ±1.

ΣΥΜΒΑΣΗ ΠΛΑΤΟΥΣ (κρίσιμη, αλλιώς βγαίνει παράγοντας 4):
    S = ±1  ->  λ = λ0 ± δ  με  δ = α·ε·W_τ(k)
    Άρα η ΜΕΤΡΗΣΙΜΗ διαφορά ρυθμού click ανάμεσα στις δύο ρυθμίσεις είναι
        Δ = rate(S=+1) - rate(S=-1) = 2δ,    δηλαδή  δ = Δ/2.

Η προσέγγιση δεύτερης τάξης που ελέγχουμε:
    I ≈ δ² / (2 ln2 · p0(1-p0))
Παράγεται από χ²(1) του 2x2 με ίσα περιθώρια ρύθμισης:
    χ² = Δ²·n / (4 p0(1-p0)),  MI = χ²/(2 n ln2) = Δ²/(8 ln2 p0(1-p0))
                                       = δ²/(2 ln2 p0(1-p0))   ✓ συνεπές.

Το α ΔΕΝ είναι ελεύθερο: ορίζουμε ε = 1 ≡ «όσο ισχυρή είναι η κανονική
κβαντομηχανική εξάρτηση αποτελέσματος-ρύθμισης του ΙΔΙΟΥ μέρους στο lag 0».
Τότε (W_τ(0)=1 για κάθε τ):
    α = δ(0)  [σε μονάδες πιθανότητας click ανά μονάδα ε]

Τρέχει και για τα δύο μέρη (OA vs SA, OB vs SB) ως έλεγχο συνέπειας.
"""
import json, math, os
import numpy as np

from lag_test import mi

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2.0)


def exact_mi_2x2(o, s):
    """Ακριβές MI σε bits από τον πίνακα 2x2 (ρύθμιση ∈ {1,2})."""
    c = np.zeros((2, 2), float)
    for oi in (0, 1):
        for si in (1, 2):
            c[oi, si - 1] = np.count_nonzero((o == oi) & (s == si))
    j = c / c.sum()
    po = j.sum(1, keepdims=True)
    ps = j.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = j * np.log2(j / (po * ps))
    return float(np.nansum(t)), c


def analyse(label, o, s):
    n = len(o)
    n1 = int(np.count_nonzero(s == 1))
    n2 = n - n1
    k1 = int(np.count_nonzero(o[s == 1]))
    k2 = int(np.count_nonzero(o[s == 2]))
    r1, r2 = k1 / n1, k2 / n2
    p0 = o.mean()

    # δ = μισή η διαφορά ρυθμού
    Delta = r2 - r1
    delta = Delta / 2.0
    # διωνυμικά σφάλματα
    se1 = math.sqrt(r1 * (1 - r1) / n1)
    se2 = math.sqrt(r2 * (1 - r2) / n2)
    se_D = math.hypot(se1, se2)
    se_delta = se_D / 2.0

    mi_meas, C = exact_mi_2x2(o, s)
    mi_fast = mi(o, s)                       # ο ίδιος υπολογιστής της αναφοράς
    denom = 2 * LN2 * p0 * (1 - p0)
    mi_pred = delta ** 2 / denom
    ratio = mi_pred / mi_meas
    # σφάλμα του mi_pred από το σφάλμα του δ (γραμμική διάδοση: 2δ·σ_δ/denom)
    se_mi_pred = abs(2 * delta * se_delta) / denom

    # σταθερά C του μοντέλου: I(k) = C·ε²·exp(-k²/τ²), C = α²/(2 ln2 p0(1-p0))
    alpha = abs(delta)          # ε ≡ 1 βαθμονόμηση
    se_alpha = se_delta
    Cmod = alpha ** 2 / denom

    print("=" * 78)
    print(label)
    print("=" * 78)
    print(f"  n = {n:,}   p0 = {p0:.8e}   p0(1-p0) = {p0*(1-p0):.8e}")
    print(f"  πίνακας 2x2 (γραμμές = click 0/1, στήλες = ρύθμιση 1/2):")
    print(f"    no-click : {int(C[0,0]):>12,}  {int(C[0,1]):>12,}")
    print(f"    click    : {int(C[1,0]):>12,}  {int(C[1,1]):>12,}")
    print(f"  ρυθμός click | S=1 : {r1:.8e}  ({k1:,} / {n1:,})")
    print(f"  ρυθμός click | S=2 : {r2:.8e}  ({k2:,} / {n2:,})")
    print(f"  Δ = r2 - r1        = {Delta:.6e} ± {se_D:.2e}   ({Delta/se_D:.0f}σ)")
    print(f"  δ(0) = Δ/2         = {delta:.6e} ± {se_delta:.2e}")
    print()
    print(f"  MI μετρημένο (ακριβές 2x2) = {mi_meas:.6e} bits/trial")
    print(f"  MI μετρημένο (mi() αναφοράς) = {mi_fast:.6e}  "
          f"[διαφορά {abs(mi_fast-mi_meas):.2e}]")
    print(f"  MI πρόβλεψη δ²/(2 ln2 p0(1-p0)) = {mi_pred:.6e} ± {se_mi_pred:.2e}")
    print(f"  λόγος πρόβλεψη/μέτρηση = {ratio:.4f}   "
          f"-> απόκλιση {abs(ratio-1)*100:.2f}%")
    ok = abs(ratio - 1) <= 0.20
    print(f"  ΤΑΙΡΙΑΖΕΙ Η ΠΡΟΣΕΓΓΙΣΗ (κριτήριο <20%); {'ΝΑΙ' if ok else 'ΟΧΙ'}")
    print()
    print(f"  ΒΑΘΜΟΝΟΜΗΣΗ (ε ≡ 1 στην κανονική κβαντομηχανική σύνδεση):")
    print(f"    α = {alpha:.6e} ± {se_alpha:.2e}  (σχετ. σφάλμα "
          f"{se_alpha/alpha*100:.2f}%)")
    print(f"    C = α²/(2 ln2 p0(1-p0)) = {Cmod:.6e} bits/trial ανά ε²")
    print()
    return dict(label=label, n=n, n1=n1, n2=n2, k1=k1, k2=k2, r1=r1, r2=r2,
                p0=float(p0), Delta=Delta, se_Delta=se_D, delta=delta,
                se_delta=se_delta, mi_measured_exact=mi_meas,
                mi_measured_fast=mi_fast, mi_predicted=mi_pred,
                se_mi_predicted=se_mi_pred, ratio=ratio,
                approximation_ok=bool(ok), alpha=alpha, se_alpha=se_alpha,
                C=Cmod, counts=C.tolist())


def main():
    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    SA, SB, OA, OB = d['SA'], d['SB'], d['OA'], d['OB']

    res = {}
    res["OA vs SA"] = analyse("OA vs SA  [ΚΥΡΙΑ ΒΑΘΜΟΝΟΜΗΣΗ]", OA, SA)
    res["OB vs SB"] = analyse("OB vs SB  [ΕΛΕΓΧΟΣ ΣΥΝΕΠΕΙΑΣ]", OB, SB)

    a = res["OA vs SA"]
    print("=" * 78)
    print("ΣΥΜΠΕΡΑΣΜΑ ΜΕΡΟΥΣ 1")
    print("=" * 78)
    print(f"  α = {a['alpha']:.4e} ± {a['se_alpha']:.1e}   "
          f"(Alice· Bob: {res['OB vs SB']['alpha']:.4e})")
    print(f"  C = {a['C']:.4e} bits/trial ανά ε²")
    print(f"  προσέγγιση 2ης τάξης: "
          f"{'ΝΑΙ' if a['approximation_ok'] else 'ΟΧΙ'} "
          f"(απόκλιση {abs(a['ratio']-1)*100:.2f}%)")

    json.dump(res, open(os.path.join(HERE, "meros1_alpha.json"), "w"), indent=2)
    print("\nΑποθηκεύτηκε: meros1_alpha.json")


if __name__ == "__main__":
    main()
