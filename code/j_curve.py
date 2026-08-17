"""
Καμπύλη Eberhard J(k): ρυθμίσεις από το trial i, αποτελέσματα από το i+k.

J = N(11|a1b1) - N(10|a1b2) - N(01|a2b1) - N(11|a2b2)   <= 0  (τοπικός ρεαλισμός)

Στο k=0 περιμένουμε παραβίαση (J>0). Σε k!=0 ρυθμίσεις και αποτελέσματα
αποσυνδέονται, οπότε ΔΕΝ πρέπει να υπάρχει παραβίαση.

σ = sqrt(άθροισμα των τεσσάρων μετρήσεων)  — διάδοση σφάλματος Poisson.
Ίδιος ορισμός με το αρχικό +11,7σ.

ΠΡΟΣΟΧΗ ΣΤΗΝ ΟΛΙΣΘΗΣΗ: το πρόσημο του k ελέγχεται με ρητό τεστ (δες
--selftest): φτιάχνουμε δεδομένα όπου η παραβίαση είναι ΤΕΧΝΗΤΑ στο k=+3
και επιβεβαιώνουμε ότι η κορυφή εμφανίζεται εκεί.
"""
import argparse, json
import numpy as np

LAGS = list(range(-50, 51)) + [-10000, -1000, -100, 100, 1000, 10000]


def align_so(SA, SB, OA, OB, k):
    """Ρυθμίσεις από trial i, αποτελέσματα από trial i+k."""
    if k > 0:
        return SA[:-k], SB[:-k], OA[k:], OB[k:]
    if k < 0:
        return SA[-k:], SB[-k:], OA[:k], OB[:k]
    return SA, SB, OA, OB


def eberhard(sa, sb, oa, ob):
    """J και σ, με ΜΙΑ διέλευση (bincount σε 16 κάδους)."""
    idx = (((sa - 1).astype(np.intp) * 2 + (sb - 1)) * 4
           + oa.astype(np.intp) * 2 + ob)
    c = np.bincount(idx, minlength=16).reshape(2, 2, 4)   # [SA-1, SB-1, (OA,OB)]
    # (OA,OB): 0=00, 1=01, 2=10, 3=11
    n11_a1b1 = int(c[0, 0, 3])
    n10_a1b2 = int(c[0, 1, 2])
    n01_a2b1 = int(c[1, 0, 1])
    n11_a2b2 = int(c[1, 1, 3])
    J = n11_a1b1 - n10_a1b2 - n01_a2b1 - n11_a2b2
    sd = np.sqrt(n11_a1b1 + n10_a1b2 + n01_a2b1 + n11_a2b2)
    return J, float(sd), (n11_a1b1, n10_a1b2, n01_a2b1, n11_a2b2)


def selftest():
    """Έλεγχος ολίσθησης: τεχνητή παραβίαση στο k=+3 πρέπει να δώσει κορυφή
    ΑΚΡΙΒΩΣ στο k=+3, όχι στο -3."""
    rng = np.random.default_rng(0)
    n = 200_000
    SA = rng.integers(1, 3, n).astype(np.int8)
    SB = rng.integers(1, 3, n).astype(np.int8)
    OA = (rng.random(n) < 0.007).astype(np.int8)
    OB = (rng.random(n) < 0.007).astype(np.int8)
    K = 3
    # στα trials όπου settings[i]==(1,1), βάζουμε click ΚΑΙ στους δύο στο i+K
    m = np.where((SA == 1) & (SB == 1))[0]
    m = m[m + K < n][:20000]
    OA[m + K] = 1
    OB[m + K] = 1
    best = None
    for k in range(-6, 7):
        sa, sb, oa, ob = align_so(SA, SB, OA, OB, k)
        J, sd, _ = eberhard(sa, sb, oa, ob)
        z = J / sd if sd else 0
        if best is None or z > best[1]:
            best = (k, z)
    print(f"SELFTEST ολίσθησης: τεχνητή παραβίαση στο k=+{K}, "
          f"κορυφή βρέθηκε στο k={best[0]} ({best[1]:+.1f}σ)")
    print("  ->", "ΠΕΡΑΣΕ" if best[0] == K else f"ΑΠΕΤΥΧΕ (αναμενόταν {K})")
    return best[0] == K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--png", default="j_curve.png")
    ap.add_argument("--label", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if not selftest():
        raise SystemExit("Ο έλεγχος ολίσθησης απέτυχε — μη συνεχίσεις.")
    print()

    d = np.load(a.path)
    SA, SB, OA, OB = d['SA'], d['SB'], d['OA'], d['OB']
    lags = sorted(LAGS)
    print(f"Trials: {len(SA):,}   k τιμές: {len(lags)}\n")

    rows = []
    for k in lags:
        sa, sb, oa, ob = align_so(SA, SB, OA, OB, k)
        J, sd, counts = eberhard(sa, sb, oa, ob)
        rows.append({"k": k, "J": int(J), "sigma": float(sd),
                     "z": float(J / sd) if sd else 0.0, "counts": counts})

    z = np.array([r["z"] for r in rows])
    ks = np.array([r["k"] for r in rows])
    i0 = int(np.where(ks == 0)[0][0])
    nz = np.array([i for i in range(len(ks)) if ks[i] != 0])

    if not a.quiet:
        print(f"{'k':>7} {'J':>12} {'σ':>9} {'J/σ':>9}")
        print("-" * 42)
        for r in rows:
            mark = "  <- k=0" if r["k"] == 0 else ""
            if abs(r["k"]) <= 6 or abs(r["k"]) >= 100 or r["k"] % 10 == 0:
                print(f"{r['k']:>7} {r['J']:>12,} {r['sigma']:>9.1f} "
                      f"{r['z']:>+9.2f}{mark}")
        print("   (τυπώνονται δείγματα· όλα στο json)\n")

    print(f"k=0:      J = {rows[i0]['J']:+,}   J/σ = {z[i0]:+.2f}")
    print(f"k!=0:     μέγιστο J/σ = {z[nz].max():+.2f} στο k = {ks[nz][int(np.argmax(z[nz]))]}")
    print(f"          ελάχιστο J/σ = {z[nz].min():+.2f}")
    print(f"          μέσος J/σ    = {z[nz].mean():+.2f}")
    pos = [int(ks[i]) for i in nz if rows[i]["J"] > 0]
    print(f"k!=0 με J > 0 (οποιαδήποτε παραβίαση): {len(pos)} / {len(nz)}"
          + (f"  -> {pos}" if pos else ""))

    # ---- γράφημα ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    near = [r for r in rows if abs(r["k"]) <= 50]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="0.6", lw=1)
    ax.plot([r["k"] for r in near], [r["z"] for r in near],
            marker="o", ms=3, lw=1, color="#1f77b4")
    ax.scatter([0], [z[i0]], color="#d62728", zorder=5, s=60,
               label=f"k=0:  {z[i0]:+.1f}σ  (παραβίαση Bell)")
    ax.set_xlabel("k   (ρυθμίσεις από trial i, αποτελέσματα από trial i+k)")
    ax.set_ylabel("J / σ")
    t = "Eberhard J(k)" + (f" — {a.label}" if a.label else "")
    ax.set_title(t + "\nπαραβίαση μόνο όταν ρυθμίσεις και αποτελέσματα "
                     "είναι από το ΙΔΙΟ trial")
    ax.legend(loc="center left", fontsize=9)
    ax.annotate(f"k≠0:  όλα ≈ {z[nz].mean():.0f}σ\nκαμία παραβίαση — J<0 και στα "
                f"{len(nz)}", xy=(0.78, 0.62), xycoords="axes fraction",
                ha="center", fontsize=9, color="0.35")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.png, dpi=140)
    print(f"\nΓράφημα: {a.png}")

    out = a.png.replace(".png", "_data.json")
    with open(out, "w") as f:
        json.dump({"rows": rows, "k0_J": rows[i0]["J"], "k0_z": float(z[i0]),
                   "max_z_nonzero": float(z[nz].max()),
                   "max_z_nonzero_k": int(ks[nz][int(np.argmax(z[nz]))]),
                   "n_positive_J_nonzero": len(pos)}, f, indent=2)
    print(f"Δεδομένα: {out}")


if __name__ == "__main__":
    main()
