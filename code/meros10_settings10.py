"""
ΜΕΡΟΣ 10 — Ο ΕΛΕΓΧΟΣ ΦΑΝΤΑΣΜΑΤΟΣ ΣΕ ΟΛΟΥΣ ΤΟΥΣ ΔΕΚΑ ΠΑΛΜΟΥΣ

Το Μέρος 8 έτρεξε μόνο στον 28297. Αφού το κύριο αποτέλεσμα είναι πλέον το
κοινό όριο των δέκα, μια συσχετισμένη γεννήτρια σε ΟΠΟΙΟΝΔΗΠΟΤΕ παλμό θα
μόλυνε το κοινό όριο. Εδώ τρέχει ο ίδιος έλεγχος και στους δέκα.

ΔΥΟ ΣΥΓΚΡΙΣΕΙΣ, ΚΑΙ ΟΙ ΔΥΟ ΧΡΕΙΑΖΟΝΤΑΙ
  (α) ανά παλμό:  ε_phantom,p(τ) = Σ W ρ_p / Q   έναντι του κοινού ορίου
  (β) σωστά σταθμισμένο: το φάντασμα μπαίνει στο κοινό όριο με τα ΙΔΙΑ βάρη
      αντίστροφης διασποράς με τα δεδομένα:

        ε_phantom,joint = Σ_p (ε_phantom,p / σ_p²) / Σ_p (1/σ_p²)

      Αυτό είναι το φυσικά σχετικό μέγεθος: πόσο μετατοπίζει το φάντασμα την
      κοινή εκτίμηση. Τα σ_p έρχονται από το Μέρος 9.

Αν κάποιος παλμός δείξει |ρ| πάνω από το κατώφλι, το γράφουμε ΠΡΩΤΟ.
"""
import json, math, os
import numpy as np

from load_curby import read_file
from meros2_injection import scan, mi_and_delta
from meros5_asym import kernel_W, KERNELS
from meros8_settings import xcorr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dedomena_curby")
LN2 = math.log(2.0)
K = 10_000
ROUNDS = [1000, 15000, 22000, 23000, 26000, 28293, 28294, 28295, 28296, 28297]


def main():
    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    m9 = json.load(open(os.path.join(HERE, "meros9_joint.json")))
    z_thr = m5["z_thr"]
    taus = np.array(m5["taus"])
    kax = np.arange(-K, K + 1, dtype=np.float64)
    Wcache = {kn: np.stack([kernel_W(kn, kax, float(t)) for t in taus])
              for kn in KERNELS}
    Qcache = {kn: (Wcache[kn] ** 2).sum(axis=1) for kn in KERNELS}

    print("=" * 78)
    print("ΜΕΡΟΣ 10 — ΣΥΣΧΕΤΙΣΗ ΡΥΘΜΙΣΕΩΝ ΣΕ ΟΛΟΥΣ ΤΟΥΣ ΔΕΚΑ ΠΑΛΜΟΥΣ")
    print("=" * 78)
    print(f"  |k| ≤ {K:,}   κατώφλι {z_thr:.3f}σ\n")
    print(f"  {'γύρος':>7} {'max |ρ|':>11} {'σε lag':>9} {'σ':>7} "
          f"{'>κατώφλι':>9}   {'max|ρ_AA|':>10} {'max|ρ_BB|':>10}")

    rows = []
    rho_by_round = {}
    for rnd in ROUNDS:
        data, _ = read_file(os.path.join(DATA, f"curby_round_{rnd}.bin"))
        SA = data['SA']; SB = data['SB']
        del data
        n = len(SA)
        se = 1.0 / math.sqrt(n)
        sa = np.where(SA == 2, 1.0, -1.0)
        sb = np.where(SB == 2, 1.0, -1.0)
        ks, r_ab = xcorr(sa, sb, K)
        _, r_aa = xcorr(sa, sa, K); r_aa[ks == 0] = 0.0
        _, r_bb = xcorr(sb, sb, K); r_bb[ks == 0] = 0.0
        j = int(np.argmax(np.abs(r_ab)))
        nab = int((np.abs(r_ab) > z_thr * se).sum())

        # αμοιβαία πληροφορία των ρυθμίσεων, ίδιο κατώφλι
        a1 = (SA == 1).astype(np.int8); b1 = (SB == 1).astype(np.int8)
        _, n11, A1, B1, nk, _ = scan(a1, b1, K)
        MI, _ = mi_and_delta(n11, A1, B1, nk)
        G = 2 * nk * LN2 * MI
        nab_mi = int((G > z_thr ** 2).sum())

        print(f"  {rnd:>7} {abs(r_ab[j]):>11.3e} {ks[j]:>+9d} "
              f"{abs(r_ab[j])/se:>7.2f} {nab:>9d}   "
              f"{np.abs(r_aa).max():>10.3e} {np.abs(r_bb).max():>10.3e}",
              flush=True)
        rows.append(dict(round=rnd, n=n, se=se,
                         max_abs_rho=float(abs(r_ab[j])), at_lag=int(ks[j]),
                         sigma=float(abs(r_ab[j]) / se), n_above=nab,
                         max_rho_AA=float(np.abs(r_aa).max()),
                         max_rho_BB=float(np.abs(r_bb).max()),
                         max_mi=float(MI.max()), max_G=float(G.max()),
                         n_above_mi=nab_mi))
        rho_by_round[rnd] = r_ab
        del SA, SB, sa, sb, r_aa, r_bb, a1, b1

    n_bad = sum(r["n_above"] + r["n_above_mi"] for r in rows)
    print(f"\n  ΣΥΝΟΛΟ πάνω από το κατώφλι: {n_bad}  "
          f"(συσχέτιση {sum(r['n_above'] for r in rows)}, "
          f"αμοιβαία πληροφορία {sum(r['n_above_mi'] for r in rows)})")
    if n_bad:
        print("\n  *** ΣΤΑΜΑΤΑ: βρέθηκε συσχέτιση ρυθμίσεων ***")

    # ---------- φάντασμα ανά παλμό και σταθμισμένο ----------
    print("\n" + "=" * 78)
    print("ΦΑΝΤΑΣΜΑ ΕΝΑΝΤΙ ΤΟΥ ΚΟΙΝΟΥ ΟΡΙΟΥ")
    print("=" * 78)
    out = {"z_thr": z_thr, "rounds": rows, "taus": taus.tolist()}
    worst_single = 0.0; worst_joint = 0.0; where = None
    detail = {}
    for pair in ("OA vs SB", "OB vs SA"):
        detail[pair] = {}
        for kn in KERNELS:
            W = Wcache[kn]; Q = Qcache[kn]
            # ε_phantom ανά παλμό: (n_pulse × n_tau)
            eph = np.stack([np.abs(W @ rho_by_round[r]) / Q for r in ROUNDS])
            sig = np.array([m9["per_pulse"][pair][kn][i]["sigma"]
                            for i in range(len(ROUNDS))])
            # τα per_pulse του Μέρους 9 είναι στη σειρά των ROUNDS
            w = 1.0 / sig ** 2
            eph_joint = (eph * w).sum(axis=0) / w.sum(axis=0)
            excl_joint = np.array(m9["joint"][pair][kn]["eps_excl"])
            r_single = (eph / excl_joint).max()
            r_joint = (eph_joint / excl_joint).max()
            worst_single = max(worst_single, float(r_single))
            if float(r_joint) > worst_joint:
                worst_joint = float(r_joint); where = (pair, kn)
            detail[pair][kn] = dict(
                eps_phantom_joint=eph_joint.tolist(),
                eps_excl_joint=excl_joint.tolist(),
                max_ratio_single_pulse=float(r_single),
                max_ratio_joint=float(r_joint))
    print(f"  χειρότερος λόγος, ΜΕΜΟΝΩΜΕΝΟΣ παλμός / κοινό όριο: "
          f"{worst_single:.4f}   ({1/worst_single:.0f}× κάτω)")
    print(f"  χειρότερος λόγος, ΣΤΑΘΜΙΣΜΕΝΟ φάντασμα / κοινό όριο: "
          f"{worst_joint:.4f}   ({1/worst_joint:.0f}× κάτω)   [{where[0]}, {where[1]}]")
    out["worst_ratio_single_pulse"] = worst_single
    out["worst_ratio_joint"] = worst_joint
    out["detail"] = detail
    out["clean"] = bool(n_bad == 0)
    json.dump(out, open(os.path.join(HERE, "meros10_settings10.json"), "w"),
              indent=2)
    print("\nΑποθηκεύτηκε: meros10_settings10.json")


if __name__ == "__main__":
    main()
