"""
ΜΕΡΟΣ 15 — ΟΜΟΙΟΓΕΝΕΙΑ ΤΟΥ ε ΣΤΟΥΣ ΔΕΚΑ ΠΑΛΜΟΥΣ + ΠΟΣΟΤΙΚΟΠΟΙΗΣΗ ΚΑΤΩΦΛΙΟΥ

(α) Η σύνθεση του §6.4 προϋποθέτει ΚΟΙΝΟ ε στους δέκα παλμούς (ένσταση
    εξωτερικού βαθμολογητή: 22 μήνες, τεκμηριωμένες αλλαγές συσκευής).
    Η υπόθεση ελέγχεται από τα ίδια νούμερα:

        Q_het = Σ_p (ε̂_p − ε̂_joint)² / σ_p²  ~  χ²(9)   αν το ε είναι κοινό

    ανά (ζεύγος, πυρήνας, τ) — 208 σημεία. Υπο-διασπορά αναμένεται ελαφρά,
    γιατί το σ_T είναι το συντηρητικό max(εμπειρικό, αναλυτικό).

(β) Πόσο θα έσφιγγε ο χάρτης με το matched κατώφλι των 208 τεστ (z = 3,672)
    αντί του δανεικού 4,848 — για τη μία πρόταση του §6.3.

Αν η ετερογένεια βγει σημαντική, το γράφουμε ΠΡΩΤΟ και ΣΤΑΜΑΤΑΜΕ.
"""
import json, math, os
import numpy as np
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
N_PULSES = 10
DOF = N_PULSES - 1


def main():
    m9 = json.load(open(os.path.join(HERE, "meros9_joint.json")))
    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    taus = m5["taus"]
    kers = m5["kernels"]
    z_bor = m5["z_thr"]

    print("=" * 78)
    print("ΜΕΡΟΣ 15α — ΕΛΕΓΧΟΣ ΟΜΟΙΟΓΕΝΕΙΑΣ: Q_het ~ χ²(9) ΥΠΟ ΚΟΙΝΟ ε")
    print("=" * 78)

    pts = []
    for pr in ("OA vs SB", "OB vs SA"):
        for kn in kers:
            for j, t in enumerate(taus):
                eps = np.array([m9["per_pulse"][pr][kn][i]["eps_hat"][j]
                                for i in range(N_PULSES)])
                sig = np.array([m9["per_pulse"][pr][kn][i]["sigma"][j]
                                for i in range(N_PULSES)])
                w = 1.0 / sig ** 2
                ej = float((eps * w).sum() / w.sum())
                q = float((((eps - ej) ** 2) * w).sum())
                pts.append(dict(pair=pr, kernel=kn, tau=float(t), Q_het=q,
                                p=float(chi2.sf(q, DOF))))
    qh = np.array([p["Q_het"] for p in pts])
    n_pts = len(pts)
    q99 = float(chi2.ppf(0.99, DOF))
    worst = max(pts, key=lambda p: p["Q_het"])
    p_fam = min(1.0, n_pts * worst["p"])

    print(f"  {n_pts} σημεία (2 ζεύγη × 4 πυρήνες × {len(taus)} τ), dof = {DOF}")
    print(f"  μέσος Q_het   = {qh.mean():.2f}   (θεωρία {DOF}.00)")
    print(f"  διάμεσος      = {np.median(qh):.2f}   (θεωρία {chi2.ppf(.5, DOF):.2f})")
    print(f"  μέγιστο       = {worst['Q_het']:.2f}  [{worst['pair']}, "
          f"{worst['kernel']}, τ = {worst['tau']:g}]")
    print(f"    p ανά σημείο = {worst['p']:.4f}   Bonferroni × {n_pts}: "
          f"p = {p_fam:.2f}")
    n99 = int((qh > q99).sum())
    print(f"  πάνω από το 99% ποσοστημόριο ({q99:.1f}): {n99} "
          f"(αναμενόμενα {0.01*n_pts:.1f} — τα σημεία μοιράζονται τα ίδια "
          f"δέκα δ̂ πεδία, άρα είναι ισχυρά συσχετισμένα)")
    verdict = p_fam > 0.05
    print(f"\n  ΕΤΥΜΗΓΟΡΙΑ: {'ΣΥΜΒΑΤΟ με κοινό ε (εδώ μηδέν)' if verdict else '*** ΕΤΕΡΟΓΕΝΕΙΑ — ΣΤΑΜΑΤΑ ***'}")

    print("\n" + "=" * 78)
    print("ΜΕΡΟΣ 15β — MATCHED ΚΑΤΩΦΛΙ 208 ΤΕΣΤ ΑΝΤΙ ΤΟΥ ΔΑΝΕΙΚΟΥ")
    print("=" * 78)
    z_mat = float(math.sqrt(chi2.ppf(1 - 0.05 / n_pts, 1)))
    sh = []
    for pr in ("OA vs SB", "OB vs SA"):
        for kn in kers:
            P = m5["pairs"][pr][kn]
            e_now = np.array(P["eps_excl"])
            e_mat = np.abs(np.array(P["eps_hat"])) + z_mat * np.array(P["sigma_eps"])
            sh.append(e_mat / e_now)
    sh = np.concatenate(sh)
    print(f"  z δανεικό = {z_bor:.3f}   z matched (m = {n_pts}) = {z_mat:.3f}")
    print(f"  λόγος ορίων matched/δανεικό: μέσος {sh.mean():.3f}   "
          f"εύρος [{sh.min():.3f}, {sh.max():.3f}]")
    print(f"  -> τα όρια θα έσφιγγαν κατά {100*(1-sh.max()):.0f}%–"
          f"{100*(1-sh.min()):.0f}% (μέσο {100*(1-sh.mean()):.0f}%)")

    out = dict(dof=DOF, n_points=n_pts,
               mean_Q_het=float(qh.mean()), median_Q_het=float(np.median(qh)),
               max_Q_het=float(worst["Q_het"]),
               max_at=dict(pair=worst["pair"], kernel=worst["kernel"],
                           tau=worst["tau"]),
               p_max_single=float(worst["p"]), p_max_family=float(p_fam),
               n_above_q99=n99, q99=q99, homogeneous=bool(verdict),
               z_borrowed=z_bor, z_matched=z_mat,
               shrink_mean=float(sh.mean()), shrink_min=float(sh.min()),
               shrink_max=float(sh.max()), points=pts)
    json.dump(out, open(os.path.join(HERE, "meros15_homogeneity.json"), "w"),
              indent=2)
    print("\nΑποθηκεύτηκε: meros15_homogeneity.json")


if __name__ == "__main__":
    main()
