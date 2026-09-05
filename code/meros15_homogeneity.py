"""
PART 15 - HOMOGENEITY OF eps ACROSS THE TEN PULSES + THRESHOLD QUANTIFICATION

(a) The combination of section 6.4 presupposes a COMMON eps across the ten
    pulses (objection of an outside reviewer: 22 months, documented changes
    to the apparatus). The assumption is tested with the same numbers:

        Q_het = sum_p (eps-hat_p - eps-hat_joint)^2 / sigma_p^2  ~  chi^2(9)
        if eps is common

    per (pair, kernel, tau) -- 208 points. Slight under-dispersion is
    expected, because sigma_T is the conservative max(empirical, analytic).

(b) How much the map would tighten under the matched threshold for 208 tests
    (z = 3.672) instead of the borrowed 4.848 -- for the one sentence in
    section 6.3.

If the heterogeneity comes out significant, we write it FIRST and STOP.
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
    print("PART 15a - HOMOGENEITY TEST: Q_het ~ chi^2(9) UNDER A COMMON eps")
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

    print(f"  {n_pts} points (2 pairs x 4 kernels x {len(taus)} tau), "
          f"dof = {DOF}")
    print(f"  mean Q_het   = {qh.mean():.2f}   (theory {DOF}.00)")
    print(f"  median       = {np.median(qh):.2f}   "
          f"(theory {chi2.ppf(.5, DOF):.2f})")
    print(f"  maximum      = {worst['Q_het']:.2f}  [{worst['pair']}, "
          f"{worst['kernel']}, tau = {worst['tau']:g}]")
    print(f"    p per point = {worst['p']:.4f}   Bonferroni x {n_pts}: "
          f"p = {p_fam:.2f}")
    n99 = int((qh > q99).sum())
    print(f"  above the 99% quantile ({q99:.1f}): {n99} "
          f"(expected {0.01*n_pts:.1f} -- the points share the same ten "
          f"delta-hat fields, so they are strongly correlated)")
    verdict = p_fam > 0.05
    print(f"\n  VERDICT: {'CONSISTENT with a common eps (here zero)' if verdict else '*** HETEROGENEITY - STOP ***'}")

    print("\n" + "=" * 78)
    print("PART 15b - THE MATCHED THRESHOLD FOR 208 TESTS VS THE BORROWED ONE")
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
    print(f"  z borrowed = {z_bor:.3f}   z matched (m = {n_pts}) = {z_mat:.3f}")
    print(f"  ratio of bounds matched/borrowed: mean {sh.mean():.3f}   "
          f"range [{sh.min():.3f}, {sh.max():.3f}]")
    print(f"  -> the bounds would tighten by {100*(1-sh.max()):.0f}%-"
          f"{100*(1-sh.min()):.0f}% (mean {100*(1-sh.mean()):.0f}%)")

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
    print("\nSaved: meros15_homogeneity.json")


if __name__ == "__main__":
    main()
