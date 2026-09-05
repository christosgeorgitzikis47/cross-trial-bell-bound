"""
PART 6.1 - p0 IN AN ASYMMETRIC CONFIGURATION (peer review objection #1)

The objection: "in an Eberhard configuration the two settings have click
rates differing by a factor 1.79. Which p0 enters the formula? Their average,
as a rough approximation?"

THREE THINGS ARE CHECKED HERE

(a) p0 is the MARGINAL click probability: p0 = (k1+k2)/n, the margin of the
    2x2 table. It is NOT "an average used as an approximation". Because the
    settings are 50/50, p0 = (r1+r2)/2 -- identically so if n1 = n2 exactly.
    How far n1/n2 departs from 1/2, and what that costs the identity, is
    measured.

(b) The EXACT MI of the 2x2 table (a sum of 4 p log p terms, no expansion)
    next to the approximate delta^2/(2 ln2 p0(1-p0)).

(c) THE MAIN POINT: where the approximation is actually USED. The eps_excl
    map is rebuilt from the JSON using ONLY (T, sigma_T, alpha, Q) and
    compared with the published one. If they agree to the last digit, then
    the approximation -- and therefore p0 -- enters the bound nowhere.
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2.0)


def exact_mi_2x2(k1, k2, n1, n2):
    """EXACT MI (bits) of the 2x2 table from the raw counts. No expansion."""
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
    print("PART 6.1 - p0: A MARGINAL, NOT AN APPROXIMATION")
    print("=" * 78)

    for label in ("OA vs SA", "OB vs SB"):
        d = cal[label]
        n1, n2 = d["n1"], d["n2"]
        k1, k2 = int(d["counts"][1][0]), int(d["counts"][1][1])
        n = n1 + n2
        r1, r2 = k1 / n1, k2 / n2

        p0_marginal = (k1 + k2) / n              # DEFINITION: the margin
        p0_mean = (r1 + r2) / 2                  # identical if n1 = n2
        frac1 = n1 / n

        print(f"\n--- {label} ---")
        print(f"  n1 = {n1:,}  n2 = {n2:,}   n1/n = {frac1:.9f}  "
              f"(departure from 1/2: {frac1-0.5:+.2e})")
        print(f"  r1 = {r1:.9e}   r2 = {r2:.9e}   r2/r1 = {r2/r1:.4f}")
        print(f"  p0 (marginal)   = {p0_marginal:.12e}")
        print(f"  (r1+r2)/2       = {p0_mean:.12e}")
        print(f"  relative difference = "
              f"{abs(p0_mean-p0_marginal)/p0_marginal:.3e}"
              f"   [identically 0 if n1 = n2]")
        print(f"  analytically: p0 = (n1 r1 + n2 r2)/n, so p0 - (r1+r2)/2 = "
              f"(n1-n2)/(2n)*(r1-r2) = "
              f"{(n1-n2)/(2*n)*(r1-r2):+.3e}")

        mi_exact = exact_mi_2x2(k1, k2, n1, n2)
        delta = (r2 - r1) / 2
        mi_approx = delta ** 2 / (2 * LN2 * p0_marginal * (1 - p0_marginal))
        # the same approximation had someone used a wrong p0
        alts = {"sqrt(r1r2)": math.sqrt(r1 * r2), "r1": r1, "r2": r2}

        print(f"\n  MI EXACT  (4 p log p terms, no expansion) = "
              f"{mi_exact:.9e} bits/trial")
        print(f"  MI APPROXIMATE  delta^2/(2 ln2 p0(1-p0))  = "
              f"{mi_approx:.9e} bits/trial")
        print(f"  ratio approximate/exact = {mi_approx/mi_exact:.6f} "
              f"({100*(mi_approx/mi_exact-1):+.2f}%)")
        print(f"  (the JSON recorded: exact {d['mi_measured_exact']:.9e}, "
              f"predicted {d['mi_predicted']:.9e}) -> "
              f"{'AGREE' if abs(mi_exact-d['mi_measured_exact'])<1e-15 else 'DISAGREE'}")
        for nm, v in alts.items():
            m = delta ** 2 / (2 * LN2 * v * (1 - v))
            print(f"    had p0 been set to {nm:>10}: "
                  f"ratio {m/mi_exact:.4f}")

    # ---------- (c) the map does NOT go through the approximation ----------
    print("\n" + "=" * 78)
    print("WHAT THE eps_excl MAP DEPENDS ON - REBUILT FROM THE RAW QUANTITIES")
    print("=" * 78)
    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    z_thr = m5["z_thr"]
    worst = 0.0
    for pair, alpha in (("OA vs SB", m5["alpha_A"]), ("OB vs SA", m5["alpha_B"])):
        for kn in m5["kernels"]:
            P = m5["pairs"][pair][kn]
            T = np.array(P["T"]); sT = np.array(P["sigma_T"])
            Q = np.array(P["Q"])
            # ONLY these: T, sigma_T, alpha, Q, z_thr. No p0, no C, no MI.
            rebuilt = np.abs(T / (alpha * Q)) + z_thr * sT / (alpha * Q)
            published = np.array(P["eps_excl"])
            worst = max(worst, float(np.abs(rebuilt / published - 1).max()))
    print(f"  eps_excl = |T|/(alpha Q) + z*sigma_T/(alpha Q)")
    print(f"  largest relative difference, rebuilt vs published, over "
          f"{2*4*len(m5['taus'])} points: {worst:.2e}")
    print(f"  -> {'IDENTICAL' if worst < 1e-12 else 'NOT identical'}")
    print("\n  The ingredients and where they come from:")
    print("    T(tau) = sum W(k)*delta-hat(k) - measured click rates, 2x2 per lag")
    print("    sigma_T = max(empirical from shuffles, binomial)")
    print("    alpha  = delta(0) = (r2-r1)/2  - measured rate difference at lag 0")
    print("    Q(tau) = sum W(k)^2            - purely geometric")
    print("  NONE of these contains p0 or the second-order expansion.")
    print("  The approximation is used ONLY as a consistency check (part 1)")
    print("  and in the SECONDARY eps_excl(I) column of report #6, which")
    print("  enters neither the map nor the figure.")

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
    print("\nSaved: meros6_p0.json")


if __name__ == "__main__":
    main()
