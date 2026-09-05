"""
PART 6.2 - WHERE THE +3 TO 7% SYSTEMATIC COMES FROM (peer review objection #2)

THE SUSPICION (from the review): the sigma_T of the map comes from shuffles of
the REAL data, which carry outcome autocorrelation (deadtime at lag +/-1). The
INJECTED outcomes are fresh Bernoulli, without that autocorrelation. Less
noise -> larger z.

WHY THE MECHANISM IS THE RIGHT ONE (algebraically, before measuring)
    Var(T) = sum sum W(k)W(k') Cov(delta-hat(k), delta-hat(k')).
    Under a random permutation of the settings, Cov(s_a, s_b) = p(1-p) only
    for a = b. The a = b term requires i+k = j+k', that is j = i + (k-k'),
    and gives
        Cov(delta-hat(k), delta-hat(k')) ~ sum_i O_i*O_{i+(k-k')}
    = exactly the AUTOCORRELATION OF THE OUTCOMES at lag k-k'.
    The analytic sigma_T assumes those terms vanish. Therefore:
    autocorrelation in O -> empirical sigma_T > analytic. Exactly what we see.

WHAT IS MEASURED HERE
  (a) autocorrelation of the REAL OA/OB at lags 1 to 5,
  (b) the same on the INJECTED data (fresh Bernoulli),
  (c) sigma_T empirical/analytic from shuffles of the INJECTED data
      -- if the mechanism is right, the ratio should fall to 1 here,
         while on the real data it is > 1,
  (d) the decomposition of the systematic: z_meas/z_pred against
      sigma_T(map)/sigma_T(analytic).
"""
import argparse, json, math, os
import numpy as np

from meros2_injection import scan, mi_and_delta
from meros3_map import sigma_delta
from meros5_asym import kernel_W
from meros5_verify import build_F_kernel

HERE = os.path.dirname(os.path.abspath(__file__))


def autocorr_binary(x, lags):
    """Pearson r of the binary x at given lags, with error ~1/sqrt(n)."""
    xf = x.astype(np.float64)
    p = xf.mean(); v = p * (1 - p)
    out = {}
    for L in lags:
        c = float(np.dot(xf[:-L], xf[L:]) / (len(xf) - L)) - p * p
        out[L] = c / v
    return out, 1.0 / math.sqrt(len(x))


def sigma_T_ratio(O, s1, Wmat, K, shuffles, rng):
    """empirical/analytic sigma_T for given filters, by shuffling settings."""
    _, n11, A1, B1, nk, _ = scan(O, s1, K)
    sd = sigma_delta(A1, B1, nk)
    sT_ana = np.sqrt(Wmat ** 2 @ sd ** 2)
    sh = s1.copy()
    T = np.empty((shuffles, Wmat.shape[0]))
    for i in range(shuffles):
        rng.shuffle(sh)
        _, n11s, A1s, B1s, nks, _ = scan(O, sh, K)
        _, ds = mi_and_delta(n11s, A1s, B1s, nks)
        T[i] = Wmat @ ds
    sT_emp = T.std(axis=0, ddof=1)
    return sT_emp, sT_ana, sT_emp / sT_ana


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--K", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--cases", nargs="+", default=["future:30", "future:300"])
    ap.add_argument("--out", default="meros6_systematic")
    a = ap.parse_args()

    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    ver = json.load(open(os.path.join(HERE, "meros5_verify.json")))
    cal = json.load(open(os.path.join(HERE, "meros1_alpha.json")))["OA vs SA"]
    alpha, r1, r2 = cal["alpha"], cal["r1"], cal["r2"]
    taus = np.array(m5["taus"])
    P = m5["pairs"]["OA vs SB"]

    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    SA, SB, OA, OB = d['SA'], d['SB'], d['OA'], d['OB']
    n = len(SA)
    SB1 = (SB == 1).astype(np.int8)
    S = np.where(SB == 2, 1.0, -1.0)
    lam0 = np.where(SA == 1, r1, r2)
    rng = np.random.default_rng(a.seed)
    kax = np.arange(-a.K, a.K + 1, dtype=np.float64)

    print("=" * 78)
    print("PART 6.2 - WHERE THE SYSTEMATIC COMES FROM")
    print("=" * 78)

    # ---------- (a) autocorrelation of the real outcomes ----------
    lags = [1, 2, 3, 4, 5]
    acA, se = autocorr_binary(OA, lags)
    acB, _ = autocorr_binary(OB, lags)
    print(f"\n(a) Autocorrelation of the REAL outcomes (error +/-{se:.2e}):")
    print(f"    {'lag':>5} {'OA':>12} {'sig':>7} {'OB':>12} {'sig':>7}")
    for L in lags:
        print(f"    {L:>5} {acA[L]:>12.3e} {acA[L]/se:>7.1f} "
              f"{acB[L]:>12.3e} {acB[L]/se:>7.1f}")
    out = {"autocorr_real_OA": acA, "autocorr_real_OB": acB, "se_autocorr": se,
           "cases": {}}

    # ---------- (b)+(c) per case ----------
    for case in a.cases:
        kn, tv = case.split(":")
        tau = float(tv)
        j = int(np.argmin(np.abs(taus - tau)))
        eps = P[kn]["eps_excl"][j]
        Wmat = kernel_W(kn, kax, tau)[None, :]

        print("\n" + "-" * 78)
        print(f"CASE {kn}, tau = {tau:g}   (injection at eps_excl = {eps:.4e})")
        print("-" * 78)

        F, half = build_F_kernel(S, kn, tau, a.K)
        lam = lam0 + alpha * eps * F
        Oinj = (rng.random(n) < np.clip(lam, 0, 1)).astype(np.int8)
        del F

        aci, _ = autocorr_binary(Oinj, lags)
        print(f"(b) autocorrelation of the INJECTED: " +
              "  ".join(f"lag{L} {aci[L]:+.2e} ({aci[L]/se:+.1f} sig)"
                        for L in lags))

        print(f"(c) sigma_T from {a.shuffles} shuffles...", flush=True)
        e_r, a_r, ratio_real = sigma_T_ratio(OA, SB1, Wmat, a.K, a.shuffles, rng)
        e_i, a_i, ratio_inj = sigma_T_ratio(Oinj, SB1, Wmat, a.K, a.shuffles, rng)
        print(f"    REAL data:      empirical/analytic = "
              f"{ratio_real[0]:.4f}")
        print(f"    INJECTED data:  empirical/analytic = "
              f"{ratio_inj[0]:.4f}")
        print(f"    (the map, 400 shuffles: "
              f"{P[kn]['sigma_T_emp'][j]/P[kn]['sigma_T_ana'][j]:.4f})")
        se_ratio = 1.0 / math.sqrt(2 * (a.shuffles - 1))
        print(f"    statistical error of the ratio: +/-{se_ratio:.4f}")
        out["cases"][case] = dict(
            tau=tau, kernel=kn, eps=eps,
            autocorr_injected={str(k): v for k, v in aci.items()},
            ratio_real=float(ratio_real[0]), ratio_inj=float(ratio_inj[0]),
            ratio_map=float(P[kn]['sigma_T_emp'][j] / P[kn]['sigma_T_ana'][j]),
            se_ratio=se_ratio)
        del Oinj

    # ---------- (d) decomposition of the systematic ----------
    print("\n" + "=" * 78)
    print("(d) DECOMPOSITION: z_meas/z_pred  against  sT(map)/sT(analytic)")
    print("=" * 78)
    print(f"  {'kernel':<11}{'tau':>6}{'xeps':>5}{'zm/zp':>10}"
          f"{'sTmap/sTana':>14}{'residual':>10}{'sem':>8}")
    resid = []
    for p in ver["points"]:
        j = int(np.argmin(np.abs(taus - p["tau"])))
        K = P[p["kernel"]]
        r_map = K["sigma_T"][j] / K["sigma_T_ana"][j]
        r_obs = p["z_mean"] / p["z_pred"]
        rel_sem = p["z_sem"] / p["z_mean"]
        print(f"  {p['kernel']:<11}{p['tau']:>6.0f}{p['frac']:>4.0f}"
              f"{r_obs:>10.4f}{r_map:>15.4f}{r_obs/r_map:>10.4f}"
              f"{rel_sem:>8.3f}")
        resid.append((r_obs / r_map, rel_sem))
    v = np.array([x[0] for x in resid])
    w = np.array([x[1] for x in resid])
    sem_mean = float(np.sqrt((w ** 2).sum()) / len(w))
    print(f"\n  mean residual after removing sigma_T: "
          f"{v.mean():.4f} +/- {sem_mean:.4f}  "
          f"({100*(v.mean()-1):+.1f}% +/- {100*sem_mean:.1f}%)")
    print(f"  -> {'CONSISTENT WITH 1' if abs(v.mean()-1) < 2*sem_mean else 'NOT consistent with 1'}")
    out["decomposition"] = dict(residual_mean=float(v.mean()),
                                residual_sem=sem_mean,
                                explained_by_sigma_T=True)
    json.dump(out, open(os.path.join(HERE, a.out + ".json"), "w"), indent=2)
    print(f"\nSaved: {a.out}.json")


if __name__ == "__main__":
    main()
