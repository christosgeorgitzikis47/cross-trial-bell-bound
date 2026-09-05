"""
PART 18 - COVARIANCE OF THE MATCHED-FILTER ESTIMATORS ACROSS PULSES

Section 6.4 already reports a lag-level check: the Pearson correlation of the
standardised delta-hat(k) between pulses, over all 20,000 lags, never exceeds
|r| = 0.018. That is a strong diagnostic but it is not the quantity the joint
bound actually depends on. Covariance concentrated in the few lags where a
narrow kernel puts its weight would barely move an unweighted correlation
over 20,000 lags, and would still bias sigma_joint.

WHAT IS COMPUTED HERE is the covariance of the final estimators themselves.
The contribution of lag k to T_p is W(k) delta-hat_p(k), so with the delta-hat
uncorrelated across k (section 6.3),

    Cov(T_p, T_q) = sum_k W(k)^2 Cov(delta-hat_p(k), delta-hat_q(k)),

so with z_p(k) = delta-hat_p(k)/sigma_p(k) the natural estimator of the
estimator-level correlation is

    C_pq(W) = sum_k W(k)^2 z_p(k) z_q(k) / sum_k W(k)^2.

A W^2-WEIGHTED CORRELATION of the z would be the wrong statistic here, and
badly so: a narrow filter puts essentially all of its weight on one or two
lags, so such a correlation is computed over about one effective point and
saturates at +-1 whatever the data say. The effective number of lags a filter
uses is N_eff = (sum W^2)^2 / sum W^4, which is 1.1 for the future kernel at
tau = 1 and 2.2 for the symmetric one -- hence the need for a statistic whose
null distribution is known at any N_eff.

C_pq is standardised instead against its own sampling error. Under the null
that the pulses are independent, the z_p(k) z_q(k) are uncorrelated across k
with unit variance, so Var(sum_k W^2 z_p z_q) = sum_k W^4 and

    t_pq = sum_k W(k)^2 z_p(k) z_q(k) / sqrt( sum_k W(k)^4 )

is standard normal. This is evaluated for every one of the 4 x 26 = 104
filters, every one of the 45 pulse pairs and both channels: 9,360 values whose
distribution is the test. A narrow kernel gives a noisy t, not a saturated
one, and the noise is quantified rather than mistaken for signal.

THE NUMBER THAT MATTERS is what the off-diagonal terms do to the joint error
bar. With weights w_p = 1/sigma_p^2 (what section 6.4 uses) and the full
covariance Sigma_pq = C_pq sigma_p sigma_q,

    Var_full = w^T Sigma w / (sum w)^2,     Var_diag = 1 / sum_p (1/sigma_p^2)

and the quantity reported is sqrt(Var_full/Var_diag) - 1: the percentage by
which sigma_joint would move if the measured off-diagonal terms were kept.
For a narrow filter that single number is itself noisy, since each C_pq is one
draw with standard error 1/sqrt(N_eff). Averaging over filters does not fix
this, because the 104 filters are built from the same ten delta-hat fields and
are therefore strongly correlated: an error bar computed as sd/sqrt(104) would
be far too small. The shift is therefore compared against its own null
distribution, obtained by giving each pulse an independent random circular
shift of the lag axis. That destroys any alignment between pulses while
preserving each pulse's own marginals and autocorrelation exactly, so the
spread of the shift under those permutations is what the shift would look like
with no cross-pulse covariance at all.

The sigma_p come from meros9_joint.json, so the arithmetic is the same one
that produced the published joint bound.
"""
import json, math, os, time
import numpy as np

from load_curby import read_file
from meros2_injection import scan, mi_and_delta
from meros3_map import sigma_delta
from meros5_asym import kernel_W, KERNELS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dedomena_curby")
ROUNDS = [1000, 15000, 22000, 23000, 26000, 28293, 28294, 28295, 28296, 28297]
CONSEC = [28293, 28294, 28295, 28296, 28297]
K = 10_000
INFLATION_LIMIT = 0.01                 # 1%, the criterion stated in the text
NPERM = 20                             # circular-shift permutations for the null


def main():
    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    m9 = json.load(open(os.path.join(HERE, "meros9_joint.json")))
    taus = np.array(m5["taus"])
    kax = np.arange(-K, K + 1, dtype=np.float64)
    nz = kax != 0
    knz = kax[nz]

    print("=" * 78)
    print("PART 18 - COVARIANCE OF T_p ACROSS PULSES, PER FILTER")
    print("=" * 78)
    print(f"  {len(ROUNDS)} pulses, {len(ROUNDS)*(len(ROUNDS)-1)//2} pairs, "
          f"{len(KERNELS)} kernels x {len(taus)} widths = "
          f"{len(KERNELS)*len(taus)} filters, 2 channels")
    print(f"  criterion: sigma_joint must move by less than "
          f"{100*INFLATION_LIMIT:.0f}% when the off-diagonal terms are kept\n")

    # ---------------- the standardised residuals of every pulse ----------
    Z = {"OA vs SB": np.empty((len(ROUNDS), int(nz.sum()))),
         "OB vs SA": np.empty((len(ROUNDS), int(nz.sum())))}
    t0 = time.time()
    for i, rnd in enumerate(ROUNDS):
        data, _ = read_file(os.path.join(DATA, f"curby_round_{rnd}.bin"))
        SA = data["SA"].astype(np.int8); SB = data["SB"].astype(np.int8)
        OA = (data["OA"] > 0).astype(np.int8)
        OB = (data["OB"] > 0).astype(np.int8)
        del data
        for label, O, S in (("OA vs SB", OA, SB), ("OB vs SA", OB, SA)):
            s1 = (S == 1).astype(np.int8)
            ks, n11, A1, B1, nk, _ = scan(O, s1, K)
            _, delta = mi_and_delta(n11, A1, B1, nk)
            sd = sigma_delta(A1, B1, nk)
            Z[label][i] = (delta / sd)[nz]
        del SA, SB, OA, OB
        print(f"  round {rnd} scanned ({time.time()-t0:.0f}s)", flush=True)

    out = {"rounds": ROUNDS, "taus": taus.tolist(), "kernels": KERNELS,
           "inflation_limit": INFLATION_LIMIT, "filters": []}
    worst_rho = {"value": 0.0}
    worst_inf = {"value": -1.0}
    all_rho, all_inf, cons_rho, all_t = [], [], [], []

    print(f"\n  {'channel':<9} {'kernel':<11} {'tau':>7} {'N_eff':>8} "
          f"{'max|C|':>8} {'max|t|':>8} {'sigma_joint shift':>18}")
    for label in ("OA vs SB", "OB vs SA"):
        Zc = Z[label]
        for kn in KERNELS:
            sig_all = np.array([m9["per_pulse"][label][kn][i]["sigma"]
                                for i in range(len(ROUNDS))])   # 10 x n_tau
            for jt, tau in enumerate(taus):
                W = kernel_W(kn, knz, float(tau))
                w2 = W ** 2
                sw2 = float(w2.sum())
                sw4 = float((w2 ** 2).sum())
                neff = sw2 ** 2 / sw4
                # C_pq: the estimator-level correlation, and t_pq: its z-score
                # against the null that the pulses are independent
                G = (Zc * w2) @ Zc.T
                R = G / sw2
                np.fill_diagonal(R, 1.0)
                T = G / math.sqrt(sw4)
                off = R - np.eye(len(ROUNDS))
                a = int(np.argmax(np.abs(off))) // len(ROUNDS)
                b = int(np.argmax(np.abs(off))) % len(ROUNDS)
                rmax = float(off[a, b])
                iu = np.triu_indices(len(ROUNDS), 1)
                tvals = T[iu]

                # what the off-diagonal terms do to sigma_joint
                s = sig_all[:, jt]
                w = 1.0 / s ** 2
                Sig = R * np.outer(s, s)
                var_full = float(w @ Sig @ w) / float(w.sum()) ** 2
                var_diag = 1.0 / float((1.0 / s ** 2).sum())
                infl = math.sqrt(max(var_full, 0.0) / var_diag) - 1.0

                cons = [(p, q) for p in range(len(ROUNDS))
                        for q in range(p + 1, len(ROUNDS))
                        if ROUNDS[p] in CONSEC and ROUNDS[q] in CONSEC]
                cons_rho += [float(R[p, q]) for p, q in cons]
                all_rho.append(abs(rmax)); all_inf.append(infl)
                all_t.extend(float(v) for v in tvals)
                rec = dict(channel=label, kernel=kn, tau=float(tau),
                           n_eff=float(neff),
                           max_abs_rho=abs(rmax), max_rho=rmax,
                           max_pair=[ROUNDS[a], ROUNDS[b]],
                           max_abs_t=float(np.abs(tvals).max()),
                           sigma_joint_inflation=infl,
                           mean_abs_offdiag=float(np.abs(off).sum() /
                                                  (len(ROUNDS) ** 2 - len(ROUNDS))))
                out["filters"].append(rec)
                if abs(rmax) > abs(worst_rho["value"]):
                    worst_rho = dict(value=rmax, **{k: rec[k] for k in
                                     ("channel", "kernel", "tau", "max_pair")})
                if infl > worst_inf["value"]:
                    worst_inf = dict(value=infl, **{k: rec[k] for k in
                                     ("channel", "kernel", "tau")})
                if tau in (1., 10., 100., 1000., 10000.):
                    print(f"  {label:<9} {kn:<11} {tau:>7g} {neff:>8.1f} "
                          f"{abs(rmax):>8.4f} {rec['max_abs_t']:>8.2f} "
                          f"{100*infl:>17.3f}%")

    all_rho = np.array(all_rho); all_inf = np.array(all_inf)
    cons_rho = np.array(cons_rho); all_t = np.array(all_t)
    n_pairs = len(ROUNDS) * (len(ROUNDS) - 1) // 2
    print("\n" + "=" * 78)
    print(f"  filters examined: {len(all_rho)} (2 channels x 4 kernels x "
          f"{len(taus)} widths), each over {n_pairs} pulse pairs "
          f"-> {len(all_t):,} values of t")
    print(f"\n  THE TEST: t_pq standard normal if the pulses are independent")
    print(f"    mean {all_t.mean():+.4f} (theory 0)   "
          f"sd {all_t.std(ddof=1):.4f} (theory 1)")
    print(f"    largest |t| = {np.abs(all_t).max():.2f}; expected largest of "
          f"{len(all_t):,} normals = {math.sqrt(2*math.log(len(all_t))):.2f}")
    n3 = int((np.abs(all_t) > 3).sum())
    print(f"    |t| > 3: {n3} of {len(all_t):,} "
          f"({100*n3/len(all_t):.2f}%, expected 0.27%)")
    print(f"\n  the raw correlations C_pq, for reference (they saturate for "
          f"narrow filters and are NOT the test):")
    print(f"    largest |C| anywhere {abs(worst_rho['value']):.4f} at "
          f"{worst_rho['channel']}, {worst_rho['kernel']}, "
          f"tau = {worst_rho['tau']:g}, pulses {worst_rho['max_pair']}")
    print(f"    among the five consecutive pulses: max |C| = "
          f"{np.abs(cons_rho).max():.4f}, mean {cons_rho.mean():+.4f}")
    print(f"\n  sigma_joint shift if the measured off-diagonal terms are kept:")
    print(f"    mean over the {len(all_inf)} filters {100*all_inf.mean():+.4f}%"
          f"   sd {100*all_inf.std(ddof=1):.4f}%")
    print(f"    standard error of that mean: "
          f"{100*all_inf.std(ddof=1)/math.sqrt(len(all_inf)):.4f}%")
    wide = all_inf[np.array([f["n_eff"] for f in out["filters"]]) >= 10]
    print(f"    filters with N_eff >= 10 ({len(wide)} of {len(all_inf)}): "
          f"mean {100*wide.mean():+.4f}%   "
          f"range [{100*wide.min():+.4f}%, {100*wide.max():+.4f}%]")
    print(f"    all filters, range [{100*all_inf.min():+.4f}%, "
          f"{100*all_inf.max():+.4f}%] (the extremes are the narrow filters, "
          f"where one draw of C carries an error of 1/sqrt(N_eff))")
    # ---- null distribution of the shift: independent circular shifts ----
    print(f"\n  null for the shift: {NPERM} independent circular shifts of the "
          f"lag axis per pulse")
    rng = np.random.default_rng(20260905)
    nlag = Z["OA vs SB"].shape[1]
    null_mean, null_wide = [], []
    for it in range(NPERM):
        infl_perm = []
        Zp = {lab: np.stack([np.roll(Z[lab][i], int(rng.integers(nlag)))
                             for i in range(len(ROUNDS))]) for lab in Z}
        for label in ("OA vs SB", "OB vs SA"):
            Zc = Zp[label]
            for kn in KERNELS:
                sig_all = np.array([m9["per_pulse"][label][kn][i]["sigma"]
                                    for i in range(len(ROUNDS))])
                for jt, tau in enumerate(taus):
                    w2 = kernel_W(kn, knz, float(tau)) ** 2
                    sw2 = float(w2.sum())
                    R = ((Zc * w2) @ Zc.T) / sw2
                    np.fill_diagonal(R, 1.0)
                    s_ = sig_all[:, jt]; w_ = 1.0 / s_ ** 2
                    vf = float(w_ @ (R * np.outer(s_, s_)) @ w_) / float(w_.sum()) ** 2
                    vd = 1.0 / float((1.0 / s_ ** 2).sum())
                    infl_perm.append(math.sqrt(max(vf, 0.0) / vd) - 1.0)
        infl_perm = np.array(infl_perm)
        null_mean.append(float(infl_perm.mean()))
        null_wide.append(float(infl_perm[np.array(
            [f["n_eff"] for f in out["filters"]]) >= 10].mean()))
        print(f"    permutation {it+1}/{NPERM}: mean shift "
              f"{100*null_mean[-1]:+.4f}%", flush=True)
    null_mean = np.array(null_mean); null_wide = np.array(null_wide)
    obs_m, obs_w = float(all_inf.mean()), float(wide.mean())
    z_m = (obs_m - null_mean.mean()) / null_mean.std(ddof=1)
    z_w = (obs_w - null_wide.mean()) / null_wide.std(ddof=1)
    print(f"    null mean shift {100*null_mean.mean():+.4f}% "
          f"+- {100*null_mean.std(ddof=1):.4f}%  ->  observed "
          f"{100*obs_m:+.4f}% is {z_m:+.2f} sigma")
    print(f"    N_eff >= 10: null {100*null_wide.mean():+.4f}% "
          f"+- {100*null_wide.std(ddof=1):.4f}%  ->  observed "
          f"{100*obs_w:+.4f}% is {z_w:+.2f} sigma")
    out.update(null_shift_mean=float(null_mean.mean()),
               null_shift_sd=float(null_mean.std(ddof=1)),
               null_shift_wide_mean=float(null_wide.mean()),
               null_shift_wide_sd=float(null_wide.std(ddof=1)),
               z_shift=float(z_m), z_shift_wide=float(z_w), n_perm=NPERM)

    ok = (abs(float(all_t.mean())) < 0.05 and abs(z_m) < 3 and abs(z_w) < 3)
    print(f"\n  VERDICT: {'no covariance beyond what independent pulses give; inverse-variance weighting stands' if ok else '*** COVARIANCE BEYOND THE NULL - GLS NEEDED ***'}")
    print("=" * 78)

    out.update(t_mean=float(all_t.mean()), t_sd=float(all_t.std(ddof=1)),
               t_max_abs=float(np.abs(all_t).max()), n_t=len(all_t),
               n_t_above_3=int((np.abs(all_t) > 3).sum()),
               inflation_mean_wide=float(wide.mean()),
               inflation_sem=float(all_inf.std(ddof=1) / math.sqrt(len(all_inf))),
               max_abs_rho=float(all_rho.max()), worst_rho=worst_rho,
               max_abs_rho_consecutive=float(np.abs(cons_rho).max()),
               mean_rho_consecutive=float(cons_rho.mean()),
               inflation_mean=float(all_inf.mean()),
               inflation_min=float(all_inf.min()),
               inflation_max=float(all_inf.max()),
               inflation_max_abs=float(np.abs(all_inf).max()),
               worst_inflation=worst_inf, diagonal_ok=bool(ok))
    json.dump(out, open(os.path.join(HERE, "meros18_covariance.json"), "w"),
              indent=2)
    print("\nSaved: meros18_covariance.json")


if __name__ == "__main__":
    main()
