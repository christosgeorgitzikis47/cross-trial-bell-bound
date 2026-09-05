"""
PART 2 - VERIFYING THE ANALYTIC RELATION BY INJECTION

We do not accept I(k) = C*eps^2*exp(-k^2/tau^2). We test it.

WHAT IS INJECTED
    S(i)  = +1 if SB(i)=2,  -1 if SB(i)=1      (Bob's real settings)
    F(i)  = sum_k W_tau(k)*S(i+k),  W_tau(k)=exp(-k^2/2tau^2), k != 0
    lam(i)= lam0(SA(i)) + alpha*eps*F(i)
    OA*   ~ Bernoulli(lam)

    lam0(SA) = the REAL click rate of round 28297 per Alice setting
             (0.004967 for SA=1, 0.008891 for SA=2).
    -> Alice's genuine lag-0 dependence is kept as the background rather
       than removed. If it contaminated the measurement of the cross
       channel, it would show up here.
    -> The SETTINGS (SA, SB) stay REAL. They are not generated
       synthetically, otherwise the verification would be empty: we would
       be testing our own generator, not the data.

WHAT IS MEASURED, FOR EVERY lag k in [-K, +K] with one pair of FFTs
    delta_meas(k) = [rate(OA*|SB(i+k)=2) - rate(OA*|SB(i+k)=1)] / 2
    I_meas(k) = MI of the 2x2 table, minus the null baseline df/(2n ln2)

WHAT IS PREDICTED
    delta_pred(k) = alpha*eps*exp(-k^2/2tau^2)   -> Gaussian of width tau
    I_pred(k) = C*eps^2*exp(-k^2/tau^2)          -> Gaussian of width tau/sqrt(2)
Both widths are checked by a Gaussian fit, separately.

CLIPPING (REPORTED AS A MATTER OF COURSE)
    Clipping lam to [0,1] is a silent non-linearity: it gives the right
    SHAPE but the wrong AMPLITUDE, and looks like a failure of the
    derivation. The fraction of clipped trials is reported at every
    (eps,tau). Above 0.1% the point is marked INVALID. The eps values are
    chosen automatically to stay below that limit.

THE eps = 0 CONTROL
    First point of every tau. It MUST give a flat curve at the null.
    If it gives a bell, the bug is in the injection and we stop.
"""
import argparse, json, math, os, sys
import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2.0)
CLIP_LIMIT = 1e-3          # 0.1% -> above this the point is invalid


# ------------------------------------------------------------- the field F
def build_F(S, tau):
    """F(i) = sum_k exp(-k^2/2tau^2) * S(i+k), by FFT convolution (linear,
    no wraparound). The sum is truncated at +/-5tau (W < 3.7e-6)."""
    half = max(1, int(math.ceil(5 * tau)))
    k = np.arange(-half, half + 1, dtype=np.float64)
    W = np.exp(-k ** 2 / (2.0 * tau ** 2))
    W[half] = 0.0                       # k = 0 is outside the model (2026-09-04)
    n = len(S)
    L = next_fast_len(n + 2 * half + 1)
    Sf = rfft(S.astype(np.float64), L)
    # F(i) = sum_j W(j) S(i+j) = a correlation -> conjugate of the kernel
    Wp = np.zeros(L)
    Wp[:len(W)] = W
    Ff = irfft(Sf * np.conjugate(rfft(Wp, L)), L)
    # Wp[m] = W(m-half) -> corr[t] = sum_m Wp[m]*S[t+m] = F(t+half)
    # so F(i) = corr[i-half]  ->  shift by +half
    F = np.roll(Ff, half)[:n]
    return F, float(W.sum()), float((W ** 2).sum()), half


def check_F(S, F, tau, half, rng, n_check=5):
    """Direct verification of F at a few positions."""
    n = len(S)
    idx = rng.integers(half + 1, n - half - 1, n_check)
    k = np.arange(-half, half + 1)
    W = np.exp(-k ** 2 / (2.0 * tau ** 2))
    W[half] = 0.0
    worst = 0.0
    for i in idx:
        direct = float(np.dot(W, S[i - half:i + half + 1]))
        worst = max(worst, abs(direct - F[i]))
    return worst


# ------------------------------------------------- measurements at all lags
def scan(o, s1, K):
    """N11(k)=sum o[i]*s1[i+k] for k in [-K,K] with one pair of FFTs (the
    same logic as full_scan.py), together with the margins."""
    n = len(o)
    L = next_fast_len(n + K + 1)
    R = irfft(np.conjugate(rfft(o.astype(np.float64), L)) *
              rfft(s1.astype(np.float64), L), L)
    c = np.concatenate([R[L - K:], R[:K + 1]])
    ferr = float(np.abs(c - np.rint(c)).max())
    n11 = np.rint(c).astype(np.int64)

    ks = np.arange(-K, K + 1)
    co = np.concatenate([[0], np.cumsum(o, dtype=np.int64)])
    cs = np.concatenate([[0], np.cumsum(s1, dtype=np.int64)])
    tot_o, tot_s = int(co[-1]), int(cs[-1])
    A1 = np.empty(len(ks), np.int64); B1 = np.empty(len(ks), np.int64)
    for j, k in enumerate(ks):
        if k >= 0:
            A1[j] = co[n - k];            B1[j] = tot_s - cs[k]
        else:
            m = -k
            A1[j] = tot_o - co[m];        B1[j] = cs[n - m]
    nk = n - np.abs(ks)
    return ks, n11, A1, B1, nk, ferr


def mi_and_delta(n11, A1, B1, nk):
    """MI (bits) and delta(k) = [rate(o|s=2) - rate(o|s=1)]/2 from the 2x2
    table. s1 is the indicator of "setting == 1", so n11 = #(o=1 & s=1)."""
    n11f = n11.astype(np.float64); A1f = A1.astype(np.float64)
    B1f = B1.astype(np.float64);   nkf = nk.astype(np.float64)
    cells = np.stack([n11f, A1f - n11f, B1f - n11f, nkf - A1f - B1f + n11f])
    rows = np.stack([A1f, A1f, nkf - A1f, nkf - A1f])
    cols = np.stack([B1f, nkf - B1f, B1f, nkf - B1f])
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (cells / nkf) * np.log2(cells * nkf / (rows * cols))
    MI = np.nansum(t, axis=0)
    r1 = n11f / B1f                        # rate(o=1 | s=1)
    r2 = (A1f - n11f) / (nkf - B1f)        # rate(o=1 | s=2)
    delta = (r2 - r1) / 2.0
    return MI, delta


def gauss_fit(ks, y, sigma0, amp0):
    """y = A*exp(-k^2/2sigma^2), no offset (the baseline is already removed)."""
    def f(k, A, s):
        return A * np.exp(-k ** 2 / (2.0 * s ** 2))
    try:
        p, cov = curve_fit(f, ks.astype(float), y, p0=[amp0, sigma0],
                           maxfev=20000)
        err = np.sqrt(np.diag(cov))
        return float(p[0]), float(p[1]), float(err[0]), float(err[1])
    except Exception as e:
        return math.nan, math.nan, math.nan, math.nan


# ------------------------------------------------------------------- main
def run_point(eps, tau, F, lam0, SB1, alpha, C, K, ks_pred, rng, n):
    lam = lam0 + alpha * eps * F
    n_lo = int(np.count_nonzero(lam < 0.0))
    n_hi = int(np.count_nonzero(lam > 1.0))
    frac_clip = (n_lo + n_hi) / n
    lam_c = np.clip(lam, 0.0, 1.0)

    O = (rng.random(n) < lam_c).astype(np.int8)

    ks, n11, A1, B1, nk, ferr = scan(O, SB1, K)
    MI, delta = mi_and_delta(n11, A1, B1, nk)
    baseline = 1.0 / (2.0 * n * LN2)          # E[MI] under H0, df=1
    MIc = MI - baseline

    # predictions
    nz = ks != 0                                   # k = 0 outside the model
    d_pred = alpha * eps * np.exp(-ks.astype(float) ** 2 / (2 * tau ** 2)) * nz
    I_pred = C * eps ** 2 * np.exp(-ks.astype(float) ** 2 / tau ** 2) * nz

    res = dict(eps=eps, tau=tau, n_clip_low=n_lo, n_clip_high=n_hi,
               frac_clipped=frac_clip, valid=bool(frac_clip <= CLIP_LIMIT),
               fft_round_error=ferr, click_rate=float(O.mean()),
               baseline_mi=baseline,
               # peak of the kernel: k = +1 (k = 0 has W = 0)
               delta_meas_1=float(delta[K + 1]), delta_pred_1=float(d_pred[K + 1]),
               mi_meas_1=float(MIc[K + 1]), mi_pred_1=float(I_pred[K + 1]),
               delta_meas_0=float(delta[K]), mi_meas_0=float(MIc[K]))

    if eps == 0.0:
        # flatness check: no bell shape is allowed
        sd_null = math.sqrt(2.0) / (2.0 * n * LN2)
        res["null_mi_mean"] = float(MIc.mean())
        res["null_mi_sd"] = float(MIc.std(ddof=1))
        res["null_mi_sd_theory"] = sd_null
        res["null_mi_max"] = float(MIc.max())
        res["null_delta_sd"] = float(delta.std(ddof=1))
        # amplitude of a bell that SHOULD NOT be found:
        A, s, dA, ds = gauss_fit(ks[nz], MIc[nz], max(tau, 1.0), MIc.max())
        res["null_fit_amp"] = A
        res["null_fit_sigma"] = s
        # centre compared with the tails
        core = np.abs(ks) <= max(1, int(tau))
        res["null_core_mean"] = float(MIc[core].mean())
        res["null_tail_mean"] = float(MIc[~core].mean())
        res["null_core_n"] = int(core.sum())
        res["null_tail_n"] = int((~core).sum())
        # error on the centre-minus-tail difference (the MI estimators at
        # different lags are practically independent -- random settings)
        res["null_diff_se"] = float(res["null_mi_sd"] * math.sqrt(
            1.0 / core.sum() + 1.0 / max(1, (~core).sum())))
        return res, ks, MIc, delta, I_pred, d_pred

    # ---- Gaussian fit to delta(k): expected sigma = tau
    # (over k != 0 only: k = 0 does not belong to the model)
    Ad, sd_, eAd, esd = gauss_fit(ks[nz], delta[nz], tau, alpha * eps)
    # ---- and to I(k): expected sigma = tau/sqrt(2)
    Ai, si_, eAi, esi = gauss_fit(ks[nz], MIc[nz], tau / math.sqrt(2),
                                  C * eps ** 2)
    res.update(
        fit_delta_amp=Ad, fit_delta_sigma=sd_,
        fit_delta_amp_err=eAd, fit_delta_sigma_err=esd,
        fit_delta_sigma_expected=float(tau),
        fit_mi_amp=Ai, fit_mi_sigma=si_,
        fit_mi_amp_err=eAi, fit_mi_sigma_err=esi,
        fit_mi_sigma_expected=float(tau / math.sqrt(2)),
        ratio_delta_amp=Ad / (alpha * eps) if eps else math.nan,
        ratio_mi_amp=Ai / (C * eps ** 2) if eps else math.nan,
        ratio_delta_sigma=sd_ / tau,
        ratio_mi_sigma=si_ / (tau / math.sqrt(2)),
        ratio_mi1=res["mi_meas_1"] / res["mi_pred_1"] if res["mi_pred_1"] else math.nan,
        ratio_delta1=res["delta_meas_1"] / res["delta_pred_1"] if res["delta_pred_1"] else math.nan,
    )
    return res, ks, MIc, delta, I_pred, d_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[1.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--out", default="meros2_injection")
    a = ap.parse_args()

    cal = json.load(open(os.path.join(HERE, "meros1_alpha.json")))["OA vs SA"]
    alpha = cal["alpha"]
    C = cal["C"]
    r_by_setting = {1: cal["r1"], 2: cal["r2"]}

    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    SA, SB = d['SA'], d['SB']
    n = len(SA)

    S = np.where(SB == 2, 1.0, -1.0)          # +/-1 encoding of Bob
    SB1 = (SB == 1).astype(np.int8)
    lam0 = np.where(SA == 1, r_by_setting[1], r_by_setting[2])

    rng = np.random.default_rng(a.seed)

    print("=" * 78)
    print("PART 2 - INJECTION WITH KNOWN (eps, tau)")
    print("=" * 78)
    print(f"  n = {n:,}   alpha = {alpha:.6e}   C = {C:.6e}")
    print(f"  lam0: SA=1 -> {r_by_setting[1]:.6e}, SA=2 -> {r_by_setting[2]:.6e}")
    print(f"  clipping limit: {CLIP_LIMIT*100:.1f}% of trials\n")

    all_res = {"alpha": alpha, "C": C, "n": n, "seed": a.seed,
               "clip_limit": CLIP_LIMIT, "points": []}
    curves = {}

    for tau in a.taus:
        print("=" * 78)
        print(f"tau = {tau:g}")
        print("=" * 78)
        F, Wsum, W2sum, half = build_F(S, tau)
        werr = check_F(S, F, tau, half, rng)
        sdF = float(F.std())
        print(f"  kernel: sum W = {Wsum:.4f}  sum W^2 = {W2sum:.4f}  "
              f"(theory sum W^2 approx tau*sqrt(pi) = "
              f"{tau*math.sqrt(math.pi):.4f})")
        print(f"  sd(F) = {sdF:.4f}  (theory sqrt(sum W^2) = "
              f"{math.sqrt(W2sum):.4f})")
        print(f"  F verified at 5 positions: largest difference {werr:.2e}")
        if werr > 1e-6:
            sys.exit("The FFT convolution for F disagreed with the direct sum.")

        # --- choose eps so that clipping stays below 0.1% ---
        # the lower branch lam0 = r1 is the binding one
        q = np.quantile(F, CLIP_LIMIT / 2)     # negative end
        eps_max = r_by_setting[1] / (alpha * abs(q))
        eps_list = [0.0, eps_max / 4, eps_max / 2, eps_max]
        print(f"  eps_max (from clipping) = {eps_max:.4g}   "
              f"trials: {['%.4g' % e for e in eps_list[1:]]}")

        K = max(20, int(math.ceil(6 * tau)))
        ks_pred = None
        for eps in eps_list:
            res, ks, MIc, delta, I_pred, d_pred = run_point(
                eps, tau, F, lam0, SB1, alpha, C, K, ks_pred, rng, n)
            all_res["points"].append(res)
            key = f"tau{tau:g}_eps{eps:.6g}"
            curves[key] = dict(lags=ks.tolist(),
                               mi=MIc.tolist(), delta=delta.tolist(),
                               mi_pred=I_pred.tolist(), delta_pred=d_pred.tolist())

            if eps == 0.0:
                print(f"\n  --- eps = 0 CONTROL (must be FLAT at the null) ---")
                print(f"    clipped: {res['frac_clipped']*100:.4f}%")
                print(f"    MI mean {res['null_mi_mean']:+.3e}  "
                      f"sd {res['null_mi_sd']:.3e}  "
                      f"(theory sd {res['null_mi_sd_theory']:.3e})")
                dif = res['null_core_mean'] - res['null_tail_mean']
                print(f"    MI core(|k|<=tau, {res['null_core_n']}) "
                      f"{res['null_core_mean']:+.3e}  vs  tails "
                      f"({res['null_tail_n']}) {res['null_tail_mean']:+.3e}")
                print(f"    difference {dif:+.3e} +/- {res['null_diff_se']:.3e}  "
                      f"-> {dif/res['null_diff_se']:+.2f} sigma")
                print(f"    bell fit: A = {res['null_fit_amp']:.3e} "
                      f"(should be ~0, against sd {res['null_mi_sd']:.1e})")
                flat = abs(dif) < 3 * res['null_diff_se']
                print(f"    FLAT? {'YES' if flat else 'NO - BUG IN THE INJECTION'}")
                res["flat"] = bool(flat)
                if not flat:
                    sys.exit("The eps=0 control failed: the injection produces "
                             "signal out of nothing. Stopping before we "
                             "interpret anything.")
                continue

            mark = "" if res["valid"] else "   *** INVALID (clipping) ***"
            print(f"\n  --- eps = {eps:.4g} ---{mark}")
            print(f"    clipped: {res['frac_clipped']*100:.4f}% "
                  f"({res['n_clip_low']:,} low, {res['n_clip_high']:,} high)"
                  f"   [limit {CLIP_LIMIT*100:.1f}%]")
            print(f"    delta(+1): meas {res['delta_meas_1']:.4e}  "
                  f"pred {res['delta_pred_1']:.4e}  "
                  f"ratio {res['ratio_delta1']:.3f}   "
                  f"[delta(0) = {res['delta_meas_0']:+.2e}, outside the model]")
            print(f"    I(+1): meas {res['mi_meas_1']:.4e}  "
                  f"pred {res['mi_pred_1']:.4e}  "
                  f"ratio {res['ratio_mi1']:.3f}")
            print(f"    fit delta(k): A = {res['fit_delta_amp']:.4e} "
                  f"(ratio {res['ratio_delta_amp']:.3f})   "
                  f"sigma = {res['fit_delta_sigma']:.3f} +/- "
                  f"{res['fit_delta_sigma_err']:.3f}"
                  f"   expected tau = {tau:g}  -> ratio "
                  f"{res['ratio_delta_sigma']:.3f}")
            print(f"    fit I(k): A = {res['fit_mi_amp']:.4e} "
                  f"(ratio {res['ratio_mi_amp']:.3f})   "
                  f"sigma = {res['fit_mi_sigma']:.3f} +/- "
                  f"{res['fit_mi_sigma_err']:.3f}"
                  f"   expected tau/sqrt(2) = {tau/math.sqrt(2):.3f}  -> "
                  f"ratio {res['ratio_mi_sigma']:.3f}")
        print()
        del F

    json.dump(all_res, open(os.path.join(HERE, a.out + ".json"), "w"), indent=2)
    np.savez_compressed(os.path.join(HERE, a.out + "_curves.npz"),
                        **{k: np.array(v["mi"]) for k, v in curves.items()},
                        **{k + "_delta": np.array(v["delta"]) for k, v in curves.items()},
                        **{k + "_lags": np.array(v["lags"]) for k, v in curves.items()})

    # ---------------------------------------------------------- summary
    print("=" * 78)
    print("SUMMARY - VALID POINTS ONLY (clipping <= 0.1%)")
    print("=" * 78)
    print(f"{'tau':>7} {'eps':>10} {'clip%':>7} {'I(+1) ratio':>11} "
          f"{'s_d/tau':>8} {'s_I/(t/r2)':>11} {'valid':>7}")
    for r in all_res["points"]:
        if r["eps"] == 0.0:
            continue
        print(f"{r['tau']:>7g} {r['eps']:>10.4g} {r['frac_clipped']*100:>7.4f} "
              f"{r['ratio_mi1']:>11.3f} {r['ratio_delta_sigma']:>8.3f} "
              f"{r['ratio_mi_sigma']:>11.3f} {'yes' if r['valid'] else 'NO':>7}")
    print(f"\nSaved: {a.out}.json + {a.out}_curves.npz")


if __name__ == "__main__":
    main()
