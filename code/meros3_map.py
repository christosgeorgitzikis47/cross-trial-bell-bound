"""
PART 3 - EXCLUSION MAP IN THE (eps, tau) PLANE

THE STATISTIC (a matched filter on delta, not on I)
    The model predicts delta(k) = alpha*eps*W_tau(k), W_tau(k)=exp(-k^2/2tau^2).
    delta-hat(k) is LINEAR in the signal and its noise is symmetric about
    zero. I-hat(k) is quadratic and its noise is positive definite, so
    filtering I would accumulate bias. Hence:

        T(tau) = sum_k W_tau(k)*delta-hat(k)     (primary)
        U(tau) = sum_k W_tau(k)^2*I-hat(k)       (secondary, for comparison)

    Under the model:  E[T] = alpha*eps*sum_k W_tau(k)^2 = alpha*eps*Q(tau)
    Under H0:         Var[T] = sum_k W_tau(k)^2*sigma_delta(k)^2
                      (if the delta-hat(k) are uncorrelated -- VERIFIED,
                      not assumed)

    -> eps-hat(tau) = T/(alpha*Q),  sigma_eps(tau) = sigma_T/(alpha*Q)
    -> UPPER BOUND:  eps_excl(tau) = |eps-hat| + z_thr*sigma_eps

k = 0 is EXCLUDED from W (revision 2026-09-04): it is the setting of the
same trial -- no-signalling, not cross-trial -- and is bounded separately.

Q(tau) = sum_{0<|k|<=10000} W_tau(k)^2 is computed EXACTLY as a finite sum,
not from the asymptotic tau*sqrt(pi). Above tau ~ 3000 the window truncates
the tails and Q falls short of tau*sqrt(pi) -> a BEND in the curve. It is
not smoothed.

THRESHOLD: the same Bonferroni as report #5, m = 40,002 hypotheses,
p = 1.25e-6, z_thr = 4.85 (chi^2(1) -> 23.5). The value "proper to this
test", m = 2*n_tau, is also printed, but the conservative one is used so
that the map stays directly comparable with the previous bound.
"""
import argparse, json, math, os, sys, time
import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.stats import chi2

from meros2_injection import scan, mi_and_delta, build_F

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2.0)
K = 10_000
M_BONF = 40_002                      # the same as report #5


def sigma_delta(A1, B1, nk):
    """sigma of delta-hat(k)=(r2-r1)/2 from the binomial, per lag."""
    A1f = A1.astype(float); B1f = B1.astype(float); nkf = nk.astype(float)
    p = A1f / nkf
    n1 = B1f; n2 = nkf - B1f
    return 0.5 * np.sqrt(p * (1 - p) * (1.0 / n1 + 1.0 / n2))


def kernel_Q(taus, K):
    """Q(tau) = sum_{|k|<=K} exp(-k^2/tau^2)  (= sum W_tau^2, EXACTLY, with
    the truncation)."""
    k = np.arange(-K, K + 1, dtype=np.float64)
    k = k[k != 0]                       # k = 0 does not belong to the model
    Q = np.array([np.exp(-k ** 2 / t ** 2).sum() for t in taus])
    Qasym = np.array([t * math.sqrt(math.pi) for t in taus])
    return Q, Qasym


def filters(taus, K):
    """Matrix of weights W_tau(k) (n_tau x n_lag). For large tau it is dense
    but small (20 x 20001)."""
    k = np.arange(-K, K + 1, dtype=np.float64)
    return np.stack([np.where(k != 0, np.exp(-k ** 2 / (2 * t ** 2)), 0.0)
                     for t in taus])


def apply_T(Wmat, delta):
    return Wmat @ delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--n-tau", type=int, default=25)
    ap.add_argument("--seed", type=int, default=4711)
    a = ap.parse_args()

    cal = json.load(open(os.path.join(HERE, "meros1_alpha.json")))
    alpha_A = cal["OA vs SA"]["alpha"]
    alpha_B = cal["OB vs SB"]["alpha"]

    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    SA, SB, OA, OB = d['SA'], d['SB'], d['OA'], d['OB']
    n = len(SA)

    taus = np.unique(np.round(np.logspace(0, 4, a.n_tau)).astype(int)).astype(float)
    Q, Qasym = kernel_Q(taus, a.K)
    Wmat = filters(taus, a.K)

    p_thr = 0.05 / M_BONF
    z_thr = float(math.sqrt(chi2.ppf(1 - p_thr, 1)))
    m_own = 2 * len(taus)
    z_own = float(math.sqrt(chi2.ppf(1 - 0.05 / m_own, 1)))

    print("=" * 78)
    print("PART 3 - EXCLUSION MAP (eps, tau)")
    print("=" * 78)
    print(f"  n = {n:,}   K = +/-{a.K:,}   tau: {len(taus)} values, "
          f"{taus[0]:g}...{taus[-1]:g}")
    print(f"  alpha (Alice) = {alpha_A:.6e}   alpha (Bob) = {alpha_B:.6e}")
    print(f"  Bonferroni threshold m = {M_BONF:,} (same as report #5): "
          f"p = {p_thr:.3e}  ->  z = {z_thr:.3f}")
    print(f"  (the m proper to this test, {m_own}, would give z = {z_own:.3f}; "
          f"the conservative one is used)")
    print(f"  tail truncation: Q(tau)/tau*sqrt(pi) at tau=10,000 -> "
          f"{Q[-1]/Qasym[-1]:.3f}\n")

    out = {"n": n, "K": a.K, "taus": taus.tolist(), "Q": Q.tolist(),
           "Q_asymptotic": Qasym.tolist(), "z_thr": z_thr, "m_bonf": M_BONF,
           "z_own": z_own, "m_own": m_own, "alpha_A": alpha_A,
           "alpha_B": alpha_B, "shuffles": a.shuffles, "pairs": {}}

    rng = np.random.default_rng(a.seed)

    for label, o, s, alpha in [("OA vs SB", OA, SB, alpha_A),
                               ("OB vs SA", OB, SA, alpha_B)]:
        print("=" * 78)
        print(label)
        print("=" * 78)
        s1 = (s == 1).astype(np.int8)
        t0 = time.time()
        ks, n11, A1, B1, nk, ferr = scan(o, s1, a.K)
        MI, delta = mi_and_delta(n11, A1, B1, nk)
        sd = sigma_delta(A1, B1, nk)
        print(f"  scanned {len(ks):,} lags in {time.time()-t0:.1f}s   "
              f"FFT error {ferr:.1e}")
        print(f"  delta-hat: mean {delta.mean():+.3e}  "
              f"sd {delta.std(ddof=1):.3e}  "
              f"(analytic sigma_delta {sd.mean():.3e})")

        # --- are the delta-hat(k) uncorrelated? (free check: autocorrelation in k)
        x = (delta - delta.mean()) / delta.std(ddof=1)
        ac = np.array([float(np.mean(x[:-l] * x[l:])) for l in (1, 2, 3, 5, 10)])
        se_ac = 1.0 / math.sqrt(len(x))
        print(f"  autocorrelation of delta-hat(k) at dk=1,2,3,5,10: "
              f"{np.array2string(ac, precision=4)}  (+/-{se_ac:.4f})")

        # --- empirical null of T(tau): we shuffle the SETTINGS ---
        print(f"  empirical null of T(tau) from {a.shuffles} shuffles...",
              flush=True)
        sh = s1.copy()
        Tnull = np.empty((a.shuffles, len(taus)))
        t0 = time.time()
        for i in range(a.shuffles):
            rng.shuffle(sh)
            _, n11s, A1s, B1s, nks, _ = scan(o, sh, a.K)
            _, ds = mi_and_delta(n11s, A1s, B1s, nks)
            Tnull[i] = apply_T(Wmat, ds)
            if i == 0:
                print(f"    (~{(time.time()-t0)*a.shuffles/60:.1f} minutes in "
                      f"total)", flush=True)
        sT_emp = Tnull.std(axis=0, ddof=1)
        sT_ana = np.sqrt(Wmat ** 2 @ sd ** 2)
        ratio_sT = sT_emp / sT_ana
        print(f"    sigma_T empirical/analytic: mean {ratio_sT.mean():.4f}  "
              f"range [{ratio_sT.min():.4f}, {ratio_sT.max():.4f}]")
        print(f"    mean T_null/sigma_T: "
              f"{(Tnull.mean(axis=0)/sT_emp).mean():+.3f}")
        # we use the LARGER of the two (conservative)
        sT = np.maximum(sT_emp, sT_ana)

        # --- the real T and the bound ---
        T = apply_T(Wmat, delta)
        z = T / sT
        eps_hat = T / (alpha * Q)
        sig_eps = sT / (alpha * Q)
        eps_excl = np.abs(eps_hat) + z_thr * sig_eps

        # --- secondary: the W^2 filter on I ---
        baseline = 1.0 / (2 * n * LN2)
        U = (Wmat ** 2) @ (MI - baseline)
        Unull = None
        # analytic sigma of U: Var(I-hat) ~ 2/(2n ln2)^2 per lag (chi^2(1))
        sI = math.sqrt(2.0) / (2 * n * LN2)
        sU = np.sqrt(((Wmat ** 2) ** 2).sum(axis=1)) * sI
        zU = U / sU
        # convert to eps: E[U] = C eps^2 sum W^4 -> eps_U = sqrt(U_lim/(C*sumW4))
        C = cal["OA vs SA"]["C"] if label == "OA vs SB" else cal["OB vs SB"]["C"]
        SW4 = ((Wmat ** 2) ** 2).sum(axis=1)
        U_lim = np.maximum(U, 0) + z_thr * sU
        eps_U = np.sqrt(U_lim / (C * SW4))

        print(f"\n  {'tau':>7} {'Q(tau)':>10} {'T':>11} {'z':>7} "
              f"{'sig_eps':>10} {'eps_ex(d)':>10} {'eps_ex(I)':>10}")
        for j, t in enumerate(taus):
            print(f"  {t:>7g} {Q[j]:>10.1f} {T[j]:>11.3e} {z[j]:>7.2f} "
                  f"{sig_eps[j]:>10.3e} {eps_excl[j]:>10.3e} {eps_U[j]:>10.3e}")
        nab = int((np.abs(z) > z_thr).sum())
        print(f"\n  |z| > {z_thr:.2f} at {nab} / {len(taus)} values of tau")
        print(f"  largest |z| = {np.abs(z).max():.2f} at tau = "
              f"{taus[int(np.argmax(np.abs(z)))]:g}")
        print(f"  eps_excl range: {eps_excl.min():.3e} ... {eps_excl.max():.3e}")
        print(f"  tau with eps_excl >= 1 (not constraining): "
              f"{int((eps_excl >= 1).sum())} / {len(taus)}\n")

        out["pairs"][label] = dict(
            T=T.tolist(), z=z.tolist(), sigma_T=sT.tolist(),
            sigma_T_emp=sT_emp.tolist(), sigma_T_ana=sT_ana.tolist(),
            eps_hat=eps_hat.tolist(), sigma_eps=sig_eps.tolist(),
            eps_excl=eps_excl.tolist(), eps_excl_I=eps_U.tolist(),
            zU=zU.tolist(), autocorr=ac.tolist(), autocorr_se=se_ac,
            n_above=nab, max_abs_z=float(np.abs(z).max()))
        np.savez_compressed(os.path.join(
            HERE, f"meros3_delta_{label.replace(' ', '_')}.npz"),
            lags=ks, delta=delta, sigma=sd, mi=MI)

    json.dump(out, open(os.path.join(HERE, "meros3_map.json"), "w"), indent=2)
    print("Saved: meros3_map.json + meros3_delta_*.npz")


if __name__ == "__main__":
    main()
