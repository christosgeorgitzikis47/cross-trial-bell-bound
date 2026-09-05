"""
PART 8 - CORRELATION OF THE SETTINGS: THE PHANTOM SIGNAL

The objection (referee #2): O_A(i) depends STRONGLY on S_A(i) -- that is plain
quantum mechanics, delta_A = 1.96e-3. If S_B(i+k) were correlated with S_A(i),
the cross channel would show a signal WITHOUT any new physics.

THE ALGEBRA (why the formula is what it is)
    With S encoded as +/-1 and equiprobable settings, Alice's click rate is
    p(i) = p0 + delta_A*S_A(i). Hence

      E[O_A | S_B(i+k)=s] = p0 + delta_A*E[S_A(i) | S_B(i+k)=s]
                          = p0 + delta_A*rho(k)*s

    (for symmetric binaries, E[S_A|S_B=s] = rho*s). The measured delta-hat is
    half the difference between s = +/-1, that is

      delta_phantom(k) = delta_A * rho(k),  rho(k) = Corr(S_A(i), S_B(i+k))

    Passed through the SAME matched filter, it gives a phantom eps:

      eps_phantom(tau) = sum_k W_tau(k)*delta_phantom(k) / (alpha Q)
                       = sum_k W_tau(k)*rho(k) / Q(tau)

    so alpha cancels: the phantom eps is just the weighted mean of the setting
    correlation. A clean quantity to compare with eps_excl.

WHAT IS MEASURED
    rho_AB(k) = Corr(S_A(i), S_B(i+k))  [the critical one: across the wings]
    rho_AA(k), rho_BB(k)                [the autocorrelations]
    I(S_A(i) ; S_B(i+k)) with the SAME chi^2(1)/Bonferroni threshold of 6.1
    eps_phantom(tau) for the 4 kernels, next to eps_excl

IF anything crosses the threshold WE SAY SO: it would be a finding about the
experiment itself (the setting generators not being independent), not about
our model.
"""
import json, math, os
import numpy as np
from scipy.fft import next_fast_len, rfft, irfft

from meros2_injection import scan, mi_and_delta
from meros5_asym import kernel_W, KERNELS

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2.0)
K = 10_000


def xcorr(x, y, K):
    """rho(k) = Corr(x(i), y(i+k)) for |k| <= K, with one pair of FFTs.
    Returns the Pearson coefficient per lag (each lag has n-|k| pairs)."""
    n = len(x)
    xf = x.astype(np.float64); yf = y.astype(np.float64)
    L = next_fast_len(n + K + 1)
    R = irfft(np.conjugate(rfft(xf, L)) * rfft(yf, L), L)
    s_xy = np.concatenate([R[L - K:], R[:K + 1]])          # sum x(i)*y(i+k)

    ks = np.arange(-K, K + 1)
    cx = np.concatenate([[0.0], np.cumsum(xf)])
    cy = np.concatenate([[0.0], np.cumsum(yf)])
    cx2 = np.concatenate([[0.0], np.cumsum(xf ** 2)])
    cy2 = np.concatenate([[0.0], np.cumsum(yf ** 2)])
    sx = np.empty(len(ks)); sy = np.empty(len(ks))
    sx2 = np.empty(len(ks)); sy2 = np.empty(len(ks))
    for j, k in enumerate(ks):
        if k >= 0:                       # i from 0 to n-k-1, i+k from k to n-1
            sx[j] = cx[n - k];           sx2[j] = cx2[n - k]
            sy[j] = cy[n] - cy[k];       sy2[j] = cy2[n] - cy2[k]
        else:
            m = -k
            sx[j] = cx[n] - cx[m];       sx2[j] = cx2[n] - cx2[m]
            sy[j] = cy[n - m];           sy2[j] = cy2[n - m]
    nk = (n - np.abs(ks)).astype(np.float64)
    cov = s_xy / nk - (sx / nk) * (sy / nk)
    vx = sx2 / nk - (sx / nk) ** 2
    vy = sy2 / nk - (sy / nk) ** 2
    return ks, cov / np.sqrt(vx * vy)


def report(name, ks, rho, se):
    j = int(np.argmax(np.abs(rho)))
    n_above = int((np.abs(rho) > 4.848 * se).sum())
    print(f"  {name:<26} max |rho| = {abs(rho[j]):.3e} at lag {ks[j]:>+7d}"
          f"   ({abs(rho[j])/se:>5.2f} sigma)   |rho| > 4.848 sigma at "
          f"{n_above} lags")
    return dict(max_abs_rho=float(abs(rho[j])), at_lag=int(ks[j]),
                sigma=float(abs(rho[j]) / se), n_above=n_above)


def main():
    cal = json.load(open(os.path.join(HERE, "meros1_alpha.json")))
    aA = cal["OA vs SA"]["alpha"]; aB = cal["OB vs SB"]["alpha"]
    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    z_thr = m5["z_thr"]
    taus = np.array(m5["taus"])

    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    SA, SB = d['SA'], d['SB']
    n = len(SA)
    sa = np.where(SA == 2, 1.0, -1.0)          # +/-1
    sb = np.where(SB == 2, 1.0, -1.0)
    se = 1.0 / math.sqrt(n)

    print("=" * 78)
    print("PART 8 - SETTING CORRELATION AND THE PHANTOM SIGNAL")
    print("=" * 78)
    print(f"  n = {n:,}   |k| <= {K:,}   standard error of rho: "
          f"1/sqrt(n) = {se:.3e}")
    print(f"  threshold 4.848 sigma  ->  |rho| > {4.848*se:.3e}\n")

    out = {"n": n, "se_rho": se, "z_thr": z_thr}
    print("  CORRELATIONS (Pearson, +/-1 encoding):")
    ks, r_ab = xcorr(sa, sb, K)
    out["rho_AB"] = report("Corr(S_A(i), S_B(i+k))", ks, r_ab, se)
    _, r_aa = xcorr(sa, sa, K)
    r_aa0 = r_aa.copy(); r_aa0[ks == 0] = 0.0          # k=0 is 1 by definition
    out["rho_AA"] = report("Corr(S_A(i), S_A(i+k)), k!=0", ks, r_aa0, se)
    _, r_bb = xcorr(sb, sb, K)
    r_bb0 = r_bb.copy(); r_bb0[ks == 0] = 0.0
    out["rho_BB"] = report("Corr(S_B(i), S_B(i+k)), k!=0", ks, r_bb0, se)

    # ---------- mutual information of the settings ----------
    print("\n  MUTUAL INFORMATION I(S_A(i) ; S_B(i+k)):")
    a1 = (SA == 1).astype(np.int8); b1 = (SB == 1).astype(np.int8)
    ks2, n11, A1, B1, nk, ferr = scan(a1, b1, K)
    MI, _ = mi_and_delta(n11, A1, B1, nk)
    G = 2 * nk * LN2 * MI
    mi_thr = z_thr ** 2 / (2 * n * LN2)
    j = int(np.argmax(MI))
    n_above = int((G > z_thr ** 2).sum())
    print(f"    threshold (same Bonferroni, m = 40,002): "
          f"MI = {mi_thr:.4e} bits/trial  (G = {z_thr**2:.2f})")
    print(f"    largest MI = {MI[j]:.4e} at lag {ks2[j]:+d}   "
          f"G = {G[j]:.2f}  (sqrt(G) = {math.sqrt(G[j]):.2f} sigma)")
    print(f"    lags above the threshold: {n_above} / {len(ks2):,}")
    out["mi_settings"] = dict(threshold=float(mi_thr), max_mi=float(MI[j]),
                              at_lag=int(ks2[j]), max_G=float(G[j]),
                              n_above=n_above, fft_err=float(ferr))

    # ---------- the phantom signal ----------
    print("\n" + "=" * 78)
    print("PHANTOM SIGNAL:  delta_phantom(k) = delta_A * rho(k)")
    print("=" * 78)
    dph = aA * np.abs(r_ab)
    jj = int(np.argmax(dph))
    m3 = np.load(os.path.join(HERE, "meros5_delta_OA_vs_SB.npz"))
    sd_obs = float(m3["sigma"].mean())
    dhat_max = float(np.abs(m3["delta"]).max())
    print(f"  largest |delta_phantom| = {dph[jj]:.3e} at lag {ks[jj]:+d}")
    print(f"  standard error of delta-hat per lag (mean sigma_delta) = "
          f"{sd_obs:.3e}")
    print(f"  -> the phantom is {dph[jj]/sd_obs:.3f}x the error of one lag")
    print(f"  for comparison, the largest observed |delta-hat| = "
          f"{dhat_max:.3e}")
    out["delta_phantom"] = dict(max=float(dph[jj]), at_lag=int(ks[jj]),
                                sigma_delta=sd_obs,
                                ratio_to_sigma=float(dph[jj] / sd_obs),
                                max_observed_delta=dhat_max)

    print("\n  PASSED THROUGH THE MATCHED FILTER:  "
          "eps_phantom(tau) = sum W*rho / Q")
    print(f"    {'kernel':<12} {'tau':>7} {'eps_phantom':>12} {'eps_excl':>12} "
          f"{'ratio':>10}")
    kax = np.arange(-K, K + 1, dtype=np.float64)
    P = m5["pairs"]["OA vs SB"]
    worst = 0.0
    eph_all = {}
    for kn in KERNELS:
        eph_all[kn] = []
        for jt, tau in enumerate(taus):
            W = kernel_W(kn, kax, float(tau))
            Q = float((W ** 2).sum())
            eph = float(abs(np.dot(W, r_ab)) / Q)
            eph_all[kn].append(eph)
            ratio = eph / P[kn]["eps_excl"][jt]
            worst = max(worst, ratio)
            if tau in (1., 10., 100., 1000., 10000.):
                print(f"    {kn:<12} {tau:>7g} {eph:>12.3e} "
                      f"{P[kn]['eps_excl'][jt]:>12.3e} {ratio:>10.4f}")
    out["eps_phantom"] = eph_all
    out["worst_ratio_to_bound"] = worst
    print(f"\n  WORST ratio eps_phantom/eps_excl over 104 points: "
          f"{worst:.4f}")
    print(f"  -> the phantom signal is {1/worst:.0f}x BELOW the bound at its "
          f"least favourable point")

    verdict = (out["rho_AB"]["n_above"] == 0 and n_above == 0)
    print("\n" + "=" * 78)
    print(f"VERDICT: {'NO setting correlation above the threshold' if verdict else '*** CORRELATION FOUND - STOP ***'}")
    print("=" * 78)
    out["clean"] = bool(verdict)
    json.dump(out, open(os.path.join(HERE, "meros8_settings.json"), "w"),
              indent=2)
    np.savez_compressed(os.path.join(HERE, "meros8_rho.npz"),
                        lags=ks, rho_AB=r_ab, rho_AA=r_aa, rho_BB=r_bb)
    print("Saved: meros8_settings.json + meros8_rho.npz")


if __name__ == "__main__":
    main()
