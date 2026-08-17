"""
ΜΕΡΟΣ 8 — ΣΥΣΧΕΤΙΣΗ ΤΩΝ ΡΥΘΜΙΣΕΩΝ: ΤΟ ΦΑΝΤΑΣΤΙΚΟ ΣΗΜΑ

Η ένσταση (κριτής #2): το O_A(i) εξαρτάται ΕΝΤΟΝΑ από το S_A(i) — αυτό είναι
απλή κβαντομηχανική, δ_A = 1,96e-3. Αν το S_B(i+k) είναι συσχετισμένο με το
S_A(i), τότε το διασταυρούμενο κανάλι δείχνει σήμα ΧΩΡΙΣ καμία νέα φυσική.

Η ΑΛΓΕΒΡΑ (γιατί ο τύπος είναι αυτός)
    Με κωδικοποίηση S = ±1 και ισόπιθανες ρυθμίσεις, ο ρυθμός click της Alice
    είναι  p(i) = p₀ + δ_A·S_A(i).  Άρα

      E[O_A | S_B(i+k)=s] = p₀ + δ_A·E[S_A(i) | S_B(i+k)=s] = p₀ + δ_A·ρ(k)·s

    (για συμμετρικά δυαδικά, E[S_A|S_B=s] = ρ·s). Το μετρούμενο δ̂ είναι η μισή
    διαφορά ανάμεσα στα s = ±1, δηλαδή

      δ_phantom(k) = δ_A · ρ(k),        ρ(k) = Corr(S_A(i), S_B(i+k))

    Περασμένο από το ΙΔΙΟ matched filter, δίνει φανταστικό ε:

      ε_phantom(τ) = Σ_k W_τ(k)·δ_phantom(k) / (α Q) = Σ_k W_τ(k)·ρ(k) / Q(τ)

    δηλαδή το α απλοποιείται: το φανταστικό ε είναι ο σταθμισμένος μέσος της
    συσχέτισης των ρυθμίσεων. Καθαρό μέγεθος για σύγκριση με το ε_excl.

ΤΙ ΜΕΤΡΙΕΤΑΙ
    ρ_AB(k) = Corr(S_A(i), S_B(i+k))   [το κρίσιμο: διασταυρούμενη]
    ρ_AA(k), ρ_BB(k)                   [αυτοσυσχετίσεις]
    I(S_A(i) ; S_B(i+k)) με το ΙΔΙΟ κατώφλι χ²(1)/Bonferroni του §6.1
    ε_phantom(τ) για τους 4 πυρήνες, δίπλα στο ε_excl

ΑΝ κάτι περάσει το κατώφλι, ΤΟ ΛΕΜΕ: θα ήταν εύρημα για το ίδιο το πείραμα
(οι γεννήτριες ρυθμίσεων δεν θα ήταν ανεξάρτητες), όχι για το μοντέλο μας.
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
    """ρ(k) = Corr(x(i), y(i+k)) για |k| ≤ K, με ένα ζεύγος FFT.
    Επιστρέφει τον συντελεστή Pearson ανά lag (κάθε lag έχει n−|k| ζεύγη)."""
    n = len(x)
    xf = x.astype(np.float64); yf = y.astype(np.float64)
    L = next_fast_len(n + K + 1)
    R = irfft(np.conjugate(rfft(xf, L)) * rfft(yf, L), L)
    s_xy = np.concatenate([R[L - K:], R[:K + 1]])          # Σ x(i)·y(i+k)

    ks = np.arange(-K, K + 1)
    cx = np.concatenate([[0.0], np.cumsum(xf)])
    cy = np.concatenate([[0.0], np.cumsum(yf)])
    cx2 = np.concatenate([[0.0], np.cumsum(xf ** 2)])
    cy2 = np.concatenate([[0.0], np.cumsum(yf ** 2)])
    sx = np.empty(len(ks)); sy = np.empty(len(ks))
    sx2 = np.empty(len(ks)); sy2 = np.empty(len(ks))
    for j, k in enumerate(ks):
        if k >= 0:                       # i από 0 ως n−k−1, i+k από k ως n−1
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
    print(f"  {name:<26} max |ρ| = {abs(rho[j]):.3e} στο lag {ks[j]:>+7d}"
          f"   ({abs(rho[j])/se:>5.2f}σ)   |ρ| > 4,848σ σε {n_above} lag")
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
    sa = np.where(SA == 2, 1.0, -1.0)          # ±1
    sb = np.where(SB == 2, 1.0, -1.0)
    se = 1.0 / math.sqrt(n)

    print("=" * 78)
    print("ΜΕΡΟΣ 8 — ΣΥΣΧΕΤΙΣΗ ΡΥΘΜΙΣΕΩΝ ΚΑΙ ΦΑΝΤΑΣΤΙΚΟ ΣΗΜΑ")
    print("=" * 78)
    print(f"  n = {n:,}   |k| ≤ {K:,}   τυπικό σφάλμα του ρ: 1/√n = {se:.3e}")
    print(f"  κατώφλι 4,848σ  ->  |ρ| > {4.848*se:.3e}\n")

    out = {"n": n, "se_rho": se, "z_thr": z_thr}
    print("  ΣΥΣΧΕΤΙΣΕΙΣ (Pearson, ±1 κωδικοποίηση):")
    ks, r_ab = xcorr(sa, sb, K)
    out["rho_AB"] = report("Corr(S_A(i), S_B(i+k))", ks, r_ab, se)
    _, r_aa = xcorr(sa, sa, K)
    r_aa0 = r_aa.copy(); r_aa0[ks == 0] = 0.0          # το k=0 είναι 1 εξ ορισμού
    out["rho_AA"] = report("Corr(S_A(i), S_A(i+k)), k≠0", ks, r_aa0, se)
    _, r_bb = xcorr(sb, sb, K)
    r_bb0 = r_bb.copy(); r_bb0[ks == 0] = 0.0
    out["rho_BB"] = report("Corr(S_B(i), S_B(i+k)), k≠0", ks, r_bb0, se)

    # ---------- αμοιβαία πληροφορία των ρυθμίσεων ----------
    print("\n  ΑΜΟΙΒΑΙΑ ΠΛΗΡΟΦΟΡΙΑ I(S_A(i) ; S_B(i+k)):")
    a1 = (SA == 1).astype(np.int8); b1 = (SB == 1).astype(np.int8)
    ks2, n11, A1, B1, nk, ferr = scan(a1, b1, K)
    MI, _ = mi_and_delta(n11, A1, B1, nk)
    G = 2 * nk * LN2 * MI
    mi_thr = z_thr ** 2 / (2 * n * LN2)
    j = int(np.argmax(MI))
    n_above = int((G > z_thr ** 2).sum())
    print(f"    κατώφλι (ίδιο Bonferroni, m = 40.002): "
          f"MI = {mi_thr:.4e} bits/trial  (G = {z_thr**2:.2f})")
    print(f"    μέγιστο MI = {MI[j]:.4e} στο lag {ks2[j]:+d}   "
          f"G = {G[j]:.2f}  (√G = {math.sqrt(G[j]):.2f}σ)")
    print(f"    lag πάνω από το κατώφλι: {n_above} / {len(ks2):,}")
    out["mi_settings"] = dict(threshold=float(mi_thr), max_mi=float(MI[j]),
                              at_lag=int(ks2[j]), max_G=float(G[j]),
                              n_above=n_above, fft_err=float(ferr))

    # ---------- το φανταστικό σήμα ----------
    print("\n" + "=" * 78)
    print("ΦΑΝΤΑΣΤΙΚΟ ΣΗΜΑ:  δ_phantom(k) = δ_A · ρ(k)")
    print("=" * 78)
    dph = aA * np.abs(r_ab)
    jj = int(np.argmax(dph))
    m3 = np.load(os.path.join(HERE, "meros5_delta_OA_vs_SB.npz"))
    sd_obs = float(m3["sigma"].mean())
    dhat_max = float(np.abs(m3["delta"]).max())
    print(f"  μέγιστο |δ_phantom| = {dph[jj]:.3e} στο lag {ks[jj]:+d}")
    print(f"  τυπικό σφάλμα του δ̂ ανά lag (μέσο σ_δ) = {sd_obs:.3e}")
    print(f"  -> το φανταστικό είναι {dph[jj]/sd_obs:.3f}× το σφάλμα ενός lag")
    print(f"  για σύγκριση, το μέγιστο παρατηρούμενο |δ̂| = {dhat_max:.3e}")
    out["delta_phantom"] = dict(max=float(dph[jj]), at_lag=int(ks[jj]),
                                sigma_delta=sd_obs,
                                ratio_to_sigma=float(dph[jj] / sd_obs),
                                max_observed_delta=dhat_max)

    print("\n  ΠΕΡΑΣΜΕΝΟ ΑΠΟ ΤΟ MATCHED FILTER:  ε_phantom(τ) = Σ W·ρ / Q")
    print(f"    {'πυρήνας':<12} {'τ':>7} {'ε_phantom':>12} {'ε_excl':>12} "
          f"{'λόγος':>10}")
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
    print(f"\n  ΧΕΙΡΟΤΕΡΟΣ λόγος ε_phantom/ε_excl σε 104 σημεία: {worst:.4f}")
    print(f"  -> το φανταστικό σήμα είναι {1/worst:.0f}× ΚΑΤΩ από το όριο "
          f"στο δυσμενέστερο σημείο")

    verdict = (out["rho_AB"]["n_above"] == 0 and n_above == 0)
    print("\n" + "=" * 78)
    print(f"ΕΤΥΜΗΓΟΡΙΑ: {'ΚΑΜΙΑ συσχέτιση ρυθμίσεων πάνω από το κατώφλι' if verdict else '*** ΒΡΕΘΗΚΕ ΣΥΣΧΕΤΙΣΗ — ΣΤΑΜΑΤΑ ***'}")
    print("=" * 78)
    out["clean"] = bool(verdict)
    json.dump(out, open(os.path.join(HERE, "meros8_settings.json"), "w"),
              indent=2)
    np.savez_compressed(os.path.join(HERE, "meros8_rho.npz"),
                        lags=ks, rho_AB=r_ab, rho_AA=r_aa, rho_BB=r_bb)
    print("Αποθηκεύτηκε: meros8_settings.json + meros8_rho.npz")


if __name__ == "__main__":
    main()
