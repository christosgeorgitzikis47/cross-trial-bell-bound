"""
ΜΕΡΟΣ 3 — ΧΑΡΤΗΣ ΑΠΟΚΛΕΙΣΜΟΥ ΣΤΟ ΕΠΙΠΕΔΟ (ε, τ)

ΤΟ ΣΤΑΤΙΣΤΙΚΟ (matched filter στο δ, όχι στο I)
    Το μοντέλο προβλέπει δ(k) = α·ε·W_τ(k),  W_τ(k)=exp(-k²/2τ²).
    Το δ̂(k) είναι ΓΡΑΜΜΙΚΟ στο σήμα και ο θόρυβός του συμμετρικός γύρω
    από το μηδέν. Το Î(k) είναι τετραγωνικό και ο θόρυβός του θετικά
    ορισμένος -> το φιλτράρισμα του I συσσωρεύει bias. Άρα:

        T(τ) = Σ_k W_τ(k)·δ̂(k)          (κύριο)
        U(τ) = Σ_k W_τ(k)²·Î(k)          (δευτερεύον, για σύγκριση)

    Υπό το μοντέλο:  E[T] = α·ε·Σ_k W_τ(k)²  ≡ α·ε·Q(τ)
    Υπό H0:          Var[T] = Σ_k W_τ(k)²·σ_δ(k)²   (αν τα δ̂(k) είναι
                     ασυσχέτιστα — ΕΠΑΛΗΘΕΥΕΤΑΙ, δεν υποτίθεται)

    -> ε̂(τ) = T/(α·Q),  σ_ε(τ) = σ_T/(α·Q)
    -> ΑΝΩ ΟΡΙΟ:  ε_excl(τ) = |ε̂| + z_thr·σ_ε

Το k = 0 ΕΞΑΙΡΕΙΤΑΙ από το W (αναθεώρηση 2026-09-04): είναι η ρύθμιση του
ίδιου trial — no-signalling, όχι cross-trial — και φράσσεται χωριστά.

Q(τ) = Σ_{0<|k|<=10000} W_τ(k)² υπολογίζεται ΑΚΡΙΒΩΣ (πεπερασμένο άθροισμα),
όχι με την ασυμπτωτική τ√π. Για τ > ~3000 το παράθυρο κόβει τις ουρές και
το Q υπολείπεται του τ√π -> ΚΑΜΨΗ στην καμπύλη. Δεν λειαίνεται.

ΚΑΤΩΦΛΙ: το ίδιο Bonferroni της αναφοράς #5, m = 40.002 υποθέσεις,
p = 1,25e-6, z_thr = 4,85 (χ²(1) -> 23,5). Δίνεται και το «σωστό για αυτό
το τεστ» m = 2·n_τ, αλλά χρησιμοποιείται το συντηρητικό ώστε ο χάρτης να
είναι άμεσα συγκρίσιμος με το προηγούμενο όριο.
"""
import argparse, json, math, os, sys, time
import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.stats import chi2

from meros2_injection import scan, mi_and_delta, build_F

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2.0)
K = 10_000
M_BONF = 40_002                      # ίδιο με την αναφορά #5


def sigma_delta(A1, B1, nk):
    """σ του δ̂(k)=(r2-r1)/2 από τη διωνυμική, ανά lag."""
    A1f = A1.astype(float); B1f = B1.astype(float); nkf = nk.astype(float)
    p = A1f / nkf
    n1 = B1f; n2 = nkf - B1f
    return 0.5 * np.sqrt(p * (1 - p) * (1.0 / n1 + 1.0 / n2))


def kernel_Q(taus, K):
    """Q(τ) = Σ_{|k|<=K} exp(-k²/τ²)  (= Σ W_τ², ΑΚΡΙΒΩΣ, με το κόψιμο)."""
    k = np.arange(-K, K + 1, dtype=np.float64)
    k = k[k != 0]                       # το k = 0 δεν ανήκει στο μοντέλο
    Q = np.array([np.exp(-k ** 2 / t ** 2).sum() for t in taus])
    Qasym = np.array([t * math.sqrt(math.pi) for t in taus])
    return Q, Qasym


def filters(taus, K):
    """Πίνακας βαρών W_τ(k) (n_τ × n_lag) — για τα μεγάλα τ είναι πυκνός
    αλλά μικρός (20 × 20001)."""
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
    print("ΜΕΡΟΣ 3 — ΧΑΡΤΗΣ ΑΠΟΚΛΕΙΣΜΟΥ (ε, τ)")
    print("=" * 78)
    print(f"  n = {n:,}   K = ±{a.K:,}   τ: {len(taus)} τιμές, {taus[0]:g}…{taus[-1]:g}")
    print(f"  α (Alice) = {alpha_A:.6e}   α (Bob) = {alpha_B:.6e}")
    print(f"  κατώφλι Bonferroni m = {M_BONF:,} (ίδιο με αναφορά #5): "
          f"p = {p_thr:.3e}  ->  z = {z_thr:.3f}")
    print(f"  (το «δικό του» m = {m_own} θα έδινε z = {z_own:.3f} — "
          f"χρησιμοποιείται το συντηρητικό)")
    print(f"  κόψιμο ουρών: Q(τ)/τ√π στο τ=10.000 -> {Q[-1]/Qasym[-1]:.3f}\n")

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
        print(f"  σάρωση {len(ks):,} lag σε {time.time()-t0:.1f}s   "
              f"σφάλμα FFT {ferr:.1e}")
        print(f"  δ̂: μέσος {delta.mean():+.3e}  sd {delta.std(ddof=1):.3e}  "
              f"(αναλυτικό σ_δ {sd.mean():.3e})")

        # --- ασυσχέτιστα τα δ̂(k); (δωρεάν έλεγχος: αυτοσυσχέτιση ως προς k)
        x = (delta - delta.mean()) / delta.std(ddof=1)
        ac = np.array([float(np.mean(x[:-l] * x[l:])) for l in (1, 2, 3, 5, 10)])
        se_ac = 1.0 / math.sqrt(len(x))
        print(f"  αυτοσυσχέτιση του δ̂(k) σε Δk=1,2,3,5,10: "
              f"{np.array2string(ac, precision=4)}  (±{se_ac:.4f})")

        # --- εμπειρικό null του T(τ): ανακατεύουμε τις ΡΥΘΜΙΣΕΙΣ ---
        print(f"  εμπειρικό null του T(τ) με {a.shuffles} ανακατέματα…",
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
                print(f"    (~{(time.time()-t0)*a.shuffles/60:.1f} λεπτά συνολικά)",
                      flush=True)
        sT_emp = Tnull.std(axis=0, ddof=1)
        sT_ana = np.sqrt(Wmat ** 2 @ sd ** 2)
        ratio_sT = sT_emp / sT_ana
        print(f"    σ_T εμπειρικό/αναλυτικό: μέσος {ratio_sT.mean():.4f}  "
              f"εύρος [{ratio_sT.min():.4f}, {ratio_sT.max():.4f}]")
        print(f"    μέσος T_null/σ_T: {(Tnull.mean(axis=0)/sT_emp).mean():+.3f}")
        # χρησιμοποιούμε το ΜΕΓΑΛΥΤΕΡΟ των δύο (συντηρητικό)
        sT = np.maximum(sT_emp, sT_ana)

        # --- το πραγματικό T και το όριο ---
        T = apply_T(Wmat, delta)
        z = T / sT
        eps_hat = T / (alpha * Q)
        sig_eps = sT / (alpha * Q)
        eps_excl = np.abs(eps_hat) + z_thr * sig_eps

        # --- δευτερεύον: φίλτρο W² πάνω στο I ---
        baseline = 1.0 / (2 * n * LN2)
        U = (Wmat ** 2) @ (MI - baseline)
        Unull = None
        # αναλυτικό σ του U: Var(Î) ≈ 2/(2n ln2)² ανά lag (χ²(1))
        sI = math.sqrt(2.0) / (2 * n * LN2)
        sU = np.sqrt(((Wmat ** 2) ** 2).sum(axis=1)) * sI
        zU = U / sU
        # μετατροπή σε ε: E[U] = C ε² Σ W⁴... -> ε_U = sqrt(U_lim / (C·ΣW⁴))
        C = cal["OA vs SA"]["C"] if label == "OA vs SB" else cal["OB vs SB"]["C"]
        SW4 = ((Wmat ** 2) ** 2).sum(axis=1)
        U_lim = np.maximum(U, 0) + z_thr * sU
        eps_U = np.sqrt(U_lim / (C * SW4))

        print(f"\n  {'τ':>7} {'Q(τ)':>10} {'T':>11} {'z':>7} "
              f"{'σ_ε':>10} {'ε_excl(δ)':>10} {'ε_excl(I)':>10}")
        for j, t in enumerate(taus):
            print(f"  {t:>7g} {Q[j]:>10.1f} {T[j]:>11.3e} {z[j]:>7.2f} "
                  f"{sig_eps[j]:>10.3e} {eps_excl[j]:>10.3e} {eps_U[j]:>10.3e}")
        nab = int((np.abs(z) > z_thr).sum())
        print(f"\n  |z| > {z_thr:.2f} σε {nab} / {len(taus)} τιμές τ")
        print(f"  μέγιστο |z| = {np.abs(z).max():.2f} στο τ = "
              f"{taus[int(np.argmax(np.abs(z)))]:g}")
        print(f"  ε_excl εύρος: {eps_excl.min():.3e} … {eps_excl.max():.3e}")
        print(f"  τ με ε_excl >= 1 (μη περιοριστικά): "
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
    print("Αποθηκεύτηκε: meros3_map.json + meros3_delta_*.npz")


if __name__ == "__main__":
    main()
