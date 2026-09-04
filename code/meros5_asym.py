"""
ΜΕΡΟΣ 5 — ΑΣΥΜΜΕΤΡΟΙ ΠΥΡΗΝΕΣ (ΑΓΝΩΣΤΟ #5 της αναφοράς #6)

Ο χάρτης της αναφοράς #6 χρησιμοποίησε ΣΥΜΜΕΤΡΙΚΟ πυρήνα W_τ(k)=exp(-k²/2τ²).
Τα retrocausal μοντέλα (Price, Wharton) είναι future-input dependent, δηλαδή
ΜΟΝΟΠΛΕΥΡΑ. Ένας συμμετρικός πυρήνας δεν τα αποκλείει.

ΣΥΜΒΑΣΗ ΠΡΟΣΗΜΟΥ (κρίσιμη)
    Το μοντέλο είναι λ(i) = λ₀(i) + ε·Σ_k W(k)·S(i+k).
    k > 0  ->  η ρύθμιση στο ΜΕΛΛΟΝ (i+k) επηρεάζει το αποτέλεσμα στο i
               = retrocausal / future-input.
    k < 0  ->  η ρύθμιση στο ΠΑΡΕΛΘΟΝ επηρεάζει το τώρα = κανονικό αίτιο
               (αλλά απαγορευμένο από no-signalling στα spacelike ζεύγη).
    Το δ̂(k) που μετράει το scan() είναι ακριβώς η εξάρτηση του O(i) από το
    S(i+k), άρα το προφίλ του δ̂ σε k είναι ο ίδιος ο πυρήνας: δ(k)=α·ε·W(k).

ΟΙ ΤΕΣΣΕΡΙΣ ΠΥΡΗΝΕΣ
    sym        exp(-k²/2τ²)              για κάθε k ≠ 0  (αναφορά #6· το k=0
                                          εξαιρέθηκε στην αναθεώρηση 2026-09-04)
    future     exp(-k²/2τ²) για k>0,  0 για k<=0
    past       exp(-k²/2τ²) για k<0,  0 για k>=0
    exp_future exp(-k/τ)    για k>0,  0 για k<=0

    Η σύμβαση του ε μένει ίδια: ε = 1 ισοδυναμεί με πλάτος πυρήνα α στην
    κορυφή. Για τους μονόπλευρους η «κορυφή» είναι το k=+1 (ή -1), όχι το
    k=0 — το k=0 ΔΕΝ ανήκει στο μοντέλο και δεν συνεισφέρει στο φίλτρο.
    Αυτό είναι σημαντικό: το k=0 είναι το μόνο lag με πραγματική (κβαντική)
    συσχέτιση, και οι μονόπλευροι πυρήνες το αγνοούν εξ ορισμού.

ΤΟ ΣΤΑΤΙΣΤΙΚΟ — ίδιο με το ΜΕΡΟΣ 3, αλλάζει μόνο το W:
    T(τ) = Σ_k W(k)·δ̂(k),  E[T] = α·ε·Q,  Q = Σ_k W(k)²
    ε̂ = T/(αQ),  σ_ε = σ_T/(αQ),  ε_excl = |ε̂| + z_thr·σ_ε

    Αφού σ_δ(k) ≈ σταθερό, σ_T ≈ σ_δ√Q -> ε_excl ≈ z·σ_δ/(α√Q).
    Μονόπλευρος γκαουσιανός: Q μισό του συμμετρικού -> ε_excl × √2.
    Αυτή είναι η ΑΝΑΜΕΝΟΜΕΝΗ απάντηση· ελέγχεται αν βγαίνει.

ΚΑΤΩΦΛΙ: το ίδιο z_thr = 4,848 (Bonferroni m = 40.002 της αναφοράς #5),
όχι διορθωμένο για τους 4 πυρήνες — μένει συντηρητικό και συγκρίσιμο.
"""
import argparse, json, math, os, time
import numpy as np
from scipy.stats import chi2

from meros2_injection import scan, mi_and_delta
from meros3_map import sigma_delta

HERE = os.path.dirname(os.path.abspath(__file__))
K = 10_000
M_BONF = 40_002

KERNELS = ["sym", "future", "past", "exp_future"]
KERNEL_LABEL = {
    "sym":        "συμμετρικός γκαουσιανός  exp(-k²/2τ²)",
    "future":     "μόνο μέλλον (γκαουσιανός)  k > 0",
    "past":       "μόνο παρελθόν (γκαουσιανός)  k < 0",
    "exp_future": "μόνο μέλλον (εκθετικός)  exp(-k/τ), k > 0",
}


def kernel_W(name, k, tau):
    """W(k) για δοσμένο πυρήνα. k: πίνακας ακεραίων (float), tau: float."""
    # k = 0 ΕΞΑΙΡΕΙΤΑΙ από ΟΛΟΥΣ τους πυρήνες (αναθεώρηση 2026-09-04): ο όρος
    # k = 0 είναι η ρύθμιση του ΙΔΙΟΥ trial (no-signalling, Πίνακας 3), όχι
    # cross-trial. Οι μονόπλευροι το απέκλειαν ήδη· τώρα και ο συμμετρικός.
    if name == "sym":
        return np.where(k != 0, np.exp(-k ** 2 / (2.0 * tau ** 2)), 0.0)
    if name == "future":
        return np.where(k > 0, np.exp(-k ** 2 / (2.0 * tau ** 2)), 0.0)
    if name == "past":
        return np.where(k < 0, np.exp(-k ** 2 / (2.0 * tau ** 2)), 0.0)
    if name == "exp_future":
        return np.where(k > 0, np.exp(-np.abs(k) / tau), 0.0)
    raise ValueError(name)


def build_filters(name, taus, K):
    k = np.arange(-K, K + 1, dtype=np.float64)
    return np.stack([kernel_W(name, k, float(t)) for t in taus])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--shuffles", type=int, default=400)
    ap.add_argument("--n-tau", type=int, default=25)
    ap.add_argument("--seed", type=int, default=4711)
    a = ap.parse_args()

    cal = json.load(open(os.path.join(HERE, "meros1_alpha.json")))
    alpha_A = cal["OA vs SA"]["alpha"]
    alpha_B = cal["OB vs SB"]["alpha"]

    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    SA, SB, OA, OB = d['SA'], d['SB'], d['OA'], d['OB']
    n = len(SA)

    # ίδιο πλέγμα με τη #6, + τα 30 και 300 όπου γίνεται η επαλήθευση
    taus = np.round(np.logspace(0, 4, a.n_tau)).astype(int)
    taus = np.unique(np.concatenate([taus, [30, 300]])).astype(float)

    Wmats = {name: build_filters(name, taus, a.K) for name in KERNELS}
    Qs = {name: (Wmats[name] ** 2).sum(axis=1) for name in KERNELS}

    p_thr = 0.05 / M_BONF
    z_thr = float(math.sqrt(chi2.ppf(1 - p_thr, 1)))

    print("=" * 78)
    print("ΜΕΡΟΣ 5 — ΧΑΡΤΗΣ ΑΠΟΚΛΕΙΣΜΟΥ ΜΕ ΑΣΥΜΜΕΤΡΟΥΣ ΠΥΡΗΝΕΣ")
    print("=" * 78)
    print(f"  n = {n:,}   K = ±{a.K:,}   τ: {len(taus)} τιμές")
    print(f"  z_thr = {z_thr:.3f} (Bonferroni m = {M_BONF:,}, ίδιο με #5/#6)")
    print("\n  Q(τ) ανά πυρήνα (κόψιμο στο ±K συμπεριλαμβάνεται):")
    hdr = "    " + f"{'τ':>7}" + "".join(f"{k:>13}" for k in KERNELS) + \
          f"{'fut/sym':>10}{'exp/sym':>10}"
    print(hdr)
    for j, t in enumerate(taus):
        if t in (1., 10., 100., 1000., 10000.):
            print("    " + f"{t:>7g}" +
                  "".join(f"{Qs[k][j]:>13.1f}" for k in KERNELS) +
                  f"{Qs['future'][j]/Qs['sym'][j]:>10.4f}"
                  f"{Qs['exp_future'][j]/Qs['sym'][j]:>10.4f}")
    print()

    out = {"n": n, "K": a.K, "taus": taus.tolist(), "z_thr": z_thr,
           "m_bonf": M_BONF, "alpha_A": alpha_A, "alpha_B": alpha_B,
           "shuffles": a.shuffles, "kernels": KERNELS,
           "Q": {k: Qs[k].tolist() for k in KERNELS}, "pairs": {}}

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

        # συμμετρία του ΘΟΡΥΒΟΥ: μέσο δ̂ στα k>0 vs k<0 (διαγνωστικό)
        pos = delta[ks > 0]; neg = delta[ks < 0]
        print(f"  δ̂ μέσος k>0: {pos.mean():+.3e}   k<0: {neg.mean():+.3e}   "
              f"(σ/√N = {sd.mean()/math.sqrt(len(pos)):.3e})")

        # --- εμπειρικό null του T(τ) ανά πυρήνα, ΕΝΑΣ βρόχος ανακατεμάτων ---
        print(f"  εμπειρικό null με {a.shuffles} ανακατέματα ρυθμίσεων…",
              flush=True)
        sh = s1.copy()
        Tnull = {k: np.empty((a.shuffles, len(taus))) for k in KERNELS}
        t0 = time.time()
        for i in range(a.shuffles):
            rng.shuffle(sh)
            _, n11s, A1s, B1s, nks, _ = scan(o, sh, a.K)
            _, ds = mi_and_delta(n11s, A1s, B1s, nks)
            for kn in KERNELS:
                Tnull[kn][i] = Wmats[kn] @ ds
            if i == 0:
                print(f"    (~{(time.time()-t0)*a.shuffles/60:.1f} λεπτά)",
                      flush=True)
        print(f"    ολοκληρώθηκε σε {(time.time()-t0)/60:.1f} λεπτά")

        pair_out = {}
        for kn in KERNELS:
            Wm = Wmats[kn]; Q = Qs[kn]
            sT_emp = Tnull[kn].std(axis=0, ddof=1)
            sT_ana = np.sqrt(Wm ** 2 @ sd ** 2)
            ratio = sT_emp / sT_ana
            null_mean = (Tnull[kn].mean(axis=0) / sT_emp).mean()
            sT = np.maximum(sT_emp, sT_ana)          # συντηρητικό

            T = Wm @ delta
            z = T / sT
            eps_hat = T / (alpha * Q)
            sig_eps = sT / (alpha * Q)
            eps_excl = np.abs(eps_hat) + z_thr * sig_eps
            nab = int((np.abs(z) > z_thr).sum())

            print(f"\n  --- {kn}: {KERNEL_LABEL[kn]}")
            print(f"      σ_T εμπ/αναλ: μέσος {ratio.mean():.4f} "
                  f"[{ratio.min():.4f}, {ratio.max():.4f}]   "
                  f"null κέντρο {null_mean:+.3f}σ")
            print(f"      {'τ':>7} {'Q':>10} {'z':>7} {'ε_excl':>11} "
                  f"{'×sym':>7}")
            for j, t in enumerate(taus):
                r = eps_excl[j] / (pair_out['sym']['eps_excl'][j]
                                   if kn != 'sym' else eps_excl[j])
                print(f"      {t:>7g} {Q[j]:>10.1f} {z[j]:>7.2f} "
                      f"{eps_excl[j]:>11.4e} {r:>7.3f}")
            print(f"      |z| > {z_thr:.2f} σε {nab}/{len(taus)} τ   "
                  f"max|z| = {np.abs(z).max():.2f} στο τ = "
                  f"{taus[int(np.argmax(np.abs(z)))]:g}")

            pair_out[kn] = dict(
                Q=Q.tolist(), T=T.tolist(), z=z.tolist(), sigma_T=sT.tolist(),
                sigma_T_emp=sT_emp.tolist(), sigma_T_ana=sT_ana.tolist(),
                ratio_sT=float(ratio.mean()), null_center=float(null_mean),
                eps_hat=eps_hat.tolist(), sigma_eps=sig_eps.tolist(),
                eps_excl=eps_excl.tolist(), n_above=nab,
                max_abs_z=float(np.abs(z).max()))

        out["pairs"][label] = pair_out
        np.savez_compressed(
            os.path.join(HERE, f"meros5_delta_{label.replace(' ', '_')}.npz"),
            lags=ks, delta=delta, sigma=sd, mi=MI)
        print()

    json.dump(out, open(os.path.join(HERE, "meros5_asym.json"), "w"), indent=2)
    print("Αποθηκεύτηκε: meros5_asym.json + meros5_delta_*.npz")


if __name__ == "__main__":
    main()
