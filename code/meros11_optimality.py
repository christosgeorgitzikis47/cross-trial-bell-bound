"""
ΜΕΡΟΣ 11 — ΟΙ ΠΡΟΫΠΟΘΕΣΕΙΣ ΤΟΥ MATCHED FILTER ΚΑΙ Η ΑΠΩΛΕΙΑ ΑΝΑΝΤΙΣΤΟΙΧΙΑΣ

(α) ΒΕΛΤΙΣΤΟΤΗΤΑ. Το matched filter είναι Neyman–Pearson βέλτιστο για ΓΝΩΣΤΟ
    σχήμα σήματος μέσα σε ΓΚΑΟΥΣΙΑΝΟ ΛΕΥΚΟ θόρυβο. Δύο προϋποθέσεις, και οι δύο
    ελέγξιμες στα ίδια τα δεδομένα:
      – λευκότητα: τα δ̂(k) ασυσχέτιστα ως προς k (ήδη μετρημένο, |r| ≤ 0,014)
      – γκαουσιανότητα: KS των τυποποιημένων δ̂(k)/σ_δ(k) στα 20.001 lag.
    Ίδια μεθοδολογία με τον έλεγχο των ρυθμίσεων του §5.1.

(β) ΑΠΩΛΕΙΑ ΑΝΑΝΤΙΣΤΟΙΧΙΑΣ. Το mirror test κάλυψε την ακραία περίπτωση (λάθος
    πρόσημο). Εδώ η ενδιάμεση: σήμα πλάτους τ_true φιλτραρισμένο με τ_filter.

    Υπάρχει και κλειστός τύπος, γιατί το z είναι λόγος αθροισμάτων:
        z(τ_f | τ_t) ∝ Σ_k W_f(k)·W_t(k) / √(Σ_k W_f(k)²)
    Δίνεται δίπλα στο μετρημένο — αν συμφωνούν, η μέτρηση επιβεβαιώνει τον τύπο
    και ο τύπος γενικεύει τη μέτρηση σε κάθε λόγο τ_f/τ_t.

    ΤΟ ΚΡΙΣΙΜΟ ΕΡΩΤΗΜΑ: το πλέγμα των 26 τ έχει βήμα 10^(4/24) = 1,468. Σήμα
    ακριβώς ανάμεσα σε δύο σημεία απέχει παράγοντα √1,468 = 1,212. Πόσο χάνουμε
    εκεί; Αν είναι < 10%, το πλέγμα είναι αρκετά πυκνό.
"""
import argparse, json, math, os
import numpy as np
from scipy.stats import kstest, norm

from meros2_injection import scan, mi_and_delta
from meros3_map import sigma_delta
from meros5_asym import kernel_W
from meros5_verify import build_F_kernel

HERE = os.path.dirname(os.path.abspath(__file__))
K = 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--seed", type=int, default=3141)
    ap.add_argument("--snr-eps", type=float, default=4.0,
                    help="ένεση σε πολλαπλάσιο του ε_excl, ώστε ο λόγος z να "
                         "μετριέται με αμελητέο θόρυβο")
    a = ap.parse_args()

    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    cal = json.load(open(os.path.join(HERE, "meros1_alpha.json")))["OA vs SA"]
    alpha, r1, r2 = cal["alpha"], cal["r1"], cal["r2"]
    taus = np.array(m5["taus"])
    kax = np.arange(-a.K, a.K + 1, dtype=np.float64)

    print("=" * 78)
    print("ΜΕΡΟΣ 11 — ΠΡΟΫΠΟΘΕΣΕΙΣ ΒΕΛΤΙΣΤΟΤΗΤΑΣ ΚΑΙ ΑΝΑΝΤΙΣΤΟΙΧΙΑ ΠΥΡΗΝΑ")
    print("=" * 78)

    # ---------------- (α) γκαουσιανότητα των δ̂ ----------------
    print("\n(α) ΓΚΑΟΥΣΙΑΝΟΤΗΤΑ ΤΩΝ δ̂(k)")
    out = {"gaussianity": {}}
    for pair in ("OA_vs_SB", "OB_vs_SA"):
        d = np.load(os.path.join(HERE, f"meros5_delta_{pair}.npz"))
        z = d["delta"] / d["sigma"]
        ks = kstest(z, "norm")
        # και οι ουρές, που είναι ό,τι μετράει για κατώφλι 4,85σ
        tails = {t: int((np.abs(z) > t).sum()) for t in (2, 3, 4, 4.848)}
        exp = {t: 2 * norm.sf(t) * len(z) for t in (2, 3, 4, 4.848)}
        print(f"  {pair.replace('_',' ')}: n = {len(z):,}   "
              f"μέσος {z.mean():+.4f}   sd {z.std(ddof=1):.4f}   "
              f"KS p = {ks.pvalue:.3f}")
        print("     ουρές: " + "  ".join(
            f"|z|>{t}: {tails[t]} (αναμ. {exp[t]:.1f})" for t in (2, 3, 4)))
        out["gaussianity"][pair] = dict(
            n=int(len(z)), mean=float(z.mean()), sd=float(z.std(ddof=1)),
            ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
            tails={str(t): tails[t] for t in tails},
            tails_expected={str(t): exp[t] for t in exp})

    # ---------------- (β) αναντιστοιχία πυρήνα ----------------
    print("\n(β) ΑΠΩΛΕΙΑ ΑΝΑΝΤΙΣΤΟΙΧΙΑΣ  z(τ_filter)/z(τ_true)")
    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    SA, SB = d['SA'], d['SB']
    n = len(SA)
    S = np.where(SB == 2, 1.0, -1.0)
    SB1 = (SB == 1).astype(np.int8)
    lam0 = np.where(SA == 1, r1, r2)
    rng = np.random.default_rng(a.seed)
    P = m5["pairs"]["OA vs SB"]["sym"]

    ratios = [0.1, 1 / 3, 0.5, 1.0, 2.0, 3.0, 10.0]
    grid_step = 10 ** (4 / 24)
    extra = [1 / math.sqrt(grid_step), math.sqrt(grid_step)]   # μισό βήμα πλέγματος
    all_ratios = sorted(set([round(r, 4) for r in ratios + extra]))

    def theory(tf, tt):
        Wf = kernel_W("sym", kax, tf); Wt = kernel_W("sym", kax, tt)
        return float(np.dot(Wf, Wt) / math.sqrt((Wf ** 2).sum()))

    rows = []
    print(f"    {'τ_true':>7} " + "".join(f"{r:>9.3g}" for r in all_ratios))
    for tt in (30.0, 300.0, 3000.0):
        j = int(np.argmin(np.abs(taus - tt)))
        eps = a.snr_eps * P["eps_excl"][j]
        F, _ = build_F_kernel(S, "sym", tt, a.K)
        zmat = np.zeros((a.reps, len(all_ratios)))
        for r in range(a.reps):
            lam = lam0 + alpha * eps * F
            O = (rng.random(n) < np.clip(lam, 0, 1)).astype(np.int8)
            _, n11, A1, B1, nk, _ = scan(O, SB1, a.K)
            _, dl = mi_and_delta(n11, A1, B1, nk)
            sd = sigma_delta(A1, B1, nk)
            for i, rr in enumerate(all_ratios):
                Wf = kernel_W("sym", kax, tt * rr)
                zmat[r, i] = float(np.dot(Wf, dl) /
                                   math.sqrt(float(np.dot(Wf ** 2, sd ** 2))))
        z = zmat.mean(axis=0)
        i1 = all_ratios.index(1.0)
        meas = z / z[i1]
        th = np.array([theory(tt * rr, tt) for rr in all_ratios])
        th = th / th[i1]
        print(f"    {tt:>7g} " + "".join(f"{v:>9.3f}" for v in meas))
        print(f"    {'(τύπος)':>7} " + "".join(f"{v:>9.3f}" for v in th))
        rows.append(dict(tau_true=tt, eps=eps, ratios=all_ratios,
                         z_mean=z.tolist(), measured=meas.tolist(),
                         theory=th.tolist(),
                         z_matched=float(z[i1])))
        del F

    # απώλεια στο μισό βήμα του πλέγματος
    half = [r for r in all_ratios if abs(r - extra[1]) < 1e-3][0]
    ih = all_ratios.index(half)
    losses = [1 - r["measured"][ih] for r in rows]
    th_loss = [1 - r["theory"][ih] for r in rows]
    print(f"\n  Βήμα πλέγματος: 10^(4/24) = {grid_step:.3f}   "
          f"μισό βήμα = ×{half:.3f}")
    print(f"  Απώλεια εκεί: μετρημένη {100*np.mean(losses):.1f}%  "
          f"(τύπος {100*np.mean(th_loss):.1f}%)  -> πλέγμα "
          f"{'ΑΡΚΕΤΑ ΠΥΚΝΟ' if np.mean(th_loss) < 0.10 else 'ΑΡΑΙΟ'}")
    out["mismatch"] = dict(rows=rows, grid_step=grid_step,
                           half_step=half,
                           loss_measured=float(np.mean(losses)),
                           loss_theory=float(np.mean(th_loss)))

    # ---------------- (γ) γιατί ±10.000 ----------------
    print("\n(γ) ΤΟ ΕΥΡΟΣ ±10.000")
    for frac in (0.001, 0.01, 0.1):
        kk = frac * n
        print(f"    n_eff πέφτει {100*frac:>4.1f}% στο |k| = {kk:,.0f}  "
              f"(ισχύς ∝ √n_eff: −{100*(1-math.sqrt(1-frac)):.2f}%)")
    print(f"    στο |k| = 10.000: n_eff = {100*(1-1e4/n):.3f}% του n, "
          f"απώλεια ισχύος {100*(1-math.sqrt(1-1e4/n)):.4f}%")
    out["range"] = dict(n=n, k_at_10pct=0.1 * n,
                        power_loss_at_10k=float(1 - math.sqrt(1 - 1e4 / n)))

    json.dump(out, open(os.path.join(HERE, "meros11_optimality.json"), "w"),
              indent=2)
    print("\nΑποθηκεύτηκε: meros11_optimality.json")


if __name__ == "__main__":
    main()
