"""
ΒΗΜΑ Β — ΠΛΗΡΗΣ ΣΑΡΩΣΗ κάθε lag στο -10.000 … +10.000 (20.001 τιμές).

Αντικαθιστά τη δήλωση «στα 111 ελεγμένα lag» με «για ΚΑΘΕ |k| <= 10.000».

--------------------------------------------------------------------------
ΔΥΟ ΤΕΧΝΙΚΑ ΣΗΜΕΙΑ, ΚΑΙ ΤΑ ΔΥΟ ΕΠΑΛΗΘΕΥΟΝΤΑΙ ΜΕΣΑ ΣΤΟ ΣΚΡΙΠΤ
--------------------------------------------------------------------------

1. ΤΟ BONFERRONI ΚΑΤΩΦΛΙ ΔΕΝ ΒΓΑΙΝΕΙ ΕΜΠΕΙΡΙΚΑ.
   20.001 lag x 2 ζεύγη = 40.002 υποθέσεις -> χρειάζεται το
   99,999875ο εκατοστημόριο του null. Με 2.000 ανακατέματα το ακρότατο
   που υπάρχει είναι το ~99,95ο· θα χρειάζονταν ~800.000 ανακατέματα.

   Αντ' αυτού: αναλυτική βαθμονόμηση. Υπό H0 ο πίνακας είναι 2x2
   (αποτέλεσμα {0,1} x ρύθμιση {1,2}), άρα df = 1 και

       G = 2 * n * ln2 * MI   ~   χ²(1)

   -> κατώφλι σε κλειστό τύπο, χωρίς ανακατέματα.
   ΕΠΑΛΗΘΕΥΣΗ (--validate-null): χτίζουμε 2.000 ανακατέματα και ελέγχουμε
   ότι το G τους όντως ακολουθεί χ²(1) — μέσος 1, διασπορά 2, KS, ουρές.
   Αν δεν ταιριάζει, ΔΕΝ χρησιμοποιείται.

   Το Bonferroni ισχύει χωρίς υπόθεση ανεξαρτησίας — και τα 40.002 lag
   είναι συσχετισμένα (μοιράζονται δεδομένα), αλλά αυτό το κάνει
   ΣΥΝΤΗΡΗΤΙΚΟ, όχι άκυρο.

2. ΤΑΧΥΤΗΤΑ: 40.002 απευθείας MI ~ 1 ώρα. Οι μετρήσεις κάθε lag είναι
   διασταυρούμενη συσχέτιση, άρα βγαίνουν ΟΛΕΣ ΜΑΖΙ με FFT:

       N11(k) = Σ_i o[i] * s1[i+k]        s1 = (setting == 1)

   Τα υπόλοιπα τρία κελιά βγαίνουν από τα περιθώρια, που είναι αθροίσματα
   προθέματος. ΕΠΑΛΗΘΕΥΣΗ: 20 τυχαία lag συγκρίνονται με απευθείας
   υπολογισμό· διαφορά != 0 -> διακοπή.
"""
import argparse, json, math, os
import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.stats import chi2, kstest

from lag_test import mi, align

HERE = os.path.dirname(os.path.abspath(__file__))
K = 10_000
LN2 = math.log(2.0)


# ------------------------------------------------------------------ counts
def n11_all_lags(o, s1, K):
    """N11(k) = Σ_i o[i]*s1[i+k] για κάθε k στο [-K, +K], με ΕΝΑ FFT ζεύγος.

    Μηδενικό γέμισμα σε μήκος >= n+K ώστε η κυκλική συσχέτιση να ΤΑΥΤΙΖΕΤΑΙ
    με τη γραμμική στα lag που μας ενδιαφέρουν (καμία αναδίπλωση)."""
    n = len(o)
    L = next_fast_len(n + K + 1)
    O = rfft(o.astype(np.float64), L)
    S = rfft(s1.astype(np.float64), L)
    R = irfft(np.conjugate(O) * S, L)
    pos = R[:K + 1]                 # k = 0 … +K
    neg = R[L - K:]                 # k = -K … -1
    c = np.concatenate([neg, pos])  # k = -K … +K
    return np.rint(c).astype(np.int64), float(np.abs(c - np.rint(c)).max())


def margins(o, s1, K):
    """A1(k) = #{o=1}, B1(k) = #{s=1}, n_k, στα ίδια ευθυγραμμισμένα δείγματα."""
    n = len(o)
    ks = np.arange(-K, K + 1)
    co = np.concatenate([[0], np.cumsum(o, dtype=np.int64)])
    cs = np.concatenate([[0], np.cumsum(s1, dtype=np.int64)])
    tot_o, tot_s = int(co[-1]), int(cs[-1])
    A1 = np.empty(len(ks), np.int64)
    B1 = np.empty(len(ks), np.int64)
    for j, k in enumerate(ks):
        if k >= 0:
            # δείκτες o: 0 … n-k-1 ;  δείκτες s: k … n-1
            A1[j] = co[n - k]
            B1[j] = tot_s - cs[k]
        else:
            m = -k
            # δείκτες o: m … n-1 ;  δείκτες s: 0 … n-m-1
            A1[j] = tot_o - co[m]
            B1[j] = cs[n - m]
    nk = n - np.abs(ks)
    return ks, A1, B1, nk


def mi_from_counts(n11, A1, B1, nk):
    """MI σε bits από τον πίνακα 2x2, διανυσματικά για όλα τα lag."""
    n11 = n11.astype(np.float64); A1 = A1.astype(np.float64)
    B1 = B1.astype(np.float64);   nk = nk.astype(np.float64)
    cells = np.stack([n11, A1 - n11, B1 - n11, nk - A1 - B1 + n11])   # 11,12,01,02
    rows = np.stack([A1, A1, nk - A1, nk - A1])
    cols = np.stack([B1, nk - B1, B1, nk - B1])
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (cells / nk) * np.log2(cells * nk / (rows * cols))
    return np.nansum(t, axis=0)


# ----------------------------------------------------------------- χ² null
def validate_chi2(o, s, n_shuffle, seed=777):
    """Ελέγχει ότι G = 2n ln2 MI των ανακατεμένων δεδομένων ~ χ²(1)."""
    rng = np.random.default_rng(seed)
    sh = s.copy()
    n = len(o)
    g = np.empty(n_shuffle)
    for i in range(n_shuffle):
        rng.shuffle(sh)
        g[i] = 2 * n * LN2 * mi(o, sh)
    ks = kstest(g, 'chi2', args=(1,))
    return {
        "n_shuffle": n_shuffle,
        "mean": float(g.mean()), "mean_theory": 1.0,
        "var": float(g.var(ddof=1)), "var_theory": 2.0,
        "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
        "q50": float(np.percentile(g, 50)), "q50_theory": float(chi2.ppf(.50, 1)),
        "q90": float(np.percentile(g, 90)), "q90_theory": float(chi2.ppf(.90, 1)),
        "q99": float(np.percentile(g, 99)), "q99_theory": float(chi2.ppf(.99, 1)),
        "q999": float(np.percentile(g, 99.9)), "q999_theory": float(chi2.ppf(.999, 1)),
        "max": float(g.max()), "max_theory": float(chi2.ppf(1 - 1/n_shuffle, 1)),
        "null_mi_mean": float(g.mean() / (2 * n * LN2)),
        "null_mi_sd": float(g.std(ddof=1) / (2 * n * LN2)),
        "null_mi_max": float(g.max() / (2 * n * LN2)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?",
                    default=os.path.join(HERE, "curby_28297.npz"))
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--shuffles", type=int, default=2000)
    ap.add_argument("--checks", type=int, default=20)
    a = ap.parse_args()

    d = np.load(a.path)
    SA, SB, OA, OB = d['SA'], d['SB'], d['OA'], d['OB']
    n = len(SA)
    n_lags = 2 * a.K + 1
    n_tests = n_lags * 2
    print(f"Παλμός: {a.path}")
    print(f"Trials: {n:,}   lag: {n_lags:,} ανά ζεύγος   "
          f"ΣΥΝΟΛΟ ΥΠΟΘΕΣΕΩΝ: {n_tests:,}\n")

    # ---------------- κατώφλι Bonferroni, αναλυτικά ----------------
    alpha = 0.05
    p_thr = alpha / n_tests
    g_thr = float(chi2.ppf(1 - p_thr, 1))
    print("=" * 78)
    print("ΚΑΤΩΦΛΙ BONFERRONI ΓΙΑ ΤΟΝ ΝΕΟ ΑΡΙΘΜΟ ΥΠΟΘΕΣΕΩΝ")
    print("=" * 78)
    print(f"  υποθέσεις m           = {n_tests:,}   (ήταν 222 στα 111 lag)")
    print(f"  α                     = {alpha}")
    print(f"  ανά υπόθεση p         = α/m = {p_thr:.4e}")
    print(f"  χ²(1) κατώφλι         = {g_thr:.4f}")
    print(f"  -> MI κατώφλι         = {g_thr/(2*n*LN2):.4e} bits/trial  (στο n={n:,})")
    print(f"  (για σύγκριση: το κατώφλι των 222 υποθέσεων ήταν 7,0e-07)\n")

    rng_chk = np.random.default_rng(12345)
    out = {"n": n, "K": a.K, "n_lags": n_lags, "n_tests": n_tests,
           "alpha": alpha, "p_per_test": p_thr, "chi2_threshold": g_thr,
           "pairs": {}}

    for label, o_full, s_full in [("OA vs SB", OA, SB), ("OB vs SA", OB, SA)]:
        print("=" * 78)
        print(label)
        print("=" * 78)

        # --- 1. επαλήθευση της χ²(1) βαθμονόμησης πάνω σε πραγματικό null ---
        print(f"  επαλήθευση χ²(1) με {a.shuffles} ανακατέματα…", flush=True)
        v = validate_chi2(o_full, s_full, a.shuffles)
        print(f"    μέσος G   {v['mean']:.4f}  (θεωρία 1)     "
              f"διασπορά {v['var']:.4f}  (θεωρία 2)")
        print(f"    διάμεσος  {v['q50']:.4f} / {v['q50_theory']:.4f}    "
              f"q90 {v['q90']:.3f} / {v['q90_theory']:.3f}    "
              f"q99 {v['q99']:.3f} / {v['q99_theory']:.3f}")
        print(f"    q99,9     {v['q999']:.3f} / {v['q999_theory']:.3f}    "
              f"max {v['max']:.3f} / {v['max_theory']:.3f} (αναμ.)")
        print(f"    KS: D = {v['ks_stat']:.4f}, p = {v['ks_p']:.3f}  -> "
              + ("ΤΑΙΡΙΑΖΕΙ" if v['ks_p'] > 0.01 else "ΔΕΝ ΤΑΙΡΙΑΖΕΙ"))
        if v['ks_p'] <= 0.01:
            raise SystemExit("Η χ²(1) βαθμονόμηση ΑΠΕΡΡΙΦΘΗ — μη συνεχίσεις.")

        # --- 2. πλήρης σάρωση με FFT ---
        print(f"  σάρωση {n_lags:,} lag με FFT…", flush=True)
        s1 = (s_full == 1).astype(np.int8)
        n11, ferr = n11_all_lags(o_full, s1, a.K)
        ks, A1, B1, nk = margins(o_full, s1, a.K)
        print(f"    μέγιστο σφάλμα στρογγυλοποίησης FFT: {ferr:.2e}  "
              + ("(ασφαλές)" if ferr < 0.1 else "(ΥΨΗΛΟ)"))

        # --- 3. επαλήθευση FFT έναντι απευθείας υπολογισμού ---
        picks = np.unique(np.concatenate([
            [-a.K, -1, 0, 1, a.K],
            rng_chk.integers(-a.K, a.K + 1, a.checks)]))
        bad = 0
        for k in picks:
            oo, ss = align(o_full, s_full, int(k))
            direct = int(((oo == 1) & (ss == 1)).sum())
            j = int(k) + a.K
            if direct != n11[j] or len(oo) != nk[j] or int((oo == 1).sum()) != A1[j] \
                    or int((ss == 1).sum()) != B1[j]:
                bad += 1
                print(f"    ΔΙΑΦΩΝΙΑ στο lag {k}: FFT {n11[j]} vs απευθείας {direct}")
        print(f"    επαλήθευση σε {len(picks)} lag: "
              + ("ΟΛΑ ΤΑΥΤΙΖΟΝΤΑΙ" if bad == 0 else f"{bad} ΔΙΑΦΩΝΙΕΣ"))
        if bad:
            raise SystemExit("Η FFT διαφώνησε με τον απευθείας υπολογισμό.")

        # --- 4. MI, G, p ---
        MI = mi_from_counts(n11, A1, B1, nk)
        G = 2 * nk * LN2 * MI
        above = G > g_thr
        i_max = int(np.argmax(G))
        # σ ως προς το εμπειρικό null, για συνέχεια με την προηγούμενη αναφορά
        sig = (MI - v["null_mi_mean"]) / v["null_mi_sd"]

        print(f"\n    ΜΕΓΙΣΤΟ σε {n_lags:,} lag:")
        print(f"      lag             = {int(ks[i_max]):+,}")
        print(f"      MI              = {MI[i_max]:.4e} bits/trial")
        print(f"      G               = {G[i_max]:.3f}   (κατώφλι {g_thr:.3f})")
        print(f"      p (χ²(1))       = {chi2.sf(G[i_max], 1):.3e}   "
              f"(κατώφλι {p_thr:.3e})")
        print(f"      σ πάνω από null = {sig[i_max]:+.2f}")
        print(f"      -> {'ΠΑΝΩ ΑΠΟ ΤΟ ΚΑΤΩΦΛΙ' if above[i_max] else 'ΚΑΤΩ ΑΠΟ ΤΟ ΚΑΤΩΦΛΙ'}"
              f"  ({MI[i_max]/(g_thr/(2*n*LN2))*100:.0f}% του κατωφλιού)")
        print(f"    lag πάνω από το κατώφλι: {int(above.sum())} / {n_lags:,}")
        if above.any():
            hit = [(int(ks[i]), float(MI[i]), float(G[i])) for i in np.where(above)[0]]
            print(f"      -> {hit[:20]}")
        # πόσα θα περιμέναμε από τύχη σε αυτό το κατώφλι
        print(f"    (αναμενόμενα από τύχη σε όλο το πείραμα: "
              f"{n_tests*p_thr:.3f} = α = {alpha})")

        # πού πέφτουν τα 111 παλιά lag μέσα στο νέο πλήρες σύνολο
        old_lags = list(range(-50, 51)) + [-10000, -3000, -1000, -300, -100,
                                           100, 300, 1000, 3000, 10000]
        oi = np.array([int(k) + a.K for k in old_lags if abs(k) <= a.K])
        i_old = oi[int(np.argmax(MI[oi]))]
        rank = int((MI > MI[i_old]).sum()) + 1
        print(f"    παλιό μέγιστο (στα 111 ελεγμένα lag): MI = {MI[i_old]:.4e} "
              f"στο lag {int(ks[i_old]):+,}  -> κατάταξη {rank} μέσα στα {n_lags:,}")

        out["pairs"][label] = {
            "chi2_validation": v,
            "fft_round_error": ferr,
            "max_lag": int(ks[i_max]), "max_mi": float(MI[i_max]),
            "max_G": float(G[i_max]), "max_p": float(chi2.sf(G[i_max], 1)),
            "max_sigma_vs_null": float(sig[i_max]),
            "n_above_threshold": int(above.sum()),
            "mi_threshold": float(g_thr / (2 * n * LN2)),
            "top20": [{"lag": int(ks[i]), "mi": float(MI[i]), "G": float(G[i]),
                       "p": float(chi2.sf(G[i], 1))}
                      for i in np.argsort(G)[::-1][:20]],
        }
        np.savez_compressed(
            os.path.join(HERE, f"full_scan_{label.replace(' ', '_')}.npz"),
            lags=ks, mi=MI, G=G)
        print()

    mx = max(p["max_mi"] for p in out["pairs"].values())
    print("=" * 78)
    print("ΤΟ ΟΡΙΟ ΓΙΑ ΚΑΘΕ |k| <= 10.000")
    print("=" * 78)
    print(f"  μέγιστο παρατηρούμενο MI σε {n_tests:,} υποθέσεις: {mx:.4e} bits/trial")
    print(f"  κατώφλι Bonferroni:                              "
          f"{g_thr/(2*n*LN2):.4e} bits/trial")
    print(f"  -> MI(αποτέλεσμα ; ρύθμιση σε lag k) < "
          f"{g_thr/(2*n*LN2):.2e} bits/trial για ΚΑΘΕ |k| <= {a.K:,}")
    out["max_mi_overall"] = mx
    out["mi_threshold"] = g_thr / (2 * n * LN2)
    json.dump(out, open(os.path.join(HERE, "full_scan_results.json"), "w"),
              indent=2)
    print(f"\nΑποθηκεύτηκε: full_scan_results.json  + full_scan_*.npz")


if __name__ == "__main__":
    main()
