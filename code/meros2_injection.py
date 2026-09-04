"""
ΜΕΡΟΣ 2 — ΕΠΑΛΗΘΕΥΣΗ ΤΗΣ ΑΝΑΛΥΤΙΚΗΣ ΣΧΕΣΗΣ ΜΕ ΕΝΕΣΗ

Δεν δεχόμαστε το I(k) = C·ε²·exp(-k²/τ²). Το δοκιμάζουμε.

ΤΙ ΕΝΙΕΤΑΙ
    S(i)  = +1 αν SB(i)=2,  -1 αν SB(i)=1      (πραγματικές ρυθμίσεις Bob)
    F(i)  = Σ_k W_τ(k)·S(i+k),   W_τ(k)=exp(-k²/2τ²), k ≠ 0
    λ(i)  = λ0(SA(i)) + α·ε·F(i)
    OA*   ~ Bernoulli(λ)

    λ0(SA) = ο ΠΡΑΓΜΑΤΙΚΟΣ ρυθμός click του 28297 ανά ρύθμιση Alice
             (0,004967 για SA=1 · 0,008891 για SA=2).
    -> Κρατάμε την υπαρκτή lag-0 σύνδεση Alice ως υπόβαθρο, δεν την
       εξαφανίζουμε. Αν αυτή μόλυνε τη μέτρηση του διασταυρούμενου
       καναλιού, θα φαινόταν εδώ.
    -> Οι ΡΥΘΜΙΣΕΙΣ (SA, SB) μένουν ΟΙ ΠΡΑΓΜΑΤΙΚΕΣ. Δεν παράγονται
       συνθετικά, αλλιώς η επαλήθευση θα ήταν κενή (θα δοκιμάζαμε τον
       γεννήτορά μας, όχι τα δεδομένα).

ΤΙ ΜΕΤΡΙΕΤΑΙ, ΓΙΑ ΚΑΘΕ lag k στο [-K, +K] με ένα ζεύγος FFT
    δ_meas(k) = [rate(OA*|SB(i+k)=2) - rate(OA*|SB(i+k)=1)] / 2
    I_meas(k) = MI του 2x2, μείον το baseline του null (df/(2n ln2))

ΤΙ ΠΡΟΒΛΕΠΕΤΑΙ
    δ_pred(k) = α·ε·exp(-k²/2τ²)          -> γκαουσιανή πλάτους τ
    I_pred(k) = C·ε²·exp(-k²/τ²)          -> γκαουσιανή πλάτους τ/√2
Και τα δύο πλάτη ελέγχονται με προσαρμογή γκαουσιανής, χωριστά.

CLIPPING (ΥΠΟΧΡΕΩΤΙΚΗ ΑΝΑΦΟΡΑ)
    Το clip του λ στο [0,1] είναι σιωπηλή μη-γραμμικότητα: δίνει σωστό
    ΣΧΗΜΑ αλλά λάθος ΠΛΑΤΟΣ, και μοιάζει με αποτυχία της παραγωγής.
    Αναφέρεται το ποσοστό clipped trials σε κάθε (ε,τ).
    Πάνω από 0,1% -> το σημείο σημειώνεται ΜΗ ΕΓΚΥΡΟ.
    Τα ε επιλέγονται αυτόματα ώστε να μένουν κάτω από αυτό το όριο.

ΕΛΕΓΧΟΣ ε=0
    Πρώτο σημείο κάθε τ. ΠΡΕΠΕΙ να δώσει επίπεδη καμπύλη στο null.
    Αν δώσει καμπάνα, το bug είναι στην ένεση και σταματάμε.
"""
import argparse, json, math, os, sys
import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2.0)
CLIP_LIMIT = 1e-3          # 0,1% -> πάνω από αυτό: μη έγκυρο σημείο


# --------------------------------------------------------------- το πεδίο F
def build_F(S, tau):
    """F(i) = Σ_k exp(-k²/2τ²) · S(i+k), με FFT συνέλιξη (γραμμική, χωρίς
    αναδίπλωση). Το άθροισμα κόβεται στα ±5τ (W < 3,7e-6)."""
    half = max(1, int(math.ceil(5 * tau)))
    k = np.arange(-half, half + 1, dtype=np.float64)
    W = np.exp(-k ** 2 / (2.0 * tau ** 2))
    W[half] = 0.0                       # k = 0 εκτός μοντέλου (2026-09-04)
    n = len(S)
    L = next_fast_len(n + 2 * half + 1)
    Sf = rfft(S.astype(np.float64), L)
    # F(i) = Σ_j W(j) S(i+j) = συσχέτιση -> συζυγής του πυρήνα
    Wp = np.zeros(L)
    Wp[:len(W)] = W
    Ff = irfft(Sf * np.conjugate(rfft(Wp, L)), L)
    # Wp[m] = W(m-half) -> corr[t] = Σ_m Wp[m]·S[t+m] = F(t+half)
    # άρα F(i) = corr[i-half]  ->  ολίσθηση κατά +half
    F = np.roll(Ff, half)[:n]
    return F, float(W.sum()), float((W ** 2).sum()), half


def check_F(S, F, tau, half, rng, n_check=5):
    """Απευθείας επαλήθευση του F σε λίγες θέσεις."""
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


# --------------------------------------------------- μετρήσεις σε όλα τα lag
def scan(o, s1, K):
    """N11(k)=Σ o[i]·s1[i+k] για k∈[-K,K] με ένα ζεύγος FFT (ίδια λογική με
    το full_scan.py), μαζί με τα περιθώρια."""
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
    """MI (bits) και δ(k) = [rate(o|s=2) - rate(o|s=1)]/2 από τον 2x2.
    s1 = δείκτης «ρύθμιση == 1», άρα n11 = #(o=1 & s=1)."""
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
    """y = A·exp(-k²/2σ²), χωρίς offset (το baseline έχει αφαιρεθεί)."""
    def f(k, A, s):
        return A * np.exp(-k ** 2 / (2.0 * s ** 2))
    try:
        p, cov = curve_fit(f, ks.astype(float), y, p0=[amp0, sigma0],
                           maxfev=20000)
        err = np.sqrt(np.diag(cov))
        return float(p[0]), float(p[1]), float(err[0]), float(err[1])
    except Exception as e:
        return math.nan, math.nan, math.nan, math.nan


# ------------------------------------------------------------------- κυρίως
def run_point(eps, tau, F, lam0, SB1, alpha, C, K, ks_pred, rng, n):
    lam = lam0 + alpha * eps * F
    n_lo = int(np.count_nonzero(lam < 0.0))
    n_hi = int(np.count_nonzero(lam > 1.0))
    frac_clip = (n_lo + n_hi) / n
    lam_c = np.clip(lam, 0.0, 1.0)

    O = (rng.random(n) < lam_c).astype(np.int8)

    ks, n11, A1, B1, nk, ferr = scan(O, SB1, K)
    MI, delta = mi_and_delta(n11, A1, B1, nk)
    baseline = 1.0 / (2.0 * n * LN2)          # E[MI] υπό H0, df=1
    MIc = MI - baseline

    # προβλέψεις
    nz = ks != 0                                   # k = 0 εκτός μοντέλου
    d_pred = alpha * eps * np.exp(-ks.astype(float) ** 2 / (2 * tau ** 2)) * nz
    I_pred = C * eps ** 2 * np.exp(-ks.astype(float) ** 2 / tau ** 2) * nz

    res = dict(eps=eps, tau=tau, n_clip_low=n_lo, n_clip_high=n_hi,
               frac_clipped=frac_clip, valid=bool(frac_clip <= CLIP_LIMIT),
               fft_round_error=ferr, click_rate=float(O.mean()),
               baseline_mi=baseline,
               # κορυφή του πυρήνα: k = +1 (το k = 0 έχει W = 0)
               delta_meas_1=float(delta[K + 1]), delta_pred_1=float(d_pred[K + 1]),
               mi_meas_1=float(MIc[K + 1]), mi_pred_1=float(I_pred[K + 1]),
               delta_meas_0=float(delta[K]), mi_meas_0=float(MIc[K]))

    if eps == 0.0:
        # έλεγχος επιπεδότητας: καμία καμπάνα δεν επιτρέπεται
        sd_null = math.sqrt(2.0) / (2.0 * n * LN2)
        res["null_mi_mean"] = float(MIc.mean())
        res["null_mi_sd"] = float(MIc.std(ddof=1))
        res["null_mi_sd_theory"] = sd_null
        res["null_mi_max"] = float(MIc.max())
        res["null_delta_sd"] = float(delta.std(ddof=1))
        # πλάτος καμπάνας που θα ΕΠΡΕΠΕ να μη βρεθεί:
        A, s, dA, ds = gauss_fit(ks[nz], MIc[nz], max(tau, 1.0), MIc.max())
        res["null_fit_amp"] = A
        res["null_fit_sigma"] = s
        # σύγκριση κέντρου με ουρές
        core = np.abs(ks) <= max(1, int(tau))
        res["null_core_mean"] = float(MIc[core].mean())
        res["null_tail_mean"] = float(MIc[~core].mean())
        res["null_core_n"] = int(core.sum())
        res["null_tail_n"] = int((~core).sum())
        # σφάλμα της διαφοράς κέντρου-ουρών (οι εκτιμητές MI σε διαφορετικά
        # lag είναι πρακτικά ανεξάρτητοι — τυχαίες ρυθμίσεις)
        res["null_diff_se"] = float(res["null_mi_sd"] * math.sqrt(
            1.0 / core.sum() + 1.0 / max(1, (~core).sum())))
        return res, ks, MIc, delta, I_pred, d_pred

    # ---- προσαρμογή γκαουσιανής στο δ(k): αναμενόμενο σ = τ
    # (μόνο στα k ≠ 0: το k = 0 δεν ανήκει στο μοντέλο)
    Ad, sd_, eAd, esd = gauss_fit(ks[nz], delta[nz], tau, alpha * eps)
    # ---- και στο I(k): αναμενόμενο σ = τ/√2
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

    S = np.where(SB == 2, 1.0, -1.0)          # ±1 κωδικοποίηση Bob
    SB1 = (SB == 1).astype(np.int8)
    lam0 = np.where(SA == 1, r_by_setting[1], r_by_setting[2])

    rng = np.random.default_rng(a.seed)

    print("=" * 78)
    print("ΜΕΡΟΣ 2 — ΕΝΕΣΗ ΜΕ ΓΝΩΣΤΑ (ε, τ)")
    print("=" * 78)
    print(f"  n = {n:,}   α = {alpha:.6e}   C = {C:.6e}")
    print(f"  λ0: SA=1 -> {r_by_setting[1]:.6e} · SA=2 -> {r_by_setting[2]:.6e}")
    print(f"  όριο clipping: {CLIP_LIMIT*100:.1f}% των trials\n")

    all_res = {"alpha": alpha, "C": C, "n": n, "seed": a.seed,
               "clip_limit": CLIP_LIMIT, "points": []}
    curves = {}

    for tau in a.taus:
        print("=" * 78)
        print(f"τ = {tau:g}")
        print("=" * 78)
        F, Wsum, W2sum, half = build_F(S, tau)
        werr = check_F(S, F, tau, half, rng)
        sdF = float(F.std())
        print(f"  πυρήνας: ΣW = {Wsum:.4f}  ΣW² = {W2sum:.4f}  "
              f"(θεωρία ΣW²≈τ√π = {tau*math.sqrt(math.pi):.4f})")
        print(f"  sd(F) = {sdF:.4f}  (θεωρία √ΣW² = {math.sqrt(W2sum):.4f})")
        print(f"  επαλήθευση F σε 5 θέσεις: μέγιστη διαφορά {werr:.2e}")
        if werr > 1e-6:
            sys.exit("Η FFT συνέλιξη του F διαφώνησε με τον απευθείας υπολογισμό.")

        # --- επιλογή ε ώστε clipping < 0,1% ---
        # το κάτω σκέλος λ0 = r1 είναι το κρίσιμο
        q = np.quantile(F, CLIP_LIMIT / 2)     # αρνητικό άκρο
        eps_max = r_by_setting[1] / (alpha * abs(q))
        eps_list = [0.0, eps_max / 4, eps_max / 2, eps_max]
        print(f"  ε_max (από clipping) = {eps_max:.4g}   "
              f"δοκιμές: {['%.4g' % e for e in eps_list[1:]]}")

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
                print(f"\n  --- ΕΛΕΓΧΟΣ ε = 0 (πρέπει: ΕΠΙΠΕΔΗ στο null) ---")
                print(f"    clipped: {res['frac_clipped']*100:.4f}%")
                print(f"    MI μέσος {res['null_mi_mean']:+.3e}  "
                      f"sd {res['null_mi_sd']:.3e}  "
                      f"(θεωρία sd {res['null_mi_sd_theory']:.3e})")
                dif = res['null_core_mean'] - res['null_tail_mean']
                print(f"    MI κέντρο(|k|<=τ, {res['null_core_n']}) "
                      f"{res['null_core_mean']:+.3e}  vs  ουρές "
                      f"({res['null_tail_n']}) {res['null_tail_mean']:+.3e}")
                print(f"    διαφορά {dif:+.3e} ± {res['null_diff_se']:.3e}  "
                      f"-> {dif/res['null_diff_se']:+.2f}σ")
                print(f"    προσαρμογή καμπάνας: A = {res['null_fit_amp']:.3e} "
                      f"(θα έπρεπε ~0, σε σύγκριση με sd {res['null_mi_sd']:.1e})")
                flat = abs(dif) < 3 * res['null_diff_se']
                print(f"    ΕΠΙΠΕΔΗ; {'ΝΑΙ' if flat else 'ΟΧΙ — BUG ΣΤΗΝ ΕΝΕΣΗ'}")
                res["flat"] = bool(flat)
                if not flat:
                    sys.exit("Ο έλεγχος ε=0 απέτυχε: η ένεση παράγει σήμα από το "
                             "τίποτα. Σταματάμε πριν ερμηνεύσουμε οτιδήποτε.")
                continue

            mark = "" if res["valid"] else "   *** ΜΗ ΕΓΚΥΡΟ (clipping) ***"
            print(f"\n  --- ε = {eps:.4g} ---{mark}")
            print(f"    clipped: {res['frac_clipped']*100:.4f}% "
                  f"({res['n_clip_low']:,} κάτω, {res['n_clip_high']:,} πάνω)"
                  f"   [όριο {CLIP_LIMIT*100:.1f}%]")
            print(f"    δ(+1): μετρ {res['delta_meas_1']:.4e}  "
                  f"προβλ {res['delta_pred_1']:.4e}  "
                  f"λόγος {res['ratio_delta1']:.3f}   "
                  f"[δ(0) = {res['delta_meas_0']:+.2e}, εκτός μοντέλου]")
            print(f"    I(+1): μετρ {res['mi_meas_1']:.4e}  "
                  f"προβλ {res['mi_pred_1']:.4e}  "
                  f"λόγος {res['ratio_mi1']:.3f}")
            print(f"    fit δ(k): A = {res['fit_delta_amp']:.4e} "
                  f"(λόγος {res['ratio_delta_amp']:.3f})   "
                  f"σ = {res['fit_delta_sigma']:.3f} ± {res['fit_delta_sigma_err']:.3f}"
                  f"   αναμ. τ = {tau:g}  -> λόγος {res['ratio_delta_sigma']:.3f}")
            print(f"    fit I(k): A = {res['fit_mi_amp']:.4e} "
                  f"(λόγος {res['ratio_mi_amp']:.3f})   "
                  f"σ = {res['fit_mi_sigma']:.3f} ± {res['fit_mi_sigma_err']:.3f}"
                  f"   αναμ. τ/√2 = {tau/math.sqrt(2):.3f}  -> "
                  f"λόγος {res['ratio_mi_sigma']:.3f}")
        print()
        del F

    json.dump(all_res, open(os.path.join(HERE, a.out + ".json"), "w"), indent=2)
    np.savez_compressed(os.path.join(HERE, a.out + "_curves.npz"),
                        **{k: np.array(v["mi"]) for k, v in curves.items()},
                        **{k + "_delta": np.array(v["delta"]) for k, v in curves.items()},
                        **{k + "_lags": np.array(v["lags"]) for k, v in curves.items()})

    # ---------------------------------------------------------- σύνοψη
    print("=" * 78)
    print("ΣΥΝΟΨΗ — ΜΟΝΟ ΕΓΚΥΡΑ ΣΗΜΕΙΑ (clipping <= 0,1%)")
    print("=" * 78)
    print(f"{'τ':>7} {'ε':>10} {'clip%':>7} {'I(+1) λόγος':>11} "
          f"{'σ_δ/τ':>8} {'σ_I/(τ/√2)':>11} {'έγκυρο':>7}")
    for r in all_res["points"]:
        if r["eps"] == 0.0:
            continue
        print(f"{r['tau']:>7g} {r['eps']:>10.4g} {r['frac_clipped']*100:>7.4f} "
              f"{r['ratio_mi1']:>11.3f} {r['ratio_delta_sigma']:>8.3f} "
              f"{r['ratio_mi_sigma']:>11.3f} {'ναι' if r['valid'] else 'ΟΧΙ':>7}")
    print(f"\nΑποθηκεύτηκε: {a.out}.json + {a.out}_curves.npz")


if __name__ == "__main__":
    main()
