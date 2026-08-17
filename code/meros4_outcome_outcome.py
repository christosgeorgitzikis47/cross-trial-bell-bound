"""
ΜΕΡΟΣ 4 — ΤΕΣΤ Β: ΣΥΣΧΕΤΙΣΗ ΑΠΟΤΕΛΕΣΜΑΤΟΣ-ΑΠΟΤΕΛΕΣΜΑΤΟΣ

Ξεχωριστό ερώτημα από τα μέρη 1-3. Αν η ΑΚΟΛΟΥΘΙΑ είναι χρονικά
ανακατεμένη (λάθος ζευγάρωμα Alice-Bob στην καταγραφή), τότε το σήμα
δεν εμφανίζεται ως συσχέτιση αποτελέσματος-ρύθμισης αλλά ως συσχέτιση
αποτελέσματος-ΑΠΟΤΕΛΕΣΜΑΤΟΣ σε lag k ≠ 0.

    I( OA(i) ; OB(i+k) )  για κάθε |k| <= 10.000, με FFT.

ΘΕΤΙΚΟΣ ΕΛΕΓΧΟΣ: στο k=0 ΠΡΕΠΕΙ να ανάβει έντονα (συσχέτιση Bell).
Αν δεν ανάβει -> bug στη φόρτωση, σταματάμε.

Ίδιο σχήμα με το full_scan.py: αναλυτικό κατώφλι χ²(1), ΕΠΑΛΗΘΕΥΜΕΝΟ με
πραγματικά ανακατέματα πριν χρησιμοποιηθεί. Bonferroni για 20.001
υποθέσεις (ΕΝΑ ζεύγος εδώ, όχι δύο).
"""
import argparse, json, math, os
import numpy as np
from scipy.stats import chi2, kstest

from lag_test import mi
from meros2_injection import scan, mi_and_delta

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2.0)


def validate_chi2(oa, ob, n_shuffle, seed=999):
    """G = 2n ln2 MI των ανακατεμένων ~ χ²(1);  ανακατεύουμε το OB."""
    rng = np.random.default_rng(seed)
    sh = ob.copy()
    n = len(oa)
    g = np.empty(n_shuffle)
    for i in range(n_shuffle):
        rng.shuffle(sh)
        # mi() θέλει «ρύθμιση» σε {1,2} -> μετατροπή του OB σε {1,2}
        g[i] = 2 * n * LN2 * mi(oa, sh + 1)
    ks = kstest(g, 'chi2', args=(1,))
    return dict(n_shuffle=n_shuffle, mean=float(g.mean()),
                var=float(g.var(ddof=1)), ks_stat=float(ks.statistic),
                ks_p=float(ks.pvalue),
                q50=float(np.percentile(g, 50)), q50_th=float(chi2.ppf(.50, 1)),
                q90=float(np.percentile(g, 90)), q90_th=float(chi2.ppf(.90, 1)),
                q99=float(np.percentile(g, 99)), q99_th=float(chi2.ppf(.99, 1)),
                q999=float(np.percentile(g, 99.9)), q999_th=float(chi2.ppf(.999, 1)),
                max=float(g.max()),
                null_mi_mean=float(g.mean() / (2 * n * LN2)),
                null_mi_sd=float(g.std(ddof=1) / (2 * n * LN2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10_000)
    ap.add_argument("--shuffles", type=int, default=2000)
    a = ap.parse_args()

    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    OA, OB = d['OA'], d['OB']
    n = len(OA)
    n_lags = 2 * a.K + 1
    n_tests = n_lags                      # ΕΝΑ ζεύγος

    p_thr = 0.05 / n_tests
    g_thr = float(chi2.ppf(1 - p_thr, 1))
    mi_thr = g_thr / (2 * n * LN2)

    print("=" * 78)
    print("ΜΕΡΟΣ 4 — I( OA(i) ; OB(i+k) ),  |k| <= 10.000")
    print("=" * 78)
    print(f"  n = {n:,}   click A {OA.mean()*100:.4f}%  B {OB.mean()*100:.4f}%")
    print(f"  υποθέσεις m = {n_tests:,}   p = {p_thr:.4e}   "
          f"χ²(1) κατώφλι {g_thr:.3f}")
    print(f"  -> κατώφλι MI = {mi_thr:.4e} bits/trial\n")

    print(f"  επαλήθευση χ²(1) με {a.shuffles} ανακατέματα του OB…", flush=True)
    v = validate_chi2(OA, OB, a.shuffles)
    print(f"    μέσος G {v['mean']:.4f} (θεωρία 1)   "
          f"διασπορά {v['var']:.4f} (θεωρία 2)")
    print(f"    q50 {v['q50']:.3f}/{v['q50_th']:.3f}  "
          f"q90 {v['q90']:.3f}/{v['q90_th']:.3f}  "
          f"q99 {v['q99']:.3f}/{v['q99_th']:.3f}  "
          f"q99,9 {v['q999']:.3f}/{v['q999_th']:.3f}")
    print(f"    KS: D = {v['ks_stat']:.4f}, p = {v['ks_p']:.3f}  -> "
          + ("ΤΑΙΡΙΑΖΕΙ" if v['ks_p'] > 0.01 else "ΔΕΝ ΤΑΙΡΙΑΖΕΙ"))
    if v['ks_p'] <= 0.01:
        raise SystemExit("Η χ²(1) βαθμονόμηση ΑΠΕΡΡΙΦΘΗ.")

    print(f"\n  σάρωση {n_lags:,} lag με FFT…", flush=True)
    ks, n11, A1, B1, nk, ferr = scan(OA, OB.astype(np.int8), a.K)
    MI, _ = mi_and_delta(n11, A1, B1, nk)
    G = 2 * nk * LN2 * MI
    print(f"    σφάλμα στρογγυλοποίησης FFT: {ferr:.2e}")

    # --- επαλήθευση FFT με απευθείας υπολογισμό ---
    rng = np.random.default_rng(31337)
    picks = np.unique(np.concatenate([[-a.K, -1, 0, 1, a.K],
                                      rng.integers(-a.K, a.K + 1, 20)]))
    bad = 0
    for k in picks:
        kk = int(k)
        if kk > 0:
            oa, ob = OA[:-kk], OB[kk:]
        elif kk < 0:
            oa, ob = OA[-kk:], OB[:kk]
        else:
            oa, ob = OA, OB
        direct = int(((oa == 1) & (ob == 1)).sum())
        j = kk + a.K
        if direct != n11[j] or len(oa) != nk[j]:
            bad += 1
            print(f"    ΔΙΑΦΩΝΙΑ lag {kk}: FFT {n11[j]} vs {direct}")
    print(f"    επαλήθευση σε {len(picks)} lag: "
          + ("ΟΛΑ ΤΑΥΤΙΖΟΝΤΑΙ" if bad == 0 else f"{bad} ΔΙΑΦΩΝΙΕΣ"))
    if bad:
        raise SystemExit("Η FFT διαφώνησε.")

    # ---------------- ΘΕΤΙΚΟΣ ΕΛΕΓΧΟΣ ----------------
    j0 = a.K
    mi0, G0 = float(MI[j0]), float(G[j0])
    p0 = float(chi2.sf(G0, 1))
    print("\n" + "=" * 78)
    print("ΘΕΤΙΚΟΣ ΕΛΕΓΧΟΣ — k = 0")
    print("=" * 78)
    print(f"  MI(OA;OB) στο k=0 = {mi0:.6e} bits/trial")
    print(f"  G = {G0:,.1f}   ({G0/g_thr:,.0f}× το κατώφλι)")
    print(f"  p = {p0:.3e}   σ ισοδύναμο ≈ {math.sqrt(G0):.1f}")
    print(f"  πίνακας 2x2 στο k=0: N11 = {n11[j0]:,}  "
          f"N(OA=1) = {A1[j0]:,}  N(OB=1) = {B1[j0]:,}")
    exp11 = A1[j0] * B1[j0] / nk[j0]
    print(f"  αναμενόμενα N11 υπό ανεξαρτησία = {exp11:,.0f}  "
          f"-> λόγος {n11[j0]/exp11:.3f}")
    lights = G0 > g_thr
    print(f"  ΑΝΑΒΕΙ; {'ΝΑΙ' if lights else 'ΟΧΙ — BUG, ΣΤΑΜΑΤΑΜΕ'}")
    if not lights:
        raise SystemExit("Ο θετικός έλεγχος k=0 απέτυχε: bug στη φόρτωση.")

    # ---------------- το ερώτημα: k ≠ 0 ----------------
    mask = ks != 0
    MInz, Gnz, ksnz = MI[mask], G[mask], ks[mask]
    i = int(np.argmax(Gnz))
    above = Gnz > g_thr
    sig = (MInz - v["null_mi_mean"]) / v["null_mi_sd"]

    print("\n" + "=" * 78)
    print("ΤΟ ΕΡΩΤΗΜΑ — k ≠ 0")
    print("=" * 78)
    print(f"  μέγιστο MI = {MInz[i]:.4e} bits/trial  στο lag {int(ksnz[i]):+,}")
    print(f"  G = {Gnz[i]:.3f}   (κατώφλι {g_thr:.3f})  "
          f"-> {Gnz[i]/g_thr*100:.0f}% του κατωφλιού")
    print(f"  p = {chi2.sf(Gnz[i],1):.3e}   (κατώφλι {p_thr:.3e})")
    print(f"  σ πάνω από το εμπειρικό null = {sig[i]:+.2f}")
    print(f"  lag πάνω από το κατώφλι: {int(above.sum())} / {len(ksnz):,}")
    if above.any():
        print(f"    -> {[(int(ksnz[j]), float(MInz[j])) for j in np.where(above)[0][:20]]}")
    print(f"\n  ±1 (deadtime, αναμενόμενο):")
    for k in (-2, -1, 1, 2):
        j = k + a.K
        print(f"    lag {k:+d}: MI = {MI[j]:.4e}  G = {G[j]:.2f}  "
              f"({'πάνω' if G[j] > g_thr else 'κάτω'} από το κατώφλι)")

    top = np.argsort(Gnz)[::-1][:20]
    print(f"\n  κορυφαία 10 σε k≠0:")
    for j in top[:10]:
        print(f"    lag {int(ksnz[j]):+7,}  MI = {MInz[j]:.4e}  "
              f"G = {Gnz[j]:6.2f}  σ = {sig[j]:+.2f}")

    print("\n" + "=" * 78)
    print(f"  ΟΡΙΟ: I(OA(i);OB(i+k)) < {mi_thr:.3e} bits/trial για κάθε "
          f"0 < |k| <= {a.K:,}")
    print(f"  Κλίμακα: το k=0 είναι {mi0:.3e} -> ο λόγος είναι "
          f"1 / {mi0/mi_thr:,.0f}")
    print("=" * 78)

    out = dict(n=n, K=a.K, n_tests=n_tests, p_thr=p_thr, g_thr=g_thr,
               mi_threshold=mi_thr, chi2_validation=v, fft_round_error=ferr,
               k0=dict(mi=mi0, G=G0, p=p0, n11=int(n11[j0]),
                       expected_n11=float(exp11),
                       ratio=float(n11[j0] / exp11), lights=bool(lights)),
               max_nonzero=dict(lag=int(ksnz[i]), mi=float(MInz[i]),
                                G=float(Gnz[i]),
                                p=float(chi2.sf(Gnz[i], 1)),
                                sigma=float(sig[i]),
                                pct_of_threshold=float(Gnz[i] / g_thr * 100)),
               n_above=int(above.sum()),
               top20=[dict(lag=int(ksnz[j]), mi=float(MInz[j]),
                           G=float(Gnz[j])) for j in top],
               deadtime={str(k): dict(mi=float(MI[k + a.K]), G=float(G[k + a.K]))
                         for k in (-2, -1, 1, 2)})
    json.dump(out, open(os.path.join(HERE, "meros4_results.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(HERE, "meros4_OA_OB.npz"),
                        lags=ks, mi=MI, G=G)
    print("\nΑποθηκεύτηκε: meros4_results.json + meros4_OA_OB.npz")


if __name__ == "__main__":
    main()
