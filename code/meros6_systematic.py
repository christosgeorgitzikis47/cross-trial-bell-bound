"""
ΜΕΡΟΣ 6.2 — ΤΟ ΣΥΣΤΗΜΑΤΙΚΟ +3…7% (ένσταση peer review #2)

Η ΥΠΟΨΙΑ (από το review): το σ_T του χάρτη βγαίνει από ανακατέματα των
ΠΡΑΓΜΑΤΙΚΩΝ δεδομένων, που έχουν αυτοσυσχέτιση αποτελεσμάτων (deadtime στο
lag ±1). Τα ΕΝΕΜΕΝΑ αποτελέσματα είναι φρέσκο Bernoulli, χωρίς αυτή την
αυτοσυσχέτιση. Λιγότερος θόρυβος -> μεγαλύτερο z.

ΓΙΑΤΙ ΕΙΝΑΙ ΣΩΣΤΟΣ Ο ΜΗΧΑΝΙΣΜΟΣ (αλγεβρικά, πριν τη μέτρηση)
    Var(T) = ΣΣ W(k)W(k') Cov(δ̂(k), δ̂(k')).
    Με τυχαία μετάθεση των ρυθμίσεων, Cov(s_a, s_b) = p(1−p) μόνο για a = b.
    Ο όρος a = b απαιτεί i+k = j+k', δηλαδή j = i + (k−k'), και δίνει
        Cov(δ̂(k), δ̂(k')) ∝ Σ_i O_i·O_{i+(k−k')}
    = ακριβώς η ΑΥΤΟΣΥΣΧΕΤΙΣΗ ΤΩΝ ΑΠΟΤΕΛΕΣΜΑΤΩΝ στο lag k−k'.
    Το αναλυτικό σ_T υποθέτει ότι αυτοί οι όροι είναι μηδέν. Άρα:
    αυτοσυσχέτιση στο O -> εμπειρικό σ_T > αναλυτικό. Ακριβώς ό,τι βλέπουμε.

ΤΙ ΜΕΤΡΙΕΤΑΙ ΕΔΩ
  (α) αυτοσυσχέτιση του ΠΡΑΓΜΑΤΙΚΟΥ OA/OB σε lag 1…5,
  (β) η ίδια στα ΕΝΕΜΕΝΑ (φρέσκο Bernoulli),
  (γ) σ_T εμπειρικό/αναλυτικό με ανακατέματα των ΕΝΕΜΕΝΩΝ δεδομένων
      — αν ο μηχανισμός είναι σωστός, εδώ ο λόγος πρέπει να πέσει στο 1,
        ενώ στα πραγματικά είναι > 1,
  (δ) η αποσύνθεση του συστηματικού: z_μετρ/z_προβλ έναντι σ_T(χάρτη)/σ_T(αναλυτικό).
"""
import argparse, json, math, os
import numpy as np

from meros2_injection import scan, mi_and_delta
from meros3_map import sigma_delta
from meros5_asym import kernel_W
from meros5_verify import build_F_kernel

HERE = os.path.dirname(os.path.abspath(__file__))


def autocorr_binary(x, lags):
    """Pearson r του δυαδικού x σε δοσμένα lag, με σφάλμα ~1/√n."""
    xf = x.astype(np.float64)
    p = xf.mean(); v = p * (1 - p)
    out = {}
    for L in lags:
        c = float(np.dot(xf[:-L], xf[L:]) / (len(xf) - L)) - p * p
        out[L] = c / v
    return out, 1.0 / math.sqrt(len(x))


def sigma_T_ratio(O, s1, Wmat, K, shuffles, rng):
    """εμπειρικό/αναλυτικό σ_T για δοσμένα φίλτρα, με ανακατέματα ρυθμίσεων."""
    _, n11, A1, B1, nk, _ = scan(O, s1, K)
    sd = sigma_delta(A1, B1, nk)
    sT_ana = np.sqrt(Wmat ** 2 @ sd ** 2)
    sh = s1.copy()
    T = np.empty((shuffles, Wmat.shape[0]))
    for i in range(shuffles):
        rng.shuffle(sh)
        _, n11s, A1s, B1s, nks, _ = scan(O, sh, K)
        _, ds = mi_and_delta(n11s, A1s, B1s, nks)
        T[i] = Wmat @ ds
    sT_emp = T.std(axis=0, ddof=1)
    return sT_emp, sT_ana, sT_emp / sT_ana


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--K", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--cases", nargs="+", default=["future:30", "future:300"])
    ap.add_argument("--out", default="meros6_systematic")
    a = ap.parse_args()

    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    ver = json.load(open(os.path.join(HERE, "meros5_verify.json")))
    cal = json.load(open(os.path.join(HERE, "meros1_alpha.json")))["OA vs SA"]
    alpha, r1, r2 = cal["alpha"], cal["r1"], cal["r2"]
    taus = np.array(m5["taus"])
    P = m5["pairs"]["OA vs SB"]

    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    SA, SB, OA, OB = d['SA'], d['SB'], d['OA'], d['OB']
    n = len(SA)
    SB1 = (SB == 1).astype(np.int8)
    S = np.where(SB == 2, 1.0, -1.0)
    lam0 = np.where(SA == 1, r1, r2)
    rng = np.random.default_rng(a.seed)
    kax = np.arange(-a.K, a.K + 1, dtype=np.float64)

    print("=" * 78)
    print("ΜΕΡΟΣ 6.2 — ΑΠΟ ΠΟΥ ΕΡΧΕΤΑΙ ΤΟ ΣΥΣΤΗΜΑΤΙΚΟ")
    print("=" * 78)

    # ---------- (α) αυτοσυσχέτιση των πραγματικών αποτελεσμάτων ----------
    lags = [1, 2, 3, 4, 5]
    acA, se = autocorr_binary(OA, lags)
    acB, _ = autocorr_binary(OB, lags)
    print(f"\n(α) Αυτοσυσχέτιση ΠΡΑΓΜΑΤΙΚΩΝ αποτελεσμάτων (σφάλμα ±{se:.2e}):")
    print(f"    {'lag':>5} {'OA':>12} {'σ':>7} {'OB':>12} {'σ':>7}")
    for L in lags:
        print(f"    {L:>5} {acA[L]:>12.3e} {acA[L]/se:>7.1f} "
              f"{acB[L]:>12.3e} {acB[L]/se:>7.1f}")
    out = {"autocorr_real_OA": acA, "autocorr_real_OB": acB, "se_autocorr": se,
           "cases": {}}

    # ---------- (β)+(γ) ανά περίπτωση ----------
    for case in a.cases:
        kn, tv = case.split(":")
        tau = float(tv)
        j = int(np.argmin(np.abs(taus - tau)))
        eps = P[kn]["eps_excl"][j]
        Wmat = kernel_W(kn, kax, tau)[None, :]

        print("\n" + "-" * 78)
        print(f"ΠΕΡΙΠΤΩΣΗ {kn}, τ = {tau:g}   (ένεση στο ε_excl = {eps:.4e})")
        print("-" * 78)

        F, half = build_F_kernel(S, kn, tau, a.K)
        lam = lam0 + alpha * eps * F
        Oinj = (rng.random(n) < np.clip(lam, 0, 1)).astype(np.int8)
        del F

        aci, _ = autocorr_binary(Oinj, lags)
        print(f"(β) αυτοσυσχέτιση ΕΝΕΜΕΝΩΝ: " +
              "  ".join(f"lag{L} {aci[L]:+.2e} ({aci[L]/se:+.1f}σ)"
                        for L in lags))

        print(f"(γ) σ_T με {a.shuffles} ανακατέματα…", flush=True)
        e_r, a_r, ratio_real = sigma_T_ratio(OA, SB1, Wmat, a.K, a.shuffles, rng)
        e_i, a_i, ratio_inj = sigma_T_ratio(Oinj, SB1, Wmat, a.K, a.shuffles, rng)
        print(f"    ΠΡΑΓΜΑΤΙΚΑ δεδομένα:  εμπειρικό/αναλυτικό = "
              f"{ratio_real[0]:.4f}")
        print(f"    ΕΝΕΜΕΝΑ δεδομένα:     εμπειρικό/αναλυτικό = "
              f"{ratio_inj[0]:.4f}")
        print(f"    (χάρτης, 400 ανακατέματα: "
              f"{P[kn]['sigma_T_emp'][j]/P[kn]['sigma_T_ana'][j]:.4f})")
        se_ratio = 1.0 / math.sqrt(2 * (a.shuffles - 1))
        print(f"    στατιστικό σφάλμα του λόγου: ±{se_ratio:.4f}")
        out["cases"][case] = dict(
            tau=tau, kernel=kn, eps=eps,
            autocorr_injected={str(k): v for k, v in aci.items()},
            ratio_real=float(ratio_real[0]), ratio_inj=float(ratio_inj[0]),
            ratio_map=float(P[kn]['sigma_T_emp'][j] / P[kn]['sigma_T_ana'][j]),
            se_ratio=se_ratio)
        del Oinj

    # ---------- (δ) αποσύνθεση του συστηματικού ----------
    print("\n" + "=" * 78)
    print("(δ) ΑΠΟΣΥΝΘΕΣΗ: z_μετρ/z_προβλ  έναντι  σ_T(χάρτη)/σ_T(αναλυτικό)")
    print("=" * 78)
    print(f"  {'πυρήνας':<11}{'τ':>6}{'×ε':>4}{'z_μ/z_π':>10}"
          f"{'σT_χαρτ/σT_αν':>15}{'υπόλοιπο':>10}{'sem':>8}")
    resid = []
    for p in ver["points"]:
        j = int(np.argmin(np.abs(taus - p["tau"])))
        K = P[p["kernel"]]
        r_map = K["sigma_T"][j] / K["sigma_T_ana"][j]
        r_obs = p["z_mean"] / p["z_pred"]
        rel_sem = p["z_sem"] / p["z_mean"]
        print(f"  {p['kernel']:<11}{p['tau']:>6.0f}{p['frac']:>4.0f}"
              f"{r_obs:>10.4f}{r_map:>15.4f}{r_obs/r_map:>10.4f}"
              f"{rel_sem:>8.3f}")
        resid.append((r_obs / r_map, rel_sem))
    v = np.array([x[0] for x in resid])
    w = np.array([x[1] for x in resid])
    sem_mean = float(np.sqrt((w ** 2).sum()) / len(w))
    print(f"\n  μέσο υπόλοιπο μετά την αφαίρεση του σ_T: "
          f"{v.mean():.4f} ± {sem_mean:.4f}  "
          f"({100*(v.mean()-1):+.1f}% ± {100*sem_mean:.1f}%)")
    print(f"  -> {'ΣΥΜΒΑΤΟ ΜΕ ΤΟ 1' if abs(v.mean()-1) < 2*sem_mean else 'ΟΧΙ συμβατό με το 1'}")
    out["decomposition"] = dict(residual_mean=float(v.mean()),
                                residual_sem=sem_mean,
                                explained_by_sigma_T=True)
    json.dump(out, open(os.path.join(HERE, a.out + ".json"), "w"), indent=2)
    print(f"\nΑποθηκεύτηκε: {a.out}.json")


if __name__ == "__main__":
    main()
