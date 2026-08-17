"""
ΜΕΡΟΣ 5, ΒΗΜΑ 2 — ΕΠΑΛΗΘΕΥΣΗ ΤΩΝ ΑΣΥΜΜΕΤΡΩΝ ΦΙΛΤΡΩΝ ΜΕ ΕΝΕΣΗ

Ίδιο σχήμα με το meros3_verify.py, αλλά ο ΙΔΙΟΣ ασύμμετρος πυρήνας
χρησιμοποιείται και στην ένεση και στο φίλτρο:

    λ(i) = λ₀(SA(i)) + α·ε·F(i),   F(i) = Σ_k W(k)·S_B(i+k)
    W = future / past / exp_future  (γκαουσιανοί ή εκθετικός, μονόπλευροι)

Αν το φίλτρο «κοιτάζει» προς τη λάθος κατεύθυνση, το T θα καταρρεύσει στο
μηδέν — γι' αυτό το τεστ έχει νόημα: ένα σήμα μόνο-μέλλον ΔΕΝ πρέπει να
ανιχνεύεται από το φίλτρο μόνο-παρελθόν (ελέγχεται ρητά, cross-check).

Κριτήριο (ίδιο με #6): E[z] = frac·(z_thr + |z_obs|). Το ε_excl είναι άνω
όριο εμπιστοσύνης, άρα στο frac=1 η ισχύς είναι ~50% ΕΞ ΟΡΙΣΜΟΥ· το
κριτήριο είναι η συμφωνία του μέσου z με την πρόβλεψη, συν πλήρης
ανίχνευση στο frac=2.

2 τιμές τ ανά πυρήνα (30 και 300 — υπάρχουν ΑΚΡΙΒΩΣ στο πλέγμα του χάρτη,
χωρίς παρεμβολή), 2 επίπεδα ε (ε_excl, 2·ε_excl), 10 επαναλήψεις.
"""
import argparse, json, math, os
import numpy as np
from scipy.fft import next_fast_len, rfft, irfft

from meros2_injection import scan, mi_and_delta
from meros3_map import sigma_delta
from meros5_asym import kernel_W, KERNEL_LABEL

HERE = os.path.dirname(os.path.abspath(__file__))


def build_F_kernel(S, name, tau, K):
    """F(i) = Σ_k W(k)·S(i+k) με γραμμική συνέλιξη FFT, για οποιονδήποτε
    πυρήνα. Το άθροισμα κόβεται εκεί που W < 1e-6 (και ποτέ πέρα από ±K,
    ώστε η ένεση να έχει ΤΟ ΙΔΙΟ παράθυρο με το φίλτρο)."""
    if name == "exp_future":
        half = int(math.ceil(14 * tau))
    else:
        half = int(math.ceil(5 * tau))
    half = max(1, min(half, K))
    k = np.arange(-half, half + 1, dtype=np.float64)
    W = kernel_W(name, k, tau)
    n = len(S)
    L = next_fast_len(n + 2 * half + 1)
    Sf = rfft(S.astype(np.float64), L)
    Wp = np.zeros(L); Wp[:len(W)] = W
    Ff = irfft(Sf * np.conjugate(rfft(Wp, L)), L)
    F = np.roll(Ff, half)[:n]
    return F, half


def check_F(S, F, name, tau, half, rng, n_check=4):
    """Απευθείας επαλήθευση του F σε λίγες θέσεις (όχι εμπιστοσύνη στο FFT)."""
    n = len(S)
    k = np.arange(-half, half + 1, dtype=np.float64)
    W = kernel_W(name, k, tau)
    worst = 0.0
    for i in rng.integers(half + 1, n - half - 1, n_check):
        direct = float(np.dot(W, S[i - half:i + half + 1]))
        worst = max(worst, abs(direct - float(F[i])))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+", default=[30.0, 300.0])
    ap.add_argument("--kernels", nargs="+",
                    default=["future", "past", "exp_future"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--K", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=909)
    a = ap.parse_args()

    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    cal = json.load(open(os.path.join(HERE, "meros1_alpha.json")))["OA vs SA"]
    alpha = cal["alpha"]; r1, r2 = cal["r1"], cal["r2"]
    z_thr = m5["z_thr"]
    grid = np.array(m5["taus"])
    P = m5["pairs"]["OA vs SB"]

    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    SA, SB, OA = d['SA'], d['SB'], d['OA']
    n = len(SA)
    S = np.where(SB == 2, 1.0, -1.0)
    SB1 = (SB == 1).astype(np.int8)
    lam0 = np.where(SA == 1, r1, r2)

    kax = np.arange(-a.K, a.K + 1, dtype=np.float64)
    rng = np.random.default_rng(a.seed)

    print("=" * 78)
    print("ΕΠΑΛΗΘΕΥΣΗ ΤΩΝ ΑΣΥΜΜΕΤΡΩΝ ΦΙΛΤΡΩΝ ΜΕ ΕΝΕΣΗ  (ζεύγος OA vs SB)")
    print("=" * 78)
    print(f"  κατώφλι z = {z_thr:.3f}   {a.reps} επαναλήψεις ανά σημείο\n")

    res = []
    for kn in a.kernels:
        print("-" * 78)
        print(f"ΠΥΡΗΝΑΣ: {kn} — {KERNEL_LABEL[kn]}")
        print("-" * 78)
        for tau in a.taus:
            j = int(np.argmin(np.abs(grid - tau)))
            assert abs(grid[j] - tau) < 1e-9, "το τ δεν είναι στο πλέγμα"
            eps_x = P[kn]["eps_excl"][j]
            z_obs = P[kn]["z"][j]
            Wf = kernel_W(kn, kax, tau)[None, :]          # φίλτρο (1 × n_lag)
            F, half = build_F_kernel(S, kn, tau, a.K)
            werr = check_F(S, F, kn, tau, half, rng)
            print(f"  τ = {tau:g}   ε_excl = {eps_x:.4e}   z_obs = {z_obs:+.2f}"
                  f"   half = {half}   σφάλμα F = {werr:.2e}   "
                  f"sd(F) = {F.std():.3f}")

            for frac in (1.0, 2.0):
                eps = frac * eps_x
                zs, clips = [], []
                for _ in range(a.reps):
                    lam = lam0 + alpha * eps * F
                    clips.append(float(np.count_nonzero(
                        (lam < 0) | (lam > 1)) / n))
                    O = (rng.random(n) < np.clip(lam, 0, 1)).astype(np.int8)
                    _, n11, A1, B1, nk, _ = scan(O, SB1, a.K)
                    _, dl = mi_and_delta(n11, A1, B1, nk)
                    sd = sigma_delta(A1, B1, nk)
                    T = float((Wf @ dl)[0])
                    sT = float(np.sqrt(Wf ** 2 @ sd ** 2)[0])
                    zs.append(T / sT)
                zs = np.array(zs)
                z_pred = frac * (z_thr + abs(z_obs))
                sem = zs.std(ddof=1) / math.sqrt(a.reps)
                dev = (zs.mean() - z_pred) / sem if sem > 0 else 0.0
                npass = int((np.abs(zs) > z_thr).sum())
                print(f"    ε = {eps:.4e} ({frac:g}×)  clip {max(clips)*100:.4f}%"
                      f"   z = {zs.mean():+.2f} ± {sem:.2f}  "
                      f"(πρόβλεψη {z_pred:+.2f}, {dev:+.1f}σ)   "
                      f"ανιχνεύθηκε {npass}/{a.reps}")
                rec = dict(kernel=kn, tau=tau, frac=frac, eps=eps,
                           z_mean=float(zs.mean()),
                           z_sd=float(zs.std(ddof=1)), z_sem=float(sem),
                           z_pred=z_pred, dev_sigma=float(dev), z_obs=z_obs,
                           n_pass=npass, reps=a.reps, max_clip=max(clips),
                           zs=zs.tolist())

                # cross-check ΜΟΝΟ στο frac=2, μία επανάληψη: το ίδιο σήμα
                # περασμένο από το ΑΝΤΙΘΕΤΟ φίλτρο (μέλλον <-> παρελθόν)
                if frac == 2.0:
                    opp = {"future": "past", "past": "future",
                           "exp_future": "exp_past (κάτοπτρο)"}[kn]
                    Wo = Wf[:, ::-1].copy()        # κάτοπτρο ως προς k -> -k
                    lam = lam0 + alpha * eps * F
                    O = (rng.random(n) < np.clip(lam, 0, 1)).astype(np.int8)
                    _, n11, A1, B1, nk, _ = scan(O, SB1, a.K)
                    _, dl = mi_and_delta(n11, A1, B1, nk)
                    sd = sigma_delta(A1, B1, nk)
                    zo = float((Wo @ dl)[0] /
                               np.sqrt(Wo ** 2 @ sd ** 2)[0])
                    print(f"      cross-check: το ίδιο σήμα μέσα από το "
                          f"φίλτρο «{opp}» -> z = {zo:+.2f}")
                    rec["cross_kernel"] = opp
                    rec["cross_z"] = zo
                res.append(rec)
            print()

    print("=" * 78)
    print("ΕΤΥΜΗΓΟΡΙΑ")
    print("=" * 78)
    worst = max(abs(r["dev_sigma"]) for r in res)
    ok_cal = worst < 3.0
    ok_hi = all(r["n_pass"] == r["reps"] for r in res if r["frac"] == 2.0)
    cross = [r for r in res if "cross_z" in r]
    ok_cross = all(abs(r["cross_z"]) < z_thr for r in cross)
    print(f"  συμφωνία z με την πρόβλεψη: μέγιστη απόκλιση {worst:.1f}σ  -> "
          f"{'ΝΑΙ' if ok_cal else 'ΟΧΙ'}")
    print(f"  2·ε_excl ανιχνεύεται ΠΑΝΤΑ: {'ΝΑΙ' if ok_hi else 'ΟΧΙ'}")
    print(f"  αντίθετο φίλτρο ΔΕΝ ανάβει: {'ΝΑΙ' if ok_cross else 'ΟΧΙ'} "
          f"(max |z| = {max(abs(r['cross_z']) for r in cross):.2f})")
    for r in res:
        if r["frac"] == 1.0:
            print(f"    (ισχύς στο ίδιο το ε_excl, {r['kernel']} τ={r['tau']:g}: "
                  f"{r['n_pass']}/{r['reps']} — αναμένεται ~50%, ΔΕΝ είναι "
                  f"κριτήριο)")
    print(f"  ΤΑ ΑΣΥΜΜΕΤΡΑ ΦΙΛΤΡΑ ΕΠΑΛΗΘΕΥΟΝΤΑΙ: "
          f"{'ΝΑΙ' if (ok_cal and ok_hi and ok_cross) else 'ΟΧΙ'}")

    json.dump(dict(z_thr=z_thr, points=res, worst_dev_sigma=worst,
                   verified=bool(ok_cal and ok_hi and ok_cross)),
              open(os.path.join(HERE, "meros5_verify.json"), "w"), indent=2)
    print("\nΑποθηκεύτηκε: meros5_verify.json")


if __name__ == "__main__":
    main()
