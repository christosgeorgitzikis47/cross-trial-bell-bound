"""
ΜΕΡΟΣ 7 — ΚΑΜΠΥΛΗ ΙΣΧΥΟΣ ΓΙΑ ΤΟ FIGURE 4

Η επαλήθευση του Μέρους 5 έτρεξε μόνο σε ε_excl και 2·ε_excl (έτσι ζητήθηκε).
Για το σχήμα χρειάζονται και τα ε = 0 και ε_excl/2, ώστε ο άξονας
ε/ε_excl να έχει τέσσερα σημεία. Εδώ τρέχουν ΜΟΝΟ τα δύο που λείπουν και
συγχωνεύονται με τα υπάρχοντα.

Τίποτα δεν ξαναϋπολογίζεται από όσα ήδη υπάρχουν: τα σημεία frac = 1 και 2
έρχονται αυτούσια από το `meros5_verify.json`.
"""
import argparse, json, math, os
import numpy as np

from meros2_injection import scan, mi_and_delta
from meros3_map import sigma_delta
from meros5_asym import kernel_W
from meros5_verify import build_F_kernel

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--K", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=1707)
    a = ap.parse_args()

    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    ver = json.load(open(os.path.join(HERE, "meros5_verify.json")))
    cal = json.load(open(os.path.join(HERE, "meros1_alpha.json")))["OA vs SA"]
    alpha, r1, r2 = cal["alpha"], cal["r1"], cal["r2"]
    z_thr = m5["z_thr"]
    taus = np.array(m5["taus"])
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
    print("ΜΕΡΟΣ 7 — τα σημεία ε = 0 και ε_excl/2 που έλειπαν")
    print("=" * 78)
    print(f"  κατώφλι z = {z_thr:.3f}   {a.reps} επαναλήψεις ανά σημείο\n")

    out = []
    for kn in ("future", "past", "exp_future"):
        for tau in (30.0, 300.0):
            j = int(np.argmin(np.abs(taus - tau)))
            eps_x = P[kn]["eps_excl"][j]
            z_obs = P[kn]["z"][j]
            Wf = kernel_W(kn, kax, tau)[None, :]
            F, half = build_F_kernel(S, kn, tau, a.K)
            print(f"{kn} τ={tau:g}  ε_excl={eps_x:.4e}", flush=True)
            for frac in (0.0, 0.5):
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
                npass = int((np.abs(zs) > z_thr).sum())
                print(f"    frac {frac:>3}: z = {zs.mean():+.2f} ± "
                      f"{zs.std(ddof=1):.2f}   ανιχνεύθηκε {npass}/{a.reps}"
                      f"   clip {max(clips)*100:.4f}%")
                out.append(dict(kernel=kn, tau=tau, frac=frac, eps=eps,
                                z_mean=float(zs.mean()),
                                z_sd=float(zs.std(ddof=1)),
                                z_sem=float(zs.std(ddof=1) / math.sqrt(a.reps)),
                                n_pass=npass, reps=a.reps,
                                max_clip=max(clips), zs=zs.tolist()))
            del F

    # --- συγχώνευση με τα υπάρχοντα frac = 1, 2 ---
    for p in ver["points"]:
        out.append({k: p[k] for k in
                    ("kernel", "tau", "frac", "eps", "z_mean", "z_sd",
                     "z_sem", "n_pass", "reps", "max_clip", "zs")})
    out.sort(key=lambda r: (r["kernel"], r["tau"], r["frac"]))

    print(f"\nΣΥΝΟΛΟ: {len(out)} σημεία "
          f"(3 πυρήνες × 2 τ × 4 επίπεδα ε)")
    for frac in (0.0, 0.5, 1.0, 2.0):
        sel = [r for r in out if r["frac"] == frac]
        det = sum(r["n_pass"] for r in sel)
        tot = sum(r["reps"] for r in sel)
        print(f"  ε/ε_excl = {frac:<4} ανιχνεύθηκε {det:>3}/{tot}   "
              f"μέσο z = {np.mean([r['z_mean'] for r in sel]):+.2f}")

    json.dump(dict(z_thr=z_thr, reps=a.reps, points=out),
              open(os.path.join(HERE, "meros7_power.json"), "w"), indent=2)
    print("\nΑποθηκεύτηκε: meros7_power.json")


if __name__ == "__main__":
    main()
