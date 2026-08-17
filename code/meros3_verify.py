"""
ΜΕΡΟΣ 3, ΒΗΜΑ 6 — ΕΠΑΛΗΘΕΥΣΗ ΤΟΥ ΧΑΡΤΗ ΜΕ ΕΝΕΣΗ

Χωρίς αυτό ο χάρτης είναι αριθμητική άσκηση. Ενίουμε σήμα ΑΚΡΙΒΩΣ στο
ε_excl(τ) που δηλώνει ο χάρτης, και στο μισό του, και ελέγχουμε ότι το
matched filter συμπεριφέρεται όπως υπόσχεται:

ΠΟΙΟ ΕΙΝΑΙ ΤΟ ΣΩΣΤΟ ΚΡΙΤΗΡΙΟ (διόρθωση προηγούμενης εκδοχής)
    Το ε_excl είναι ΑΝΩ ΟΡΙΟ ΕΜΠΙΣΤΟΣΥΝΗΣ, όχι όριο ισχύος. Εξ ορισμού,
    σήμα ΑΚΡΙΒΩΣ στο ε_excl ανιχνεύεται μόνο τις μισές φορές (το z
    κατανέμεται γύρω από το κατώφλι με sd 1). Το «ανιχνεύεται πάντα» θα
    ήταν ΛΑΘΟΣ απαίτηση. Το σωστό τεστ είναι ποσοτικό:

        E[z] = frac · ( z_thr + |z_obs(τ)| )

    (το συνθετικό O δεν κληρονομεί το z_obs των πραγματικών δεδομένων —
     είναι φρέσκο Bernoulli — αλλά το ε_excl το περιέχει μέσω του |ε̂|)

    frac = 0    -> E[z] = 0        , 0% ανίχνευση
    frac = 0,5  -> E[z] ≈ 2,7      , ~2% ανίχνευση
    frac = 1    -> E[z] ≈ 5,4      , ~50-70% ανίχνευση
    frac = 2    -> E[z] ≈ 10,8     , ~100% ανίχνευση

    ΠΕΡΝΑΕΙ αν το μετρημένο μέσο z συμφωνεί με το E[z] εντός 3 τυπικών
    σφαλμάτων του μέσου, ΚΑΙ το frac=2 ανιχνεύεται πάντα, ΚΑΙ το frac=0
    ποτέ.

Πολλαπλά seeds ανά σημείο: η ένεση είναι τυχαία, ένα δείγμα δεν λέει
τίποτα. Αναφέρεται μέσος ± sd του z.

Το clipping ελέγχεται και εδώ (τα ε είναι μικρά, αναμένεται μηδενικό).
"""
import argparse, json, math, os
import numpy as np

from meros2_injection import build_F, scan, mi_and_delta
from meros3_map import sigma_delta, kernel_Q, filters

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[3.0, 30.0, 300.0, 3000.0])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--K", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=808)
    a = ap.parse_args()

    m3 = json.load(open(os.path.join(HERE, "meros3_map.json")))
    cal = json.load(open(os.path.join(HERE, "meros1_alpha.json")))["OA vs SA"]
    alpha = cal["alpha"]
    r1, r2 = cal["r1"], cal["r2"]
    z_thr = m3["z_thr"]

    grid = np.array(m3["taus"])
    eps_grid = np.array(m3["pairs"]["OA vs SB"]["eps_excl"])
    z_obs_grid = np.array(m3["pairs"]["OA vs SB"]["z"])

    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    SA, SB, OA = d['SA'], d['SB'], d['OA']
    n = len(SA)
    S = np.where(SB == 2, 1.0, -1.0)
    SB1 = (SB == 1).astype(np.int8)
    lam0 = np.where(SA == 1, r1, r2)

    rng = np.random.default_rng(a.seed)

    print("=" * 78)
    print("ΕΠΑΛΗΘΕΥΣΗ ΤΟΥ ΧΑΡΤΗ ΜΕ ΕΝΕΣΗ  (ζεύγος OA vs SB)")
    print("=" * 78)
    print(f"  κατώφλι z = {z_thr:.3f}   {a.reps} επαναλήψεις ανά σημείο\n")

    res = []
    for tau in a.taus:
        # ε_excl και z_obs στο τ αυτό: γραμμική παρεμβολή σε log-log
        eps_x = float(np.exp(np.interp(math.log(tau), np.log(grid),
                                       np.log(eps_grid))))
        Q, _ = kernel_Q([tau], a.K)
        Wmat = filters([tau], a.K)
        Qv = float(Q[0])

        F, _, _, _ = build_F(S, tau)
        print(f"τ = {tau:g}   ε_excl (παρεμβολή) = {eps_x:.4e}   Q = {Qv:.1f}")

        z_obs = float(np.interp(math.log(tau), np.log(grid), z_obs_grid))
        for frac, name in [(0.0, "ε = 0"), (0.5, "ε_excl/2"),
                           (1.0, "ε_excl"), (2.0, "2·ε_excl")]:
            eps = frac * eps_x
            zs, clips = [], []
            for r in range(a.reps):
                lam = lam0 + alpha * eps * F
                clips.append(float(np.count_nonzero(
                    (lam < 0) | (lam > 1)) / n))
                O = (rng.random(n) < np.clip(lam, 0, 1)).astype(np.int8)
                _, n11, A1, B1, nk, _ = scan(O, SB1, a.K)
                _, dl = mi_and_delta(n11, A1, B1, nk)
                sd = sigma_delta(A1, B1, nk)
                T = float((Wmat @ dl)[0])
                sT = float(np.sqrt(Wmat ** 2 @ sd ** 2)[0])
                zs.append(T / sT)
            zs = np.array(zs)
            z_pred = frac * (z_thr + abs(z_obs))
            passed = int((np.abs(zs) > z_thr).sum())
            sem = zs.std(ddof=1) / math.sqrt(a.reps)
            dev = (zs.mean() - z_pred) / sem if sem > 0 else 0.0
            print(f"    {name:>10}  ε = {eps:.4e}  clip {max(clips)*100:.4f}%   "
                  f"z = {zs.mean():+.2f} ± {sem:.2f}  "
                  f"(πρόβλεψη {z_pred:+.2f}, απόκλιση {dev:+.1f}σ)   "
                  f"ανιχνεύθηκε: {passed}/{a.reps}")
            res.append(dict(tau=tau, frac=frac, eps=eps,
                            z_mean=float(zs.mean()),
                            z_sd=float(zs.std(ddof=1)), z_sem=float(sem),
                            z_pred=z_pred, dev_sigma=float(dev),
                            z_obs=z_obs, n_pass=passed, reps=a.reps,
                            max_clip=max(clips), zs=zs.tolist()))
        print()
        del F

    print("=" * 78)
    print("ΕΤΥΜΗΓΟΡΙΑ")
    print("=" * 78)
    worst = max(abs(r["dev_sigma"]) for r in res if r["frac"] > 0)
    ok_cal = worst < 3.0
    ok_hi = all(r["n_pass"] == r["reps"] for r in res if r["frac"] == 2.0)
    ok_z0 = all(r["n_pass"] == 0 for r in res if r["frac"] == 0.0)
    print(f"  βαθμονόμηση: μέγιστη απόκλιση του z από την πρόβλεψη "
          f"{worst:.1f}σ  -> {'ΝΑΙ' if ok_cal else 'ΟΧΙ'}")
    print(f"  2·ε_excl -> ανιχνεύεται ΠΑΝΤΑ: {'ΝΑΙ' if ok_hi else 'ΟΧΙ'}")
    print(f"  ε = 0    -> ΔΕΝ ανιχνεύεται ΠΟΤΕ: {'ΝΑΙ' if ok_z0 else 'ΟΧΙ'}")
    for r in res:
        if r["frac"] == 1.0:
            print(f"    (ισχύς στο ίδιο το ε_excl, τ={r['tau']:g}: "
                  f"{r['n_pass']}/{r['reps']} — αναμένεται ~50-70%, "
                  f"ΔΕΝ είναι κριτήριο)")
    print(f"  Ο ΧΑΡΤΗΣ ΕΠΑΛΗΘΕΥΕΤΑΙ: "
          f"{'ΝΑΙ' if (ok_cal and ok_hi and ok_z0) else 'ΟΧΙ'}")

    json.dump(dict(z_thr=z_thr, points=res, worst_dev_sigma=worst,
                   verified=bool(ok_cal and ok_hi and ok_z0)),
              open(os.path.join(HERE, "meros3_verify.json"), "w"), indent=2)
    print("\nΑποθηκεύτηκε: meros3_verify.json")


if __name__ == "__main__":
    main()
