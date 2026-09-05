"""
PART 3, STEP 6 - VERIFYING THE MAP BY INJECTION

Without this the map is an arithmetic exercise. We inject a signal EXACTLY at
the eps_excl(tau) the map states, and at half of it, and check that the
matched filter behaves as promised:

WHAT THE RIGHT CRITERION IS (correcting an earlier version)
    eps_excl is a CONFIDENCE UPPER BOUND, not a power threshold. By
    construction a signal EXACTLY at eps_excl is detected only half the time
    (z is distributed about the threshold with sd 1). "Always detected" would
    be the WRONG requirement. The right test is quantitative:

        E[z] = frac * ( z_thr + |z_obs(tau)| )

    (the synthetic O does not inherit the z_obs of the real data -- it is
     fresh Bernoulli -- but eps_excl contains it through |eps-hat|)

    frac = 0    -> E[z] = 0     , 0% detection
    frac = 0.5  -> E[z] ~ 2.7   , ~2% detection
    frac = 1    -> E[z] ~ 5.4   , ~50-70% detection
    frac = 2    -> E[z] ~ 10.8  , ~100% detection

    It PASSES if the measured mean z agrees with E[z] to within 3 standard
    errors of the mean, AND frac=2 is always detected, AND frac=0 never is.

Several seeds per point: the injection is random and one sample says
nothing. The mean and sd of z are reported.

Clipping is checked here too (the eps values are small, so zero is expected).
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
    print("VERIFYING THE MAP BY INJECTION  (pair OA vs SB)")
    print("=" * 78)
    print(f"  threshold z = {z_thr:.3f}   {a.reps} repetitions per point\n")

    res = []
    for tau in a.taus:
        # eps_excl and z_obs at this tau: linear interpolation in log-log
        eps_x = float(np.exp(np.interp(math.log(tau), np.log(grid),
                                       np.log(eps_grid))))
        Q, _ = kernel_Q([tau], a.K)
        Wmat = filters([tau], a.K)
        Qv = float(Q[0])

        F, _, _, _ = build_F(S, tau)
        print(f"tau = {tau:g}   eps_excl (interpolated) = {eps_x:.4e}   "
              f"Q = {Qv:.1f}")

        z_obs = float(np.interp(math.log(tau), np.log(grid), z_obs_grid))
        for frac, name in [(0.0, "eps = 0"), (0.5, "eps_excl/2"),
                           (1.0, "eps_excl"), (2.0, "2*eps_excl")]:
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
            print(f"    {name:>10}  eps = {eps:.4e}  "
                  f"clip {max(clips)*100:.4f}%   "
                  f"z = {zs.mean():+.2f} +/- {sem:.2f}  "
                  f"(predicted {z_pred:+.2f}, deviation {dev:+.1f} sigma)   "
                  f"detected: {passed}/{a.reps}")
            res.append(dict(tau=tau, frac=frac, eps=eps,
                            z_mean=float(zs.mean()),
                            z_sd=float(zs.std(ddof=1)), z_sem=float(sem),
                            z_pred=z_pred, dev_sigma=float(dev),
                            z_obs=z_obs, n_pass=passed, reps=a.reps,
                            max_clip=max(clips), zs=zs.tolist()))
        print()
        del F

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    worst = max(abs(r["dev_sigma"]) for r in res if r["frac"] > 0)
    ok_cal = worst < 3.0
    ok_hi = all(r["n_pass"] == r["reps"] for r in res if r["frac"] == 2.0)
    ok_z0 = all(r["n_pass"] == 0 for r in res if r["frac"] == 0.0)
    print(f"  calibration: largest deviation of z from the prediction "
          f"{worst:.1f} sigma  -> {'YES' if ok_cal else 'NO'}")
    print(f"  2*eps_excl -> ALWAYS detected: {'YES' if ok_hi else 'NO'}")
    print(f"  eps = 0    -> NEVER detected: {'YES' if ok_z0 else 'NO'}")
    for r in res:
        if r["frac"] == 1.0:
            print(f"    (power at the bound itself, tau={r['tau']:g}: "
                  f"{r['n_pass']}/{r['reps']} -- ~50-70% expected, "
                  f"NOT a criterion)")
    print(f"  THE MAP IS VERIFIED: "
          f"{'YES' if (ok_cal and ok_hi and ok_z0) else 'NO'}")

    json.dump(dict(z_thr=z_thr, points=res, worst_dev_sigma=worst,
                   verified=bool(ok_cal and ok_hi and ok_z0)),
              open(os.path.join(HERE, "meros3_verify.json"), "w"), indent=2)
    print("\nSaved: meros3_verify.json")


if __name__ == "__main__":
    main()
