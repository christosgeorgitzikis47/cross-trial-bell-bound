"""
PART 6.4 - alpha ACROSS THE TEN PULSES (peer review objection #4)

Limitation #1 said "alpha was calibrated on 28297 and is not a constant of
the dataset". True but vague. Here it becomes a number.

    alpha = delta(0) = (r2 - r1)/2,  r_s = click rate under setting s,
    SAME pair.

It is computed from the raw files for all ten pulses, separately for Alice
(OA vs SA) and Bob (OB vs SB). Range, mean and sd are reported.

WHY IT MATTERS: eps_excl is proportional to 1/alpha. A pulse with smaller
alpha gives a looser bound at the same n. The spread of alpha is directly
the uncertainty in transferring the map to another pulse.
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dedomena_curby")
sys.path.insert(0, HERE)
from load_curby import read_file                                  # noqa: E402

ROUNDS = [1000, 15000, 22000, 23000, 26000,
          28293, 28294, 28295, 28296, 28297]
GROUP = {r: ("spread" if r < 28000 else "consecutive") for r in ROUNDS}


def rates(O, S):
    """r1, r2, n1, n2 for the lag-0 2x2 table."""
    m1 = (S == 1)
    n1 = int(m1.sum()); n2 = int(len(S) - n1)
    k1 = int(O[m1].sum()); k2 = int(O.sum() - k1)
    return k1 / n1, k2 / n2, n1, n2, k1, k2


def main():
    print("=" * 78)
    print("PART 6.4 - alpha IN EACH OF THE 10 PULSES")
    print("=" * 78)
    print(f"  {'round':>7} {'group':>12} {'p0(A)':>9} {'r1(A)':>10} "
          f"{'r2(A)':>10} {'alpha(A)':>11} {'alpha(B)':>11} {'r2/r1(A)':>9}")

    res = []
    for r in ROUNDS:
        path = os.path.join(DATA, f"curby_round_{r}.bin")
        data, n_raw = read_file(path)
        SA = data['SA'].astype(np.int8); SB = data['SB'].astype(np.int8)
        OA = (data['OA'] > 0).astype(np.int8)
        OB = (data['OB'] > 0).astype(np.int8)
        n = len(SA)

        rA1, rA2, nA1, nA2, kA1, kA2 = rates(OA, SA)
        rB1, rB2, nB1, nB2, kB1, kB2 = rates(OB, SB)
        aA = (rA2 - rA1) / 2.0
        aB = (rB2 - rB1) / 2.0
        p0A = (kA1 + kA2) / n
        p0B = (kB1 + kB2) / n
        # standard error of alpha from the binomial
        seA = 0.5 * math.sqrt(rA1 * (1 - rA1) / nA1 + rA2 * (1 - rA2) / nA2)
        seB = 0.5 * math.sqrt(rB1 * (1 - rB1) / nB1 + rB2 * (1 - rB2) / nB2)

        print(f"  {r:>7} {GROUP[r]:>12} {p0A:>9.5f} {rA1:>10.6f} "
              f"{rA2:>10.6f} {aA:>11.4e} {aB:>11.4e} {rA2/rA1:>9.4f}")
        res.append(dict(round=r, group=GROUP[r], n=n, n_raw=n_raw,
                        r1_A=rA1, r2_A=rA2, alpha_A=aA, se_alpha_A=seA,
                        p0_A=p0A, r1_B=rB1, r2_B=rB2, alpha_B=aB,
                        se_alpha_B=seB, p0_B=p0B, ratio_A=rA2 / rA1))
        del data, SA, SB, OA, OB

    aA = np.array([x["alpha_A"] for x in res])
    aB = np.array([x["alpha_B"] for x in res])
    seA = np.array([x["se_alpha_A"] for x in res])
    p0A = np.array([x["p0_A"] for x in res])

    print("\n" + "-" * 78)
    for nm, v in (("alpha Alice", aA), ("alpha Bob", aB)):
        print(f"  {nm}: mean {v.mean():.4e}  sd {v.std(ddof=1):.4e} "
              f"({100*v.std(ddof=1)/v.mean():.1f}%)   "
              f"range [{v.min():.4e}, {v.max():.4e}]  "
              f"max/min = {v.max()/v.min():.3f}")
    print(f"  MEASUREMENT standard error per pulse: ~{seA.mean():.2e} "
          f"({100*seA.mean()/aA.mean():.2f}%) -> the spread is "
          f"{aA.std(ddof=1)/seA.mean():.0f}x larger, hence REAL")
    print(f"  p0 Alice: range [{p0A.min():.5f}, {p0A.max():.5f}]  "
          f"({100*p0A.min():.3f}% - {100*p0A.max():.3f}%)")

    a28297 = [x for x in res if x["round"] == 28297][0]["alpha_A"]
    print(f"\n  The map uses alpha(28297) = {a28297:.4e}.")
    print(f"  Since eps_excl is proportional to 1/alpha, transferring the map")
    print(f"  to another pulse changes the bound by alpha(28297)/alpha(pulse):")
    print(f"    {'round':>7} {'alpha(A)':>11} {'a(28297)/a':>11}")
    for x in res:
        print(f"    {x['round']:>7} {x['alpha_A']:>11.4e} "
              f"{a28297/x['alpha_A']:>11.3f}")
    fac = a28297 / aA
    print(f"\n  -> eps_excl on another pulse would be {fac.min():.2f}x to "
          f"{fac.max():.2f}x that of this map (from alpha alone, same n).")

    json.dump(dict(rounds=res,
                   alpha_A_mean=float(aA.mean()), alpha_A_sd=float(aA.std(ddof=1)),
                   alpha_A_min=float(aA.min()), alpha_A_max=float(aA.max()),
                   alpha_B_mean=float(aB.mean()), alpha_B_sd=float(aB.std(ddof=1)),
                   alpha_B_min=float(aB.min()), alpha_B_max=float(aB.max()),
                   alpha_28297_A=a28297,
                   transfer_factor_min=float(fac.min()),
                   transfer_factor_max=float(fac.max())),
              open(os.path.join(HERE, "meros6_alpha10.json"), "w"), indent=2)
    print("\nSaved: meros6_alpha10.json")


if __name__ == "__main__":
    main()
