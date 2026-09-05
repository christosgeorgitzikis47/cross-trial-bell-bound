"""
DIAGNOSTIC (post hoc -- NOT the pre-specified criterion).

Observation: the OA vs SB peaks of the 5 NEW pulses all sit at negative lags
and are clustered: -39, -41, -39, -46, -32.

Two things must be told apart:
  (a) A POINTWISE COINCIDENCE -- the argmax of 5 independent noisy curves
      happened to land close together. Then the rest of the curve is flat.
  (b) BROAD STRUCTURE -- the whole region of negative lags is raised. Then the
      argmax is merely the tip of a hill, and the effect is real (though not
      necessarily physical).

We measure the mean MI in lag bands per pulse. No null here: all pulses have
the same n, so the raw MI values are comparable BETWEEN BANDS of the same
pulse.
"""
import json, os
import numpy as np
from lag_test import mi, align
from lag_dense import LAGS
from load_curby import read_file

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "..", "dedomena_curby")
OLD = [28293, 28294, 28295, 28296, 28297]
NEW = [1000, 15000, 22000, 23000, 26000]
LAGSET = sorted(LAGS)


def curves(rnd):
    d, _ = read_file(os.path.join(BIN, f"curby_round_{rnd}.bin"))
    SA = d['SA'].astype(np.int8); SB = d['SB'].astype(np.int8)
    OA = (d['OA'] > 0).astype(np.int8); OB = (d['OB'] > 0).astype(np.int8)
    out = {}
    for label, o, s in [("OA vs SB", OA, SB), ("OB vs SA", OB, SA)]:
        out[label] = [mi(*align(o, s, k)) for k in LAGSET]
    return out


def zones(vals):
    v = np.array(vals); k = np.array(LAGSET)
    near = np.abs(k) <= 50
    return {
        "neg_32_46": float(v[(k >= -46) & (k <= -32)].mean()),
        "neg_rest":  float(v[near & (k < 0) & ~((k >= -46) & (k <= -32))].mean()),
        "pos_all":   float(v[near & (k > 0)].mean()),
        "far":       float(v[~near].mean()),
    }


def main():
    res = {}
    for grp, rounds in [("OLD", OLD), ("NEW", NEW)]:
        for r in rounds:
            print(f"  round {r} ...", flush=True)
            res[str(r)] = curves(r)
    json.dump({"lags": LAGSET, "curves": res},
              open(os.path.join(HERE, "diagnostiko_curves.json"), "w"))

    for label in ("OA vs SB", "OB vs SA"):
        print("\n" + "=" * 84)
        print(f"{label} - mean MI per lag band  (x1e-7 bits)")
        print("=" * 84)
        print(f"{'group':>8} {'round':>7} {'band -46..-32':>15} {'other neg.':>13} "
              f"{'positive':>10} {'far':>10} {'ratio':>8}")
        for grp, rounds in [("OLD", OLD), ("NEW", NEW)]:
            for r in rounds:
                z = zones(res[str(r)][label])
                ratio = z["neg_32_46"] / ((z["neg_rest"] + z["pos_all"]) / 2)
                print(f"{grp:>8} {r:>7} {z['neg_32_46']*1e7:>15.3f} "
                      f"{z['neg_rest']*1e7:>13.3f} {z['pos_all']*1e7:>10.3f} "
                      f"{z['far']*1e7:>10.3f} {ratio:>8.2f}")

    # --- how unlikely is the clustering, AS A POST-HOC statistic? ---
    rng = np.random.default_rng(7)
    obs = np.array([-39, -41, -39, -46, -32])
    w = obs.max() - obs.min()                 # range = 14
    L = len(LAGSET)
    idx = rng.integers(0, L, size=(400_000, 5))
    ks = np.array(LAGSET)[idx]
    inside = np.abs(ks) <= 50                 # the range is defined on near lags only
    rng_w = ks.max(axis=1) - ks.min(axis=1)
    p_range = float(((rng_w <= w) & inside.all(axis=1)).mean())
    p_allneg = float((ks < 0).all(axis=1).mean())
    print(f"\nPOST HOC (NOT the pre-specified criterion):")
    print(f"  P(5 random peaks all near and range <= {w}) = {p_range:.5f}")
    print(f"  P(5 random peaks all negative)             = {p_allneg:.5f}")
    print(f"  NOTE: the window was chosen AFTER seeing the data. The p value is")
    print(f"  an indicator, not a test. The same applies to the other channels")
    print(f"  and groups that did NOT stand out - they are not in the denominator.")


if __name__ == "__main__":
    main()
