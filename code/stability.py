"""
Σταθερότητα στους 5 παλμούς: lag test + J(k) σε κάθε γύρο.

ΤΟ ΕΡΩΤΗΜΑ ΔΕΝ ΕΙΝΑΙ «βρέθηκε σήμα;» αλλά «εμφανίζεται ΤΟ ΙΔΙΟ lag σε
πολλούς παλμούς;». Τυχαίο σήμα μετακινείται από παλμό σε παλμό·
συστηματικό μένει στο ίδιο lag.

500 ανακατέματα ανά ζεύγος: αρκούν, γιατί συγκρίνουμε ΘΕΣΕΙΣ κορυφών
μεταξύ παλμών, όχι απόλυτη σημαντικότητα.
"""
import json, os
import numpy as np
from lag_test import mi, align
from lag_dense import build_null, LAGS
from j_curve import align_so, eberhard, LAGS as JLAGS
from load_curby import read_file

ROUNDS = [28293, 28294, 28295, 28296, 28297]
BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "dedomena_curby")
N_SHUFFLE = 500


def analyse(rnd):
    path = os.path.join(BIN_DIR, f"curby_round_{rnd}.bin")
    data, n_raw = read_file(path)
    SA = data['SA'].astype(np.int8)
    SB = data['SB'].astype(np.int8)
    OA = (data['OA'] > 0).astype(np.int8)
    OB = (data['OB'] > 0).astype(np.int8)
    out = {"round": rnd, "n": int(len(SA)),
           "rateA": float(OA.mean()), "rateB": float(OB.mean())}

    # ---- lag test ----
    lags = sorted(LAGS)
    for label, o, s in [("OA vs SB", OA, SB), ("OB vs SA", OB, SA)]:
        nd = build_null(o, s, N_SHUFFLE, seed=4242 + rnd)
        mu, sd, mx = nd.mean(), nd.std(ddof=1), nd.max()
        vals = []
        for k in lags:
            oo, ss = align(o, s, k)
            vals.append(mi(oo, ss))
        vals = np.array(vals)
        z = (vals - mu) / sd
        i = int(np.argmax(z))
        out[label] = {"max_sigma": float(z[i]), "max_sigma_lag": int(lags[i]),
                      "n_above_null_max": int((vals > mx).sum()),
                      "null_max_bits": float(mx),
                      "max_mi_bits": float(vals.max())}

    # ---- J(k) ----
    jl = sorted(JLAGS)
    Js, zs = [], []
    for k in jl:
        sa, sb, oa, ob = align_so(SA, SB, OA, OB, k)
        J, s_, _ = eberhard(sa, sb, oa, ob)
        Js.append(int(J)); zs.append(float(J / s_) if s_ else 0.0)
    zs = np.array(zs); jl = np.array(jl)
    i0 = int(np.where(jl == 0)[0][0])
    nz = np.array([i for i in range(len(jl)) if jl[i] != 0])
    out["J"] = {"J0": Js[i0], "z0": float(zs[i0]),
                "max_z_nonzero": float(zs[nz].max()),
                "max_z_nonzero_k": int(jl[nz][int(np.argmax(zs[nz]))]),
                "n_positive_J_nonzero": int(sum(1 for i in nz if Js[i] > 0))}
    return out


def main():
    res = []
    for r in ROUNDS:
        print(f"--- γύρος {r} ---", flush=True)
        o = analyse(r)
        res.append(o)
        print(f"    OA vs SB: μέγ. {o['OA vs SB']['max_sigma']:+.2f}σ @ lag "
              f"{o['OA vs SB']['max_sigma_lag']}   πάνω από null max: "
              f"{o['OA vs SB']['n_above_null_max']}", flush=True)
        print(f"    OB vs SA: μέγ. {o['OB vs SA']['max_sigma']:+.2f}σ @ lag "
              f"{o['OB vs SA']['max_sigma_lag']}   πάνω από null max: "
              f"{o['OB vs SA']['n_above_null_max']}", flush=True)
        print(f"    J(0) = {o['J']['J0']:+,} ({o['J']['z0']:+.2f}σ)   "
              f"k≠0 με J>0: {o['J']['n_positive_J_nonzero']}", flush=True)

    print("\n" + "=" * 78)
    print("ΣΥΝΟΨΗ ΣΤΑΘΕΡΟΤΗΤΑΣ")
    print("=" * 78)
    print(f"{'γύρος':>7} | {'OA vs SB':>18} | {'OB vs SA':>18} | "
          f"{'J(0)':>10} {'J/σ':>8}")
    print(f"{'':>7} | {'μέγ.σ':>9}{'@lag':>9} | {'μέγ.σ':>9}{'@lag':>9} | ")
    print("-" * 78)
    for o in res:
        a, b = o["OA vs SB"], o["OB vs SA"]
        print(f"{o['round']:>7} | {a['max_sigma']:>+9.2f}{a['max_sigma_lag']:>9} | "
              f"{b['max_sigma']:>+9.2f}{b['max_sigma_lag']:>9} | "
              f"{o['J']['J0']:>+10,} {o['J']['z0']:>+8.2f}")

    print("\nΕΠΑΝΑΛΑΜΒΑΝΟΜΕΝΑ LAG;")
    for label in ("OA vs SB", "OB vs SA"):
        ls = [o[label]["max_sigma_lag"] for o in res]
        uniq = len(set(ls))
        print(f"  {label}: κορυφές στα lag {ls}")
        print(f"     -> {uniq} διαφορετικά lag σε {len(ls)} παλμούς: "
              + ("ΜΕΤΑΚΙΝΕΙΤΑΙ = τυχαίο" if uniq >= len(ls) - 1
                 else "ΕΠΑΝΑΛΑΜΒΑΝΕΤΑΙ — αξίζει διερεύνηση"))
    tot_above = sum(o[l]["n_above_null_max"] for o in res
                    for l in ("OA vs SB", "OB vs SA"))
    tot_pos = sum(o["J"]["n_positive_J_nonzero"] for o in res)
    print(f"\n  σύνολο lag πάνω από null max (5 παλμοί × 2 ζεύγη × 111): {tot_above}")
    print(f"  σύνολο k≠0 με J>0 (5 παλμοί × 106): {tot_pos}")

    with open("stability_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nΑποθηκεύτηκε: stability_results.json")


if __name__ == "__main__":
    main()
