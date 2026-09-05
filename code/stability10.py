"""
STEP A - Temporal independence: a lag test and J(k) over 10 pulses.

5 OLD : 28293-28297, consecutive within 65 minutes of one day (22 Aug 2025).
        Parametrically IDENTICAL -- one calibration block.
5 NEW : 1000, 15000, 22000, 23000, 26000 -- spread over 20 months.
        They differ in the DERIVED CALIBRATION values (beta, gain, pefs,
        nTrialsNeeded, seedLength) -- unavoidable: the calibration changes
        when the day changes. The PROTOCOL CONFIGURATION is identical
        (isQuantum, stoppingCriteria, nBitsOut, nBitsThreshold, epsilonBias,
        errorExtractor, errorSmoothness).
        -> SEPARATE TABLES.

CRITERION FIXED BEFORE THE DATA (not changed afterwards):
  1 pair at the same lag        -> expected by chance, NOT a finding
  >=3 pulses at the same lag    -> worth investigating
  the same lag in BOTH channels -> a real signal

The same settings as stability.py: 500 shuffles, the same 111 lags, the same
107 values of k for J, the same seed (4242 + round).
"""
import json, os, sys
import numpy as np

from stability import analyse, ROUNDS as OLD_ROUNDS
from lag_dense import LAGS
from j_curve import LAGS as JLAGS

HERE = os.path.dirname(os.path.abspath(__file__))
NEW_ROUNDS = [1000, 15000, 22000, 23000, 26000]
MANIFEST = os.path.join(HERE, "..", "dedomena_curby", "manifest.json")
N_LAGS = len(sorted(LAGS))            # 111 possible peak positions per channel
CHANNELS = ("OA vs SB", "OB vs SA")
N_MC = 400_000                        # Monte Carlo for the birthday numbers


# ----------------------------------------------------------------- birthday
def birthday(m, L=N_LAGS, n_ch=len(CHANNELS), n_mc=N_MC, seed=99):
    """Under the null the peaks fall UNIFORMLY over L positions, independently
    per pulse and per channel. Returns the probabilities of the three criteria
    for m pulses."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, L, size=(n_mc, n_ch, m))
    # largest multiplicity per channel
    mult = np.zeros((n_mc, n_ch), dtype=np.int64)
    for ch in range(n_ch):
        cnt = np.zeros((n_mc, L), dtype=np.int16)
        np.add.at(cnt, (np.repeat(np.arange(n_mc), m), x[:, ch, :].ravel()), 1)
        mult[:, ch] = cnt.max(axis=1)
        if ch == 0:
            cnt0 = cnt
        else:
            cnt1 = cnt
    both = ((cnt0 >= 2) & (cnt1 >= 2)).any(axis=1)     # same lag, BOTH channels
    return {
        "m": m, "L": L,
        "exp_pairs_per_channel": m * (m - 1) / 2 / L,
        "p_any_pair_per_channel": float((mult[:, 0] >= 2).mean()),
        "p_any_pair_either_channel": float((mult >= 2).any(axis=1).mean()),
        "p_triple_per_channel": float((mult[:, 0] >= 3).mean()),
        "p_triple_either_channel": float((mult >= 3).any(axis=1).mean()),
        "p_both_channels_same_lag": float(both.mean()),
    }


# -------------------------------------------------------------------- table
def load_dates():
    man = json.load(open(MANIFEST))
    return {p["round"]: p["stages"]["request"]["timestamp"][:10]
            for p in man["pulses"]}


def table(title, res, dates, extra=None):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    head = (f"{'round':>6} {'date':>12} | {'OA vs SB':>17} | "
            f"{'OB vs SA':>17} | {'J(0)':>8} {'J/sig':>7} {'k!=0,J>0':>9}")
    print(head)
    print(f"{'':>6} {'':>12} | {'max sig':>8}{'@lag':>9} | {'max sig':>8}{'@lag':>9} |")
    print("-" * 92)
    for o in res:
        a, b = o["OA vs SB"], o["OB vs SA"]
        print(f"{o['round']:>6} {dates.get(o['round'],'?'):>12} | "
              f"{a['max_sigma']:>+8.2f}{a['max_sigma_lag']:>9} | "
              f"{b['max_sigma']:>+8.2f}{b['max_sigma_lag']:>9} | "
              f"{o['J']['J0']:>+8,} {o['J']['z0']:>+7.2f} "
              f"{o['J']['n_positive_J_nonzero']:>4}/{len(sorted(JLAGS))-1}")
    if extra:
        print("\n  derived calibration values (hence a separate table):")
        print(f"  {'round':>6} {'beta':>10} {'gain':>10} {'nTrialsNeeded':>14} "
              f"{'seedLength':>11}")
        for o in res:
            e = extra.get(o["round"], {})
            print(f"  {o['round']:>6} {e.get('beta',0):>10.6f} "
                  f"{e.get('gain',0):>10.7f} {e.get('nTrialsNeeded',0):>14,} "
                  f"{e.get('seedLength',0):>11,}")


def main():
    dates = load_dates()

    # ---- old: from the existing stability_results.json, else compute ----
    old_path = os.path.join(HERE, "stability_results.json")
    if os.path.exists(old_path):
        old = json.load(open(old_path))
        print(f"Old 5: from {old_path} (already computed)")
    else:
        old = [analyse(r) for r in OLD_ROUNDS]

    # ---- new ----
    new_path = os.path.join(HERE, "stability10_new.json")
    if os.path.exists(new_path):
        new = json.load(open(new_path))
        print(f"New 5: from {new_path}")
    else:
        new = []
        for r in NEW_ROUNDS:
            print(f"--- round {r} ---", flush=True)
            o = analyse(r)
            new.append(o)
            print(f"    OA vs SB: {o['OA vs SB']['max_sigma']:+.2f} sig @ "
                  f"{o['OA vs SB']['max_sigma_lag']}   "
                  f"OB vs SA: {o['OB vs SA']['max_sigma']:+.2f} sig @ "
                  f"{o['OB vs SA']['max_sigma_lag']}   "
                  f"J(0)={o['J']['J0']:+,} ({o['J']['z0']:+.2f} sig)", flush=True)
            json.dump(new, open(new_path, "w"), indent=2)

    # ---- derived calibration values for the second table ----
    cache = json.load(open(os.path.join(HERE, "..", "dedomena_curby",
                                        "xartografisi_cache.json")))
    extra = {}
    for rec in cache.values():
        if rec.get("ok"):
            p = rec["parameters"]
            extra[rec["round"]] = {k: p[k] for k in
                                   ("beta", "gain", "nTrialsNeeded", "seedLength")}

    table("TABLE 1 - 5 CONSECUTIVE PULSES, 65 MINUTES, ONE CALIBRATION",
          old, dates)
    table("TABLE 2 - 5 SPREAD PULSES, 20 MONTHS, DIFFERENT CALIBRATIONS",
          new, dates, extra)

    # ---- repetition of lags ----
    allr = old + new
    print("\n" + "=" * 92)
    print("DOES ANY LAG REPEAT ACROSS THE 10?")
    print("=" * 92)
    bd10 = birthday(10)
    bd5 = birthday(5)
    print(f"Space: {N_LAGS} lag positions per channel, 2 channels, 10 pulses.")
    print(f"Under chance - expected pairs at the same lag per channel: "
          f"{bd10['exp_pairs_per_channel']:.3f}")
    print(f"  P(>=1 pair in one channel)        = {bd10['p_any_pair_per_channel']:.3f}")
    print(f"  P(>=1 pair in EITHER channel)     = {bd10['p_any_pair_either_channel']:.3f}")
    print(f"  P(>=3 pulses same lag, 1 channel) = {bd10['p_triple_per_channel']:.4f}")
    print(f"  P(>=3 in EITHER channel)          = {bd10['p_triple_either_channel']:.4f}")
    print(f"  P(same lag in BOTH channels)      = {bd10['p_both_channels_same_lag']:.4f}")

    verdict = {}
    for ch in CHANNELS:
        lags = [o[ch]["max_sigma_lag"] for o in allr]
        rounds = [o["round"] for o in allr]
        uniq, cnts = np.unique(lags, return_counts=True)
        rep = {int(u): [rounds[i] for i, l in enumerate(lags) if l == u]
               for u, c in zip(uniq, cnts) if c >= 2}
        print(f"\n  {ch}")
        print(f"    peaks (old then new): {lags}")
        print(f"    distinct lags: {len(uniq)} / 10")
        if not rep:
            print(f"    NO REPETITION  (expected by chance: "
                  f"{bd10['exp_pairs_per_channel']:.3f} pairs; "
                  f"P(>=1) = {bd10['p_any_pair_per_channel']:.3f})")
        for k, rs in rep.items():
            n = len(rs)
            if n == 2:
                tag = (f"1 pair -> EXPECTED BY CHANCE, NOT A FINDING "
                       f"(P(>=1 pair) = {bd10['p_any_pair_per_channel']:.3f})")
            else:
                tag = (f"{n} pulses -> WORTH INVESTIGATING "
                       f"(P(>=3) = {bd10['p_triple_per_channel']:.4f})")
            print(f"    lag {k}: rounds {rs}  <-- {tag}")
        verdict[ch] = rep

    common = set(verdict[CHANNELS[0]]) & set(verdict[CHANNELS[1]])
    print()
    if common:
        print(f"  !!! SAME LAG IN BOTH CHANNELS: {sorted(common)}  "
              f"-> A REAL SIGNAL by the pre-specified criterion "
              f"(P by chance = {bd10['p_both_channels_same_lag']:.4f})")
    else:
        print(f"  No lag common to both channels. "
              f"(P by chance = {bd10['p_both_channels_same_lag']:.4f})")

    tot_above = sum(o[c]["n_above_null_max"] for o in allr for c in CHANNELS)
    tot_pos = sum(o["J"]["n_positive_J_nonzero"] for o in allr)
    print(f"\n  total lags above the null max (10 x 2 x {N_LAGS} = "
          f"{10*2*N_LAGS}): {tot_above}   "
          f"(expected ~{10*2*N_LAGS/501:.1f} with 500 shuffles)")
    print(f"  total k!=0 with J>0 (10 x {len(sorted(JLAGS))-1} = "
          f"{10*(len(sorted(JLAGS))-1)}): {tot_pos}")

    json.dump({"old": old, "new": new, "birthday_10": bd10, "birthday_5": bd5,
               "repeats": {k: {str(a): b for a, b in v.items()}
                           for k, v in verdict.items()}},
              open(os.path.join(HERE, "stability10_results.json"), "w"), indent=2)
    print("\nSaved: stability10_results.json")


if __name__ == "__main__":
    main()
