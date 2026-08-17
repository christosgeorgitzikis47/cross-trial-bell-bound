"""
ΒΗΜΑ Α — Χρονική ανεξαρτησία: lag test + J(k) σε 10 παλμούς.

5 ΠΑΛΙΟΙ  : 28293–28297, διαδοχικοί μέσα σε 65 λεπτά μιας ημέρας (22 Αυγ 2025).
             Παραμετρικά ΤΑΥΤΟΣΗΜΟΙ — ένα μπλοκ βαθμονόμησης.
5 ΝΕΟΙ    : 1000, 15000, 22000, 23000, 26000 — απλωμένοι σε 20 μήνες.
             Διαφέρουν στα ΠΑΡΑΓΩΓΑ ΒΑΘΜΟΝΟΜΗΣΗΣ (beta, gain, pefs,
             nTrialsNeeded, seedLength) — αναπόφευκτο: αλλάζει η βαθμονόμηση
             όταν αλλάζει η ημέρα. Η ΡΥΘΜΙΣΗ ΠΡΩΤΟΚΟΛΛΟΥ είναι ταυτόσημη
             (isQuantum, stoppingCriteria, nBitsOut, nBitsThreshold,
             epsilonBias, errorExtractor, errorSmoothness).
             -> ΞΕΧΩΡΙΣΤΟΙ ΠΙΝΑΚΕΣ.

ΚΡΙΤΗΡΙΟ ΟΡΙΣΜΕΝΟ ΠΡΙΝ ΤΑ ΔΕΔΟΜΕΝΑ (δεν αλλάζει μετά):
  1 ζευγάρι ίδιου lag        -> αναμενόμενο από τύχη, ΟΧΙ εύρημα
  >=3 παλμοί στο ίδιο lag    -> αξίζει διερεύνηση
  ίδιο lag ΚΑΙ στα δύο κανάλια -> πραγματικό σήμα

Ίδιες ρυθμίσεις με το stability.py: 500 ανακατέματα, ίδια 111 lag,
ίδια 107 k για το J, ίδιος σπόρος (4242 + γύρος).
"""
import json, os, sys
import numpy as np

from stability import analyse, ROUNDS as OLD_ROUNDS
from lag_dense import LAGS
from j_curve import LAGS as JLAGS

HERE = os.path.dirname(os.path.abspath(__file__))
NEW_ROUNDS = [1000, 15000, 22000, 23000, 26000]
MANIFEST = os.path.join(HERE, "..", "dedomena_curby", "manifest.json")
N_LAGS = len(sorted(LAGS))            # 111 θέσεις κορυφής ανά κανάλι
CHANNELS = ("OA vs SB", "OB vs SA")
N_MC = 400_000                        # Monte Carlo για τα birthday νούμερα


# ----------------------------------------------------------------- birthday
def birthday(m, L=N_LAGS, n_ch=len(CHANNELS), n_mc=N_MC, seed=99):
    """Υπό τη μηδενική υπόθεση οι κορυφές πέφτουν ΟΜΟΙΟΜΟΡΦΑ σε L θέσεις,
    ανεξάρτητα ανά παλμό και ανά κανάλι. Επιστρέφει τις πιθανότητες των
    τριών κριτηρίων για m παλμούς."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, L, size=(n_mc, n_ch, m))
    # μέγιστη πολλαπλότητα ανά κανάλι
    mult = np.zeros((n_mc, n_ch), dtype=np.int64)
    for ch in range(n_ch):
        cnt = np.zeros((n_mc, L), dtype=np.int16)
        np.add.at(cnt, (np.repeat(np.arange(n_mc), m), x[:, ch, :].ravel()), 1)
        mult[:, ch] = cnt.max(axis=1)
        if ch == 0:
            cnt0 = cnt
        else:
            cnt1 = cnt
    both = ((cnt0 >= 2) & (cnt1 >= 2)).any(axis=1)     # ίδιο lag, ΚΑΙ τα 2 κανάλια
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
    head = (f"{'γύρος':>6} {'ημερομηνία':>12} | {'OA vs SB':>17} | "
            f"{'OB vs SA':>17} | {'J(0)':>8} {'J/σ':>7} {'k≠0,J>0':>8}")
    print(head)
    print(f"{'':>6} {'':>12} | {'μέγ.σ':>8}{'@lag':>9} | {'μέγ.σ':>8}{'@lag':>9} |")
    print("-" * 92)
    for o in res:
        a, b = o["OA vs SB"], o["OB vs SA"]
        print(f"{o['round']:>6} {dates.get(o['round'],'?'):>12} | "
              f"{a['max_sigma']:>+8.2f}{a['max_sigma_lag']:>9} | "
              f"{b['max_sigma']:>+8.2f}{b['max_sigma_lag']:>9} | "
              f"{o['J']['J0']:>+8,} {o['J']['z0']:>+7.2f} "
              f"{o['J']['n_positive_J_nonzero']:>4}/{len(sorted(JLAGS))-1}")
    if extra:
        print("\n  παράγωγα βαθμονόμησης (γι' αυτό ξεχωριστός πίνακας):")
        print(f"  {'γύρος':>6} {'beta':>10} {'gain':>10} {'nTrialsNeeded':>14} "
              f"{'seedLength':>11}")
        for o in res:
            e = extra.get(o["round"], {})
            print(f"  {o['round']:>6} {e.get('beta',0):>10.6f} "
                  f"{e.get('gain',0):>10.7f} {e.get('nTrialsNeeded',0):>14,} "
                  f"{e.get('seedLength',0):>11,}")


def main():
    dates = load_dates()

    # ---- παλιοί: από το υπάρχον stability_results.json, αλλιώς υπολογισμός ----
    old_path = os.path.join(HERE, "stability_results.json")
    if os.path.exists(old_path):
        old = json.load(open(old_path))
        print(f"Παλιοί 5: από {old_path} (ήδη υπολογισμένοι)")
    else:
        old = [analyse(r) for r in OLD_ROUNDS]

    # ---- νέοι ----
    new_path = os.path.join(HERE, "stability10_new.json")
    if os.path.exists(new_path):
        new = json.load(open(new_path))
        print(f"Νέοι 5: από {new_path}")
    else:
        new = []
        for r in NEW_ROUNDS:
            print(f"--- γύρος {r} ---", flush=True)
            o = analyse(r)
            new.append(o)
            print(f"    OA vs SB: {o['OA vs SB']['max_sigma']:+.2f}σ @ "
                  f"{o['OA vs SB']['max_sigma_lag']}   "
                  f"OB vs SA: {o['OB vs SA']['max_sigma']:+.2f}σ @ "
                  f"{o['OB vs SA']['max_sigma_lag']}   "
                  f"J(0)={o['J']['J0']:+,} ({o['J']['z0']:+.2f}σ)", flush=True)
            json.dump(new, open(new_path, "w"), indent=2)

    # ---- παράγωγα βαθμονόμησης για τον νέο πίνακα ----
    cache = json.load(open(os.path.join(HERE, "..", "dedomena_curby",
                                        "xartografisi_cache.json")))
    extra = {}
    for rec in cache.values():
        if rec.get("ok"):
            p = rec["parameters"]
            extra[rec["round"]] = {k: p[k] for k in
                                   ("beta", "gain", "nTrialsNeeded", "seedLength")}

    table("ΠΙΝΑΚΑΣ 1 — 5 ΔΙΑΔΟΧΙΚΟΙ ΠΑΛΜΟΙ, 65 ΛΕΠΤΑ, ΜΙΑ ΒΑΘΜΟΝΟΜΗΣΗ", old, dates)
    table("ΠΙΝΑΚΑΣ 2 — 5 ΑΠΛΩΜΕΝΟΙ ΠΑΛΜΟΙ, 20 ΜΗΝΕΣ, ΔΙΑΦΟΡΕΤΙΚΕΣ ΒΑΘΜΟΝΟΜΗΣΕΙΣ",
          new, dates, extra)

    # ---- επανάληψη lag ----
    allr = old + new
    print("\n" + "=" * 92)
    print("ΕΠΑΝΑΛΑΜΒΑΝΕΤΑΙ ΚΑΝΕΝΑ LAG ΣΤΟΥΣ 10;")
    print("=" * 92)
    bd10 = birthday(10)
    bd5 = birthday(5)
    print(f"Χώρος: {N_LAGS} θέσεις lag ανά κανάλι, 2 κανάλια, 10 παλμοί.")
    print(f"Υπό τύχη — αναμενόμενα ζευγάρια ίδιου lag ανά κανάλι: "
          f"{bd10['exp_pairs_per_channel']:.3f}")
    print(f"  P(>=1 ζευγάρι σε ένα κανάλι)      = {bd10['p_any_pair_per_channel']:.3f}")
    print(f"  P(>=1 ζευγάρι σε ΟΠΟΙΟΔΗΠΟΤΕ)     = {bd10['p_any_pair_either_channel']:.3f}")
    print(f"  P(>=3 παλμοί ίδιο lag, ένα κανάλι)= {bd10['p_triple_per_channel']:.4f}")
    print(f"  P(>=3 σε ΟΠΟΙΟΔΗΠΟΤΕ κανάλι)      = {bd10['p_triple_either_channel']:.4f}")
    print(f"  P(ίδιο lag ΚΑΙ στα δύο κανάλια)   = {bd10['p_both_channels_same_lag']:.4f}")

    verdict = {}
    for ch in CHANNELS:
        lags = [o[ch]["max_sigma_lag"] for o in allr]
        rounds = [o["round"] for o in allr]
        uniq, cnts = np.unique(lags, return_counts=True)
        rep = {int(u): [rounds[i] for i, l in enumerate(lags) if l == u]
               for u, c in zip(uniq, cnts) if c >= 2}
        print(f"\n  {ch}")
        print(f"    κορυφές (παλιοί→νέοι): {lags}")
        print(f"    διαφορετικά lag: {len(uniq)} / 10")
        if not rep:
            print(f"    ΚΑΜΙΑ ΕΠΑΝΑΛΗΨΗ  (αναμενόμενα από τύχη: "
                  f"{bd10['exp_pairs_per_channel']:.3f} ζευγάρια· "
                  f"P(>=1) = {bd10['p_any_pair_per_channel']:.3f})")
        for k, rs in rep.items():
            n = len(rs)
            if n == 2:
                tag = (f"1 ζευγάρι -> ΑΝΑΜΕΝΟΜΕΝΟ ΑΠΟ ΤΥΧΗ, ΟΧΙ ΕΥΡΗΜΑ "
                       f"(P(>=1 ζευγάρι) = {bd10['p_any_pair_per_channel']:.3f})")
            else:
                tag = (f"{n} παλμοί -> ΑΞΙΖΕΙ ΔΙΕΡΕΥΝΗΣΗ "
                       f"(P(>=3) = {bd10['p_triple_per_channel']:.4f})")
            print(f"    lag {k}: γύροι {rs}  <-- {tag}")
        verdict[ch] = rep

    common = set(verdict[CHANNELS[0]]) & set(verdict[CHANNELS[1]])
    print()
    if common:
        print(f"  !!! ΙΔΙΟ LAG ΚΑΙ ΣΤΑ ΔΥΟ ΚΑΝΑΛΙΑ: {sorted(common)}  "
              f"-> ΠΡΑΓΜΑΤΙΚΟ ΣΗΜΑ κατά το προκαθορισμένο κριτήριο "
              f"(P υπό τύχη = {bd10['p_both_channels_same_lag']:.4f})")
    else:
        print(f"  Κανένα lag κοινό στα δύο κανάλια. "
              f"(P υπό τύχη = {bd10['p_both_channels_same_lag']:.4f})")

    tot_above = sum(o[c]["n_above_null_max"] for o in allr for c in CHANNELS)
    tot_pos = sum(o["J"]["n_positive_J_nonzero"] for o in allr)
    print(f"\n  σύνολο lag πάνω από null max (10 × 2 × {N_LAGS} = "
          f"{10*2*N_LAGS}): {tot_above}   "
          f"(αναμενόμενα ~{10*2*N_LAGS/501:.1f} με 500 ανακατέματα)")
    print(f"  σύνολο k≠0 με J>0 (10 × {len(sorted(JLAGS))-1} = "
          f"{10*(len(sorted(JLAGS))-1)}): {tot_pos}")

    json.dump({"old": old, "new": new, "birthday_10": bd10, "birthday_5": bd5,
               "repeats": {k: {str(a): b for a, b in v.items()}
                           for k, v in verdict.items()}},
              open(os.path.join(HERE, "stability10_results.json"), "w"), indent=2)
    print("\nΑποθηκεύτηκε: stability10_results.json")


if __name__ == "__main__":
    main()
