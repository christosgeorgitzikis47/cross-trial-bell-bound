"""
ΔΙΑΓΝΩΣΤΙΚΟ (μετά τα δεδομένα — ΔΕΝ είναι το προκαθορισμένο κριτήριο).

Παρατήρηση: οι κορυφές OA vs SB των 5 ΝΕΩΝ παλμών είναι όλες σε αρνητικό lag
και μαζεμένες: -39, -41, -39, -46, -32.

Δύο πράγματα πρέπει να ξεχωρίσουν:
  (α) ΣΗΜΕΙΑΚΗ ΣΥΜΠΤΩΣΗ — ο argmax 5 ανεξάρτητων θορυβωδών καμπυλών έτυχε
      να πέσει κοντά. Τότε η υπόλοιπη καμπύλη είναι επίπεδη.
  (β) ΠΛΑΤΙΑ ΔΟΜΗ — ολόκληρη η περιοχή αρνητικών lag είναι ανυψωμένη.
      Τότε ο argmax είναι απλώς η μύτη ενός λόφου, και το φαινόμενο
      είναι πραγματικό (αν και όχι απαραίτητα φυσικό).

Μετράμε το μέσο MI σε ζώνες lag ανά παλμό. Χωρίς null εδώ: όλοι οι παλμοί
έχουν ίδιο n, οπότε τα ωμά MI συγκρίνονται μεταξύ ΖΩΝΩΝ του ίδιου παλμού.
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
    for grp, rounds in [("ΠΑΛΙΟΙ", OLD), ("ΝΕΟΙ", NEW)]:
        for r in rounds:
            print(f"  γύρος {r} …", flush=True)
            res[str(r)] = curves(r)
    json.dump({"lags": LAGSET, "curves": res},
              open(os.path.join(HERE, "diagnostiko_curves.json"), "w"))

    for label in ("OA vs SB", "OB vs SA"):
        print("\n" + "=" * 84)
        print(f"{label} — μέσο MI ανά ζώνη lag  (×1e-7 bits)")
        print("=" * 84)
        print(f"{'ομάδα':>8} {'γύρος':>7} {'ζώνη -46..-32':>15} {'άλλα αρνητ.':>13} "
              f"{'θετικά':>10} {'μακρινά':>10} {'λόγος':>8}")
        for grp, rounds in [("ΠΑΛΙΟΙ", OLD), ("ΝΕΟΙ", NEW)]:
            for r in rounds:
                z = zones(res[str(r)][label])
                ratio = z["neg_32_46"] / ((z["neg_rest"] + z["pos_all"]) / 2)
                print(f"{grp:>8} {r:>7} {z['neg_32_46']*1e7:>15.3f} "
                      f"{z['neg_rest']*1e7:>13.3f} {z['pos_all']*1e7:>10.3f} "
                      f"{z['far']*1e7:>10.3f} {ratio:>8.2f}")

    # --- πόσο απίθανη είναι η συσσωμάτωση, ΩΣ POST-HOC στατιστικό; ---
    rng = np.random.default_rng(7)
    obs = np.array([-39, -41, -39, -46, -32])
    w = obs.max() - obs.min()                 # εύρος = 14
    L = len(LAGSET)
    idx = rng.integers(0, L, size=(400_000, 5))
    ks = np.array(LAGSET)[idx]
    inside = np.abs(ks) <= 50                 # το εύρος ορίζεται μόνο στα κοντινά
    rng_w = ks.max(axis=1) - ks.min(axis=1)
    p_range = float(((rng_w <= w) & inside.all(axis=1)).mean())
    p_allneg = float((ks < 0).all(axis=1).mean())
    print(f"\nPOST-HOC (ΔΕΝ είναι το προκαθορισμένο κριτήριο):")
    print(f"  P(5 τυχαίες κορυφές όλες κοντινές και εύρος <= {w}) = {p_range:.5f}")
    print(f"  P(5 τυχαίες κορυφές όλες αρνητικές)                = {p_allneg:.5f}")
    print(f"  ΠΡΟΣΟΧΗ: το παράθυρο επιλέχθηκε ΑΦΟΥ είδα τα δεδομένα. Το p είναι")
    print(f"  δείκτης, όχι τεστ. Το ίδιο ισχύει για τα άλλα κανάλια/ομάδες που")
    print(f"  ΔΕΝ ξεχώρισαν — δεν μπαίνουν στον παρονομαστή.")


if __name__ == "__main__":
    main()
