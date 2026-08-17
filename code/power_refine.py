"""
Εκλέπτυνση της καμπύλης ισχύος κοντά στο σημείο θραύσης.

Η πρώτη σάρωση (power_curve.py) ανίχνευσε ΚΑΘΕ επίπεδο θορύβου μέχρι 45%,
οπότε δεν βρήκε πού σπάει το τεστ. Εδώ σαρώνουμε 46%..49,9% θόρυβο, όπου
ο ρυθμός εντροπίας πλησιάζει το 1,0.

Ίδιες ρυθμίσεις: N=524.288, ακριβώς 259.094 άσσοι, stride_samples=800.
Κατώφλι από το ήδη υπολογισμένο permutation null.
"""
import json, math, time
import numpy as np
import scratch_test as T
from power_curve import (N, TARGET_ONES, STRIDE_SAMPLES, MAKERS,
                         match_bias, sampled_strides, h2)

NOISE = [0.46, 0.47, 0.48, 0.49, 0.495, 0.498, 0.499]


def main():
    t0 = time.time()
    prev = json.load(open("power_curve_524288.json"))
    thr = prev["null"]["max"]
    A, s_target = prev["scramble_a"], prev["undo_stride"]
    idx = (A * np.arange(N)) % N

    print(f"N = {N:,}   άσσοι = {TARGET_ONES:,}   stride_samples = {STRIDE_SAMPLES}")
    print(f"ΚΑΤΩΦΛΙ (permutation null, max από {len(prev['null']['values'])}): "
          f"{thr:.2f} bits")
    print(f"ανακάτεμα a={A}, αναιρείται από stride a={s_target}\n")

    out = {}
    for label, maker in MAKERS:
        print(f"--- {label} ---")
        print(f"    {'θόρυβος':>8} {'h2(θορ.)':>9} {'score φυσ.':>11} "
              f"{'κέρδος':>12} {'ανιχνεύθηκε':>13}")
        rows_out = []
        # ξεκινάμε από τα ήδη γνωστά σημεία της πρώτης σάρωσης
        for r in prev["curves"][label]["rows"]:
            rows_out.append(r)
        for p in NOISE:
            g = np.random.default_rng(777)
            seq = maker(N, p, g)
            seq = match_bias(seq, g)
            h_meas = T.score(seq)
            b, rows = T.search(seq[idx], stride_samples=STRIDE_SAMPLES)
            gain = b - rows[0][3]
            det = gain > thr
            print(f"    {p:8.3f} {h2(p):9.5f} {h_meas:11.5f} {gain:12.1f} "
                  f"{('ΝΑΙ' if det else 'οχι'):>13}   ({time.time()-t0:.0f}s)")
            rows_out.append({"noise": p, "h2_noise": h2(p),
                             "score_natural": float(h_meas), "gain": float(gain),
                             "detected": bool(det), "winner": rows[0][0]})
        # σημείο θραύσης: τελευταίο ανιχνευμένο και πρώτο μη ανιχνευμένο
        det_rows = [r for r in rows_out if r["detected"]]
        und_rows = [r for r in rows_out if not r["detected"]]
        last_det = max(det_rows, key=lambda r: r["noise"]) if det_rows else None
        first_und = min(und_rows, key=lambda r: r["noise"]) if und_rows else None
        out[label] = {"rows": rows_out, "last_detected": last_det,
                      "first_undetected": first_und}
        if first_und is None:
            print(f"    ΔΕΝ ΕΣΠΑΣΕ ακόμη: ανιχνεύεται και στο 49,9% θόρυβο "
                  f"(score {last_det['score_natural']:.5f})\n")
        else:
            print(f"    ΣΗΜΕΙΟ ΘΡΑΥΣΗΣ μεταξύ θορύβου {last_det['noise']:.3f} "
                  f"και {first_und['noise']:.3f}")
            print(f"    -> ανιχνεύσιμο μέχρι score {last_det['score_natural']:.5f}"
                  f"  (κέρδος {last_det['gain']:.1f} έναντι κατωφλίου {thr:.2f})\n")

    print("=== ΣΥΝΟΨΗ ΚΑΤΩΦΛΙΩΝ ===")
    print(f"κατώφλι ανίχνευσης: {thr:.2f} bits (permutation null, ταιριασμένο bias)")
    print(f"{'τύπος δομής':24s} {'μέγιστο ανιχν. score':>21} {'θόρυβος θραύσης':>18}")
    for label, _ in MAKERS:
        d = out[label]
        ld, fu = d["last_detected"], d["first_undetected"]
        sc = f"{ld['score_natural']:.5f}" if ld else "—"
        br = (f"{ld['noise']:.3f}–{fu['noise']:.3f}" if fu else ">0.499")
        print(f"{label:24s} {sc:>21} {br:>18}")

    with open("power_refine_524288.json", "w") as f:
        json.dump({"threshold": thr, "curves": out}, f, indent=2, ensure_ascii=False)
    print(f"\nΑποθηκεύτηκε: power_refine_524288.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
