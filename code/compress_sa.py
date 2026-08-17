"""
Τεστ συμπίεσης ΜΟΝΟ στη σειρά ρυθμίσεων SA.

ΓΙΑΤΙ ΜΟΝΟ ΤΟ SA: είναι 50/50 ισορροπημένο, άρα ο συμπιεστής δουλεύει
κανονικά — σε αντίθεση με το OA (bias 0,69%), όπου το score είναι σχεδόν
ολόκληρο το bias και ο συμπιεστής σχεδόν τυφλός.

ΤΙ ΣΗΜΑΙΝΕΙ ΑΝ ΒΡΕΘΕΙ ΔΟΜΗ: το SA παράγεται από τη ΓΕΝΝΗΤΡΙΑ ΕΠΙΛΟΓΗΣ
ΡΥΘΜΙΣΕΩΝ, όχι από κβαντική μέτρηση. Δομή εδώ είναι εύρημα για τη
γεννήτρια — ΟΧΙ για την κβαντική φυσική.

Δείγμα: τα ΠΡΩΤΑ 2.000.000 trials, ΣΥΝΕΧΟΜΕΝΑ. Τυχαία δειγματοληψία θα
κατέστρεφε ακριβώς τη χρονική σειρά που ελέγχουμε.

Το bit-reversal αποκλείεται αυτόματα: το perm_bitrev του scratch_test
επιστρέφει κενό όταν το n δεν είναι δύναμη του 2 (2.000.000 δεν είναι).
"""
import argparse, json
import numpy as np
import scratch_test as T

N_SUB = 2_000_000
N_NULL = 8
STRIDE_SAMPLES = 800


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--n", type=int, default=N_SUB)
    ap.add_argument("--nulls", type=int, default=N_NULL)
    ap.add_argument("--strides", type=int, default=STRIDE_SAMPLES)
    a = ap.parse_args()

    d = np.load(a.path)
    SA = d['SA'][:a.n]
    bits = (SA - 1).astype(np.int8)          # {1,2} -> {0,1}
    n = len(bits)
    ones = int(bits.sum())
    print(f"Σειρά: SA (ρυθμίσεις Alice), πρώτα {n:,} trials ΣΥΝΕΧΟΜΕΝΑ")
    print(f"ρυθμός 1: {bits.mean():.6f}   άσσοι: {ones:,}")

    fam = list(T.family(n, a.strides))
    has_bitrev = any(f[0] == "bitreverse" for f in fam)
    print(f"αναδιατάξεις: {len(fam)}   bit-reversal στην οικογένεια: "
          f"{'ΝΑΙ (πρόβλημα!)' if has_bitrev else 'ΟΧΙ (σωστό)'}\n")

    print("--- βασικό score (φυσική σειρά) ---")
    base_score = T.score(bits)
    print(f"    {base_score:.6f} bits/σύμβολο\n")

    print("--- αναζήτηση ---")
    base, rows = T.search(bits, stride_samples=a.strides)
    gain = base - rows[0][3]
    for name, s, cost, tot in rows[:5]:
        print(f"    {name:22s} score {s/n:.6f}  κόστος π {cost:7.1f}  σύνολο {tot:12.1f}")
    print(f"\n    ΚΕΡΔΟΣ: {gain:.2f} bits\n")

    print(f"--- permutation null (ταιριασμένο bias), {a.nulls} ανακατέματα ---")
    null = []
    for i in range(a.nulls):
        rng = np.random.default_rng(50000 + i)
        x = bits[rng.permutation(n)]
        b2, r2 = T.search(x, stride_samples=a.strides)
        g = b2 - r2[0][3]
        null.append(g)
        print(f"    {i+1}/{a.nulls}: κέρδος {g:8.2f}   ({r2[0][0]})")
    null = np.array(null)
    thr = float(null.max())
    print(f"\n    μέσος {null.mean():.2f}  τυπ.απόκλιση {null.std(ddof=1):.2f}  max {thr:.2f}")
    print(f"    ΚΑΤΩΦΛΙ = {thr:.2f} bits")

    print("\n--- ΕΤΥΜΗΓΟΡΙΑ ---")
    if gain > thr:
        print(f"    ΣΗΜΑ στο SA (κέρδος {gain:.2f} > κατώφλι {thr:.2f}).")
        print("    ΠΡΟΣΟΧΗ: αυτό αφορά τη ΓΕΝΝΗΤΡΙΑ ΕΠΙΛΟΓΗΣ ΡΥΘΜΙΣΕΩΝ,")
        print("    ΟΧΙ κβαντική φυσική. Το SA δεν είναι αποτέλεσμα μέτρησης.")
    else:
        print(f"    ΜΗΔΕΝ (κέρδος {gain:.2f} <= κατώφλι {thr:.2f}).")
        print("    Καμία ανιχνεύσιμη δομή στη γεννήτρια επιλογής ρυθμίσεων.")

    with open("compress_sa_results.json", "w") as f:
        json.dump({"n": n, "rate1": float(bits.mean()), "base_score": float(base_score),
                   "gain": float(gain), "threshold": thr,
                   "null": [float(v) for v in null],
                   "winner": rows[0][0], "n_perms": len(fam),
                   "bitrev_present": has_bitrev}, f, indent=2)
    print("\nΑποθηκεύτηκε: compress_sa_results.json")


if __name__ == "__main__":
    main()
