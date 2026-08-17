"""
Μηδενική κατανομή με ΤΑΙΡΙΑΣΜΕΝΟ bias (permutation test).

Χρήση:  python3 null_matched.py raw_ibm_marrakesh_524288.npy [πλήθος]

ΓΙΑΤΙ: ο ενσωματωμένος έλεγχος του analyse_raw.py χρησιμοποιεί κρυπτογραφικά
τυχαία bits με ρυθμό 0,5. Τα πραγματικά δεδομένα έχουν readout bias (π.χ.
0,4942), άρα score κάτω από την οροφή 1,0 — ενώ τα αμερόληπτα δεδομένα
ελέγχου κολλάνε ΣΤΗΝ οροφή και δίνουν πάντα κέρδος ακριβώς 0. Η σύγκριση
είναι άκυρη.

Εδώ ανακατεύουμε ΤΑ ΙΔΙΑ ΤΑ BITS. Το πλήθος των άσσων μένει ταυτόσημο, άρα
το bias είναι ταυτόσημο — αλλά κάθε χρονική δομή καταστρέφεται. Αυτό είναι
το σωστό «τίποτα» για σύγκριση.
"""
import sys, math
import numpy as np
import scratch_test as T


def main(path, n_null=5, stride_samples=800):
    bits = np.load(path).astype(np.int8)
    n2 = 1 << int(math.log2(len(bits)))
    bits = bits[:n2]

    print(f"Αρχείο: {path}")
    print(f"Bits: {n2:,}   ρυθμός 1: {bits.mean():.6f}")
    print(f"Άσσοι: {int(bits.sum()):,}  (μένουν ΑΚΡΙΒΩΣ ίδιοι σε κάθε ανακάτεμα)\n")

    base, rows = T.search(bits, stride_samples=stride_samples)
    obs = base - rows[0][3]
    print(f"ΠΑΡΑΤΗΡΟΥΜΕΝΟ κέρδος: {obs:.2f} bits   (νικητής: {rows[0][0]})\n")

    print(f"--- μηδενική κατανομή: {n_null} τυχαία ανακατέματα των ΙΔΙΩΝ bits ---")
    null = []
    for i in range(n_null):
        rng = np.random.default_rng(90000 + i)
        x = bits[rng.permutation(n2)]
        b2, r2 = T.search(x, stride_samples=stride_samples)
        g = b2 - r2[0][3]
        null.append(g)
        print(f"    ανακάτεμα {i+1}: ρυθμός {x.mean():.6f}  κέρδος {g:8.2f}  ({r2[0][0]})")

    null = np.array(null)
    print(f"\n    μηδενικό κέρδος: μέσος {null.mean():.2f}, "
          f"max {null.max():.2f}, τυπ.απόκλιση {null.std(ddof=1):.2f}")

    print("\n--- ΕΤΥΜΗΓΟΡΙΑ (με ταιριασμένο bias) ---")
    n_ge = int((null >= obs).sum())
    p = (n_ge + 1) / (len(null) + 1)          # συντηρητικό p permutation test
    print(f"    ανακατέματα με κέρδος >= παρατηρούμενο: {n_ge}/{len(null)}   p >= {p:.3f}")
    if obs > null.max() and obs > 50:
        print("    ΣΗΜΑ. Ξεπερνά και τη μηδενική κατανομή και το απόλυτο όριο.")
    else:
        print("    ΜΗΔΕΝ. Το παρατηρούμενο κέρδος είναι μέσα στον θόρυβο.")
        print("    Το bias του readout ΔΕΝ παρήγαγε ψευδές σήμα.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Δώσε το αρχείο .npy")
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
