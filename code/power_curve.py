"""
Καμπύλη ισχύος με ΤΙΣ ΡΥΘΜΙΣΕΙΣ ΤΗΣ ΚΥΡΙΑΣ ΑΝΑΛΥΣΗΣ.

    N = 524.288      bias = 0,4942 (ακριβώς 259.094 άσσοι)
    stride_samples = 800
    κατώφλι = permutation null με ταιριασμένο bias (ΟΧΙ το σταθερό 50)

Τρεις τύποι δομής: περιοδική, Thue-Morse, Markov 1ης τάξης.

ΔΥΟ ΚΟΜΜΑΤΙΑ ΙΣΧΥΟΣ — μετρώνται χωριστά:

  (α) ΙΣΧΥΣ ΥΠΟ ΣΥΝΘΗΚΗ: πόσο λεπτή δομή ανιχνεύεται ΟΤΑΝ η σωστή
      αναδιάταξη είναι μέσα στο δείγμα των 800. Εδώ το ανακάτεμα
      καρφώνεται σε stride που ΕΙΝΑΙ στο δείγμα, ώστε να μετρηθεί
      καθαρά η ευαισθησία του ανιχνευτή.

  (β) ΚΑΛΥΨΗ: η πιθανότητα να είναι όντως εκεί. Στα N=524.288 τα
      έγκυρα stride είναι 262.143, άρα 800/262.143 = 0,305%.

Η συνολική ισχύς κατά προσέγγιση = (α) × (β). Το (β) είναι ο κυρίαρχος
περιορισμός στο τρέχον στήσιμο, όχι η ευαισθησία του ανιχνευτή.
"""
import sys, json, math, time
import numpy as np
from math import gcd
import scratch_test as T

N = 1 << 19                 # 524.288
TARGET_ONES = 259094        # ακριβώς όσοι στα πραγματικά δεδομένα -> ίδιο bias
STRIDE_SAMPLES = 800
NOISE = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
N_NULL = 20


def h2(p):
    if p <= 0: return 0.0
    if p >= 1: return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def match_bias(bits, rng, target=TARGET_ONES):
    """Φέρνει το πλήθος των άσσων ΑΚΡΙΒΩΣ στο target, γυρίζοντας τυχαία bits.

    Έτσι τα συνθετικά δεδομένα έχουν ΤΑΥΤΟΣΗΜΟ bias με τα πραγματικά, άρα
    η μηδενική κατανομή από ανακάτεμα ισχύει αυτούσια και για τα δύο."""
    bits = bits.copy().astype(np.int8)
    ones = int(bits.sum())
    if ones > target:
        idx = np.flatnonzero(bits == 1)
        bits[rng.choice(idx, ones - target, replace=False)] = 0
    elif ones < target:
        idx = np.flatnonzero(bits == 0)
        bits[rng.choice(idx, target - ones, replace=False)] = 1
    return bits


# ---------- οι τρεις τύποι δομής ----------

def make_periodic(n, p, rng, period=12):
    """Περιοδική: τυχαίο μοτίβο μήκους `period`, επαναλαμβανόμενο, + θόρυβος.

    Περίοδος 12 και όχι μεγαλύτερη: ο εκτιμητής εντροπίας του scratch_test
    φτάνει μέχρι τάξη kmax=12, οπότε περίοδο 1024 ΔΕΝ θα μπορούσε καν να
    δει — θα μετρούσαμε την τυφλότητα του εργαλείου, όχι την ισχύ του."""
    pat = rng.integers(0, 2, period).astype(np.int8)
    seq = np.tile(pat, n // period + 1)[:n]
    return (seq ^ (rng.uniform(size=n) < p)).astype(np.int8)


def make_thue_morse(n, p, rng):
    """Thue-Morse: η ευκολότερη δυνατή δομή, + θόρυβος."""
    i = np.arange(n); out = np.zeros(n, dtype=np.int8)
    for b in range(24):
        out ^= ((i >> b) & 1).astype(np.int8)
    return (out ^ (rng.uniform(size=n) < p)).astype(np.int8)


def make_markov(n, p, rng):
    """Markov 1ης τάξης («κολλώδης»): x[i] = x[i-1] XOR Bernoulli(p).
    Ρυθμός εντροπίας ΑΚΡΙΒΩΣ h2(p)."""
    flips = (rng.uniform(size=n) < p).astype(np.int8)
    flips[0] = rng.integers(0, 2)
    return np.cumsum(flips) % 2


MAKERS = [("περιοδική P=12", make_periodic),
          ("Thue-Morse", make_thue_morse),
          ("Markov 1ης τάξης", make_markov)]


def sampled_strides(n, samples):
    """ΑΚΡΙΒΩΣ το ίδιο δείγμα stride που παράγει το scratch_test.perm_stride."""
    valid = [a for a in range(3, n, 2) if gcd(a, n) == 1]
    rng = np.random.default_rng(0)
    if len(valid) > samples:
        valid = list(rng.choice(valid, samples, replace=False))
    return [int(a) for a in valid], len(valid)


def main():
    t0 = time.time()
    print(f"N = {N:,}   άσσοι = {TARGET_ONES:,}  (bias {TARGET_ONES/N:.6f})")
    print(f"stride_samples = {STRIDE_SAMPLES}\n")

    all_valid = (N // 2) - 1
    samp, _ = sampled_strides(N, STRIDE_SAMPLES)
    coverage = len(samp) / all_valid
    # ανακάτεμα που ΕΓΓΥΗΜΕΝΑ αναιρείται από stride μέσα στο δείγμα
    s_target = samp[len(samp) // 2]
    A = pow(s_target, -1, N)
    print(f"έγκυρα stride: {all_valid:,}   στο δείγμα: {len(samp)}"
          f"   ΚΑΛΥΨΗ: {coverage*100:.3f}%")
    print(f"ανακάτεμα a={A}  (αναιρείται από stride a={s_target}, ΕΙΝΑΙ στο δείγμα)\n")

    idx = (A * np.arange(N)) % N

    # ---------- κατώφλι από permutation null με ταιριασμένο bias ----------
    print(f"--- ΚΑΤΩΦΛΙ: permutation null, {N_NULL} ανακατέματα ---")
    rng = np.random.default_rng(4242)
    base_bits = match_bias(np.zeros(N, dtype=np.int8), rng)
    null = []
    for i in range(N_NULL):
        r = np.random.default_rng(90000 + i)
        x = base_bits[r.permutation(N)]
        b, rows = T.search(x, stride_samples=STRIDE_SAMPLES)
        null.append(b - rows[0][3])
        print(f"    {i+1:2d}/{N_NULL}  κέρδος {null[-1]:7.2f}   ({time.time()-t0:.0f}s)")
    null = np.array(null)
    thr = float(null.max())
    print(f"\n    μέσος {null.mean():.2f}   τυπ.απόκλιση {null.std(ddof=1):.2f}"
          f"   max {null.max():.2f}")
    print(f"    95ο εκατοστημόριο {np.percentile(null,95):.2f}")
    print(f"    ΚΑΤΩΦΛΙ ΑΝΙΧΝΕΥΣΗΣ = {thr:.2f} bits  (max του null)")
    print(f"    [το παλιό σταθερό κατώφλι ήταν 50 — {50/max(thr,1e-9):.0f}× αυστηρότερο]\n")

    # ---------- καμπύλη ισχύος ανά τύπο δομής ----------
    out = {"N": N, "ones": TARGET_ONES, "stride_samples": STRIDE_SAMPLES,
           "coverage": coverage, "scramble_a": A, "undo_stride": s_target,
           "null": {"values": [float(v) for v in null], "max": thr,
                    "mean": float(null.mean()), "sd": float(null.std(ddof=1))},
           "curves": {}}

    for label, maker in MAKERS:
        print(f"--- {label} ---")
        print(f"    {'θόρυβος':>8} {'h2(θορ.)':>9} {'score φυσ.':>11} "
              f"{'κέρδος':>12} {'ανιχνεύθηκε':>13}")
        rows_out, last_det_h = [], None
        for p in NOISE:
            g = np.random.default_rng(777)
            seq = maker(N, p, g)
            seq = match_bias(seq, g)
            # score() = min(συμπίεση, εντροπία) — ΤΟ ΙΔΙΟ μέτρο με το κύριο τεστ
            h_meas = T.score(seq)                     # στη ΦΥΣΙΚΗ σειρά
            b, rows = T.search(seq[idx], stride_samples=STRIDE_SAMPLES)
            gain = b - rows[0][3]
            det = gain > thr
            if det:
                last_det_h = h_meas
            print(f"    {p:8.2f} {h2(p):9.4f} {h_meas:11.4f} {gain:12.1f} "
                  f"{('ΝΑΙ' if det else 'οχι'):>13}   ({time.time()-t0:.0f}s)")
            rows_out.append({"noise": p, "h2_noise": h2(p), "score_natural": float(h_meas),
                             "gain": float(gain), "detected": bool(det),
                             "winner": rows[0][0]})
        out["curves"][label] = {"rows": rows_out,
                                "max_detectable_score": last_det_h}
        if last_det_h is None:
            print(f"    ΚΑΤΩΦΛΙ: καμία ανίχνευση σε κανένα επίπεδο\n")
        else:
            print(f"    ΚΑΤΩΦΛΙ: ανιχνεύεται δομή μέχρι score "
                  f"{last_det_h:.4f}\n")

    print("=== ΣΥΝΟΨΗ ===")
    print(f"κατώφλι ανίχνευσης (permutation null, ταιριασμένο bias): {thr:.2f} bits")
    for label, _ in MAKERS:
        h = out["curves"][label]["max_detectable_score"]
        print(f"  {label:22s} ανιχνεύσιμο μέχρι score = "
              f"{f'{h:.4f}' if h is not None else 'καμία ανίχνευση'}")
    print(f"\nΤΑ ΠΑΡΑΠΑΝΩ ΕΙΝΑΙ ΥΠΟ ΣΥΝΘΗΚΗ ότι η σωστή αναδιάταξη είναι στο δείγμα.")
    print(f"Η πιθανότητα να ισχύει αυτό είναι η ΚΑΛΥΨΗ: {coverage*100:.3f}%.")

    with open("power_curve_524288.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nΑποθηκεύτηκε: power_curve_524288.json   ({time.time()-t0:.0f}s συνολικά)")


if __name__ == "__main__":
    main()
