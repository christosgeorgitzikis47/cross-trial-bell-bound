"""
Το κοφτερό τεστ: συσχέτιση αποτελέσματος με ΜΕΛΛΟΝΤΙΚΗ ρύθμιση.

Η υπόθεση: αν το μέλλον επηρεάζει το παρόν, το αποτέλεσμα στο trial i
συσχετίζεται με τη ρύθμιση στο trial i+k για k > 0.

Το k=0 στο ζεύγος OA vs SA ΠΡΕΠΕΙ να ανάβει — αυτή είναι η κανονική
κβαντομηχανική. Θετικός έλεγχος: αν δεν ανάβει, η φόρτωση είναι λάθος.

ΠΡΟΣΟΧΗ — DEADTIME: αναμένεται ψευδές σήμα στο lag ±1 από deadtime του
ανιχνευτή (ένα click επηρεάζει την επόμενη χρονοθυρίδα). ΔΕΝ είναι
ανακάλυψη. Ενδιαφέρον μόνο αν επιμένει σε |lag| >= 3.

Χρήση:  python3 lag_test.py curby_28297.npz
"""
import argparse
import numpy as np

LAGS = [-1000, -100, -10, -3, -1, 0, 1, 3, 10, 100, 1000]
N_SHUFFLE = 100
NS = 3          # οι ρυθμίσεις παίρνουν τιμές {1,2} -> δείκτες 0..2


def mi(o, s, ns=NS):
    """Αμοιβαία πληροφορία σε bits. bincount αντί για np.add.at (9x γρηγορότερο,
    ταυτόσημο αποτέλεσμα)."""
    c = np.bincount(o.astype(np.intp) * ns + s.astype(np.intp),
                    minlength=2 * ns).reshape(2, ns).astype(float)
    j = c / c.sum()
    po = j.sum(axis=1, keepdims=True)
    ps = j.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = j * np.log2(j / (po * ps))
    return float(np.nansum(t))


def align(outcome, setting, lag):
    if lag > 0:
        return outcome[:-lag], setting[lag:]
    if lag < 0:
        return outcome[-lag:], setting[:lag]
    return outcome, setting


def null_distribution(o, s, n_shuffle=N_SHUFFLE, seed=0):
    """Ανακατεύουμε τις ΡΥΘΜΙΣΕΙΣ -> καταστρέφεται κάθε συσχέτιση, τα
    περιθώρια μένουν ΑΚΡΙΒΩΣ ίδια. Το σωστό «τίποτα»."""
    rng = np.random.default_rng(seed)
    sh = s.copy()
    vals = np.empty(n_shuffle)
    for i in range(n_shuffle):
        rng.shuffle(sh)
        vals[i] = mi(o, sh)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--shuffles", type=int, default=N_SHUFFLE)
    a = ap.parse_args()

    d = np.load(a.path)
    SA, SB, OA, OB = d['SA'], d['SB'], d['OA'], d['OB']
    print(f"Trials: {len(SA):,}   ρυθμός click A={OA.mean()*100:.4f}% "
          f"B={OB.mean()*100:.4f}%\n")

    # same_party=True -> lag 0 ΠΡΕΠΕΙ να ανάβει (κανονική κβαντομηχανική).
    # same_party=False -> lag 0 ΔΕΝ πρέπει να ανάβει: αυτό είναι no-signalling,
    # το αποτέλεσμα του ενός δεν εξαρτάται από τη ρύθμιση του ΑΛΛΟΥ.
    pairs = [("OA vs SA  [ΘΕΤΙΚΟΣ ΕΛΕΓΧΟΣ]", OA, SA, True),
             ("OA vs SB  [το πραγματικό ερώτημα]", OA, SB, False),
             ("OB vs SA  [το πραγματικό ερώτημα]", OB, SA, False),
             ("OB vs SB  [ΘΕΤΙΚΟΣ ΕΛΕΓΧΟΣ]", OB, SB, True)]

    results = {}
    for label, out, st, same_party in pairs:
        print("=" * 78)
        print(label)
        print("=" * 78)
        print(f"{'lag':>6} {'MI (bits)':>12} {'null mean':>12} {'null max':>12} "
              f"{'σ πάνω':>9} {'σήμα?':>7}")
        print("-" * 66)
        rows = []
        for k in LAGS:
            o, s = align(out, st, k)
            m = mi(o, s)
            nd = null_distribution(o, s, a.shuffles, seed=abs(k) + 1)
            sd = nd.std(ddof=1)
            sig = (m - nd.mean()) / sd if sd > 0 else float('nan')
            hit = m > nd.max()
            note = ""
            if k == 0:
                note = (" <- πρέπει ΝΑΙ" if same_party
                        else " <- no-signalling: πρέπει '-'")
            elif abs(k) == 1 and hit:
                note = " <- deadtime;"
            print(f"{k:>6} {m:12.4e} {nd.mean():12.4e} {nd.max():12.4e} "
                  f"{sig:>+9.1f} {('ΝΑΙ' if hit else '-'):>7}{note}")
            rows.append({"lag": k, "mi": m, "null_mean": float(nd.mean()),
                         "null_max": float(nd.max()), "null_sd": float(sd),
                         "sigma": float(sig), "hit": bool(hit)})
        results[label] = rows
        print()

    # --- κρίση deadtime ---
    print("=" * 78)
    print("ΕΛΕΓΧΟΣ DEADTIME")
    print("=" * 78)
    for label, rows in results.items():
        if "ΘΕΤΙΚΟΣ" in label:
            continue
        near = [r for r in rows if abs(r["lag"]) == 1 and r["hit"]]
        far = [r for r in rows if abs(r["lag"]) >= 3 and r["hit"]]
        if not near and not far:
            print(f"  {label:34s} καθαρό — κανένα σήμα")
        elif near and not far:
            print(f"  {label:34s} σήμα ΜΟΝΟ στο lag ±1 -> deadtime, ΟΧΙ ανακάλυψη")
        elif far:
            lags = [r["lag"] for r in far]
            print(f"  {label:34s} ΣΗΜΑ ΣΕ |lag|>=3: {lags}  <-- ΑΞΙΖΕΙ ΔΙΕΡΕΥΝΗΣΗ")

    print("\nΠΩΣ ΔΙΑΒΑΖΕΤΑΙ:")
    print("  Το lag=0 στο OA vs SA πρέπει να δείχνει ΝΑΙ. Αν όχι -> bug.")
    print("  Σήμα στο lag ±1 = deadtime ανιχνευτή. Αναμενόμενο, όχι ανακάλυψη.")
    print("  Σήμα σε |lag| >= 3 ανάμεσα σε outcome και ΑΠΟΜΑΚΡΥΣΜΕΝΗ ρύθμιση")
    print("  θα ήταν παραβίαση measurement independence.")

    np.save("lag_results.npy", results, allow_pickle=True)
    print("\nΑποθηκεύτηκε: lag_results.npy")


if __name__ == '__main__':
    main()
