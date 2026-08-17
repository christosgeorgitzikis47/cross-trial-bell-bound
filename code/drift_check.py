"""
Έλεγχος drift στα όρια των jobs.

Χρήση:  python3 drift_check.py raw_ibm_marrakesh_524288.npy

Διαβάζει και το αντίστοιχο *_jobs.json για να ξέρει πού τελειώνει κάθε job.

ΓΙΑΤΙ: η ακολουθία είναι κολλημένα κομμάτια από ξεχωριστά jobs. Αν ο
επεξεργαστής βαθμονομήθηκε ή έκανε drift ανάμεσά τους, θα υπάρχει ασυνέχεια
ΑΚΡΙΒΩΣ στα όρια. Αυτό είναι τεχνούργημα του μηχανήματος, όχι φυσική — και
θα μπορούσε να παραστήσει ψεύτικη «δομή» στο κύριο τεστ.
"""
import sys, json, math
import numpy as np
from scipy import stats


def main(path):
    bits = np.load(path).astype(np.int8)
    meta = json.load(open(path.replace(".npy", "_jobs.json")))
    runs = meta["runs"]

    print(f"Αρχείο: {path}")
    print(f"Επεξεργαστής: {meta['backend']}")
    print(f"Σύνολο: {len(bits):,} bits σε {len(runs)} jobs")
    print(f"Συνολικός ρυθμός 1: {bits.mean():.6f}\n")

    print("--- 1. Ρυθμός ανά job ---")
    print(f"    {'job':>3} {'job_id':24s} {'shots':>8} {'ρυθμός 1':>10} {'απόκλιση σ':>11}")
    rates, ks = [], []
    for r in runs:
        seg = bits[r["start"]:r["end"]]
        n, k = len(seg), int(seg.sum())
        p = k / n
        # τυπικό σφάλμα υπό την υπόθεση p=0.5
        z = (p - 0.5) / (0.5 / math.sqrt(n))
        rates.append(p); ks.append(k)
        jid = (r["job_id"] or "—")[:24]
        print(f"    {r['run']:>3} {jid:24s} {n:>8,} {p:>10.6f} {z:>+11.2f}")

    print("\n--- 2. Είναι τα jobs ομοιογενή; ---")
    # χ² ομοιογένειας: έχουν όλα τα jobs τον ίδιο υποκείμενο ρυθμό;
    ns = [len(bits[r["start"]:r["end"]]) for r in runs]
    table = np.array([ks, [n - k for n, k in zip(ns, ks)]])
    chi2, pval, dof, _ = stats.chi2_contingency(table)
    print(f"    χ² = {chi2:.2f}  (dof={dof})   p = {pval:.4f}")
    print(f"    εύρος ρυθμών: {min(rates):.6f} .. {max(rates):.6f}"
          f"   (διαφορά {max(rates)-min(rates):.6f})")
    if pval < 0.01:
        print("    ΠΡΟΣΟΧΗ: τα jobs ΔΕΝ έχουν ίδιο ρυθμό -> υπάρχει drift μεταξύ jobs.")
    else:
        print("    OK: συμβατά με κοινό ρυθμό. Κανένα drift μεταξύ jobs.")

    print("\n--- 3. Ασυνέχεια ΑΚΡΙΒΩΣ στα όρια; ---")
    # Σύγκριση: συσχέτιση διαδοχικών bits ΜΕΣΑ σε job vs ΠΑΝΩ στο όριο.
    # Αν το μηχάνημα «πηδάει» στα όρια, η δεύτερη θα διαφέρει.
    inside = []
    for r in runs:
        seg = bits[r["start"]:r["end"]].astype(float)
        if len(seg) > 1:
            inside.append(np.corrcoef(seg[:-1], seg[1:])[0, 1])
    bnd_a, bnd_b = [], []
    for r in runs[:-1]:
        i = r["end"]
        bnd_a.append(bits[i - 1]); bnd_b.append(bits[i])
    inside = np.array(inside)
    print(f"    συσχέτιση διαδοχικών bits ΜΕΣΑ σε job: "
          f"μέσος {inside.mean():+.5f}  (εύρος {inside.min():+.5f}..{inside.max():+.5f})")
    if len(bnd_a) > 2:
        bc = np.corrcoef(np.array(bnd_a, float), np.array(bnd_b, float))[0, 1]
        print(f"    συσχέτιση ΠΑΝΩ στα {len(bnd_a)} όρια: {bc:+.5f}")
        print("    (λίγα σημεία — ενδεικτικό μόνο)")

    print("\n--- 4. Τάση στον χρόνο ---")
    # Ανεβαίνει ή κατεβαίνει σταθερά ο ρυθμός από job σε job;
    x = np.arange(len(rates), dtype=float)
    sl, ic, rv, pv, se = stats.linregress(x, rates)
    print(f"    κλίση ανά job: {sl:+.2e}   p = {pv:.4f}")
    if pv < 0.01:
        print("    ΠΡΟΣΟΧΗ: συστηματική τάση στον χρόνο -> drift.")
    else:
        print("    OK: καμία συστηματική τάση.")

    print("\n--- ΣΥΜΠΕΡΑΣΜΑ ---")
    bad = (pval < 0.01) or (pv < 0.01)
    if bad:
        print("    Υπάρχει ένδειξη drift. Τα jobs ΔΕΝ είναι ισοδύναμα δείγματα.")
        print("    Το κύριο τεστ σε ενωμένα δεδομένα μπορεί να δει ψεύτικη δομή.")
        print("    Πρόταση: τρέξε το analyse_raw.py και ΑΝΑ JOB ξεχωριστά.")
    else:
        print("    Καμία ένδειξη drift. Τα jobs μπορούν να ενωθούν με ασφάλεια.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Δώσε το αρχείο .npy (θέλει και το *_jobs.json δίπλα του)")
    main(sys.argv[1])
