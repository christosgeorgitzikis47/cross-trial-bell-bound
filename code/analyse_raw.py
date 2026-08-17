"""
Τρέχει το τεστ του DJ σε ωμά κβαντικά δεδομένα.

Χρήση:  python3 analyse_raw.py raw_ibm_brisbane_20000.npy

Χρειάζεται στον ίδιο φάκελο: scratch_test.py, data_gen.py
"""
import sys, numpy as np, math
import scratch_test as T
import data_gen as D


def main(path, stride_samples=800):
    bits = np.load(path).astype(np.int8)
    n = len(bits)
    # κόβουμε σε δύναμη του 2 (χρειάζεται για bit-reversal και stride)
    n2 = 1 << int(math.log2(n))
    bits = bits[:n2]
    print(f"Αρχείο: {path}")
    print(f"Bits: {n2:,}   ρυθμός 1: {bits.mean():.4f}\n")

    print("--- 1. Βασική μέτρηση δομής (χωρίς αναδιάταξη) ---")
    raw = T.score(bits)
    print(f"    {raw:.4f} bits/σύμβολο   (1.0000 = ασυμπίεστο)\n")

    print("--- 2. Αναζήτηση σε περιορισμένες μεταθέσεις ---")
    base, rows = T.search(bits, stride_samples=stride_samples)
    gain = base - rows[0][3]
    for name, s, cost, tot in rows[:5]:
        print(f"    {name:22s} score {s/n2:.4f}  κόστος π {cost:7.1f}  σύνολο {tot:10.1f}")
    print(f"\n    ΚΕΡΔΟΣ: {gain:.1f} bits")

    print("\n--- 3. Έλεγχος αναφοράς: το ίδιο τεστ σε κρυπτογραφικά τυχαία ---")
    ctrl = []
    for s in range(5):
        x = D.crypto(n=n2, seed=7000 + s)
        b2, r2 = T.search(x, stride_samples=stride_samples)
        ctrl.append(b2 - r2[0][3])
    ctrl = np.array(ctrl)
    print(f"    κέρδος ελέγχου: μέσος {ctrl.mean():.1f}, max {ctrl.max():.1f}")

    print("\n--- ΕΤΥΜΗΓΟΡΙΑ ---")
    if gain > max(50, ctrl.max() * 2):
        print("    ΣΗΜΑ. Πριν πανηγυρίσεις: ο πρώτος ύποπτος είναι drift/θόρυβος")
        print("    του επεξεργαστή, ΟΧΙ νέα φυσική. Έλεγχος: τρέξε ξανά σε άλλη ώρα")
        print("    και σε άλλο backend. Αν το σήμα αλλάζει, είναι το μηχάνημα.")
    else:
        print("    ΜΗΔΕΝ. Καμία χρονικά ανακατεμένη δομή ανιχνεύσιμη.")
        print(f"    Όριο: αποκλείεται δομή με ρυθμό εντροπίας κάτω από ~{0.88 + 0.02*math.log2(n2/16384):.2f}")
        print("    (από την καμπύλη ισχύος· χρειάζεται επανυπολογισμός για ακρίβεια)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Δώσε το αρχείο .npy   [--full = σάρωσε ΟΛΑ τα stride, ~10x πιο αργό]")
    # --full: σαρώνει όλες τις stride αναδιατάξεις αντί για δείγμα 800 (10%).
    # Το κόστος περιγραφής μεγαλώνει μόνο ~3 bits, αλλά δεν χάνεται σωστή λύση.
    ss = None if "--full" in sys.argv else 800
    main(sys.argv[1], stride_samples=ss)
