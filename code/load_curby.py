"""
Φόρτωση ωμών δεδομένων Bell test από το CURBy beacon.

ΜΟΡΦΗ (επιβεβαιωμένη — δες ANAFORA.md):
  base64 -> zlib -> dtype = [('SA','u1'),('SB','u1'),('OA','u1'),('OB','u1')]

  Πηγή: trevisan_python_interface/python/extractor_server.py:7-11
        (aggregate=True είναι η ΠΡΟΕΠΙΛΟΓΗ -> 4 bytes ανά trial)

  ΟΧΙ η παλιά εργαστηριακή μορφή [u1,u1,u8,u8] (18 bytes): δεν διαιρεί
  τα αρχεία CURBy και δεν έχει base64. Τα OA/OB εδώ είναι ΗΔΗ
  συγκεντρωμένα σε click/no-click — ΜΗΝ ξαναεφαρμόσεις pulse mask.

  Κοπή στο stoppingCriteria (15.000.000): extractor_jobs.py:67-73

Χρήση:
    python3 load_curby.py <αρχείο.bin> [--out curby_trials.npz]
"""
import argparse, base64, os, sys, zlib
import numpy as np

DTYPE = np.dtype([('SA', 'u1'), ('SB', 'u1'), ('OA', 'u1'), ('OB', 'u1')])
STOPPING_CRITERIA = 15_000_000


def read_file(path, stopping=STOPPING_CRITERIA):
    with open(path, 'rb') as fh:
        raw = fh.read()
    data = np.frombuffer(zlib.decompress(base64.b64decode(raw, validate=False)),
                         dtype=DTYPE)
    n_raw = len(data)
    if stopping:
        data = data[:stopping]
    return data, n_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="curby_trials.npz")
    ap.add_argument("--stopping", type=int, default=STOPPING_CRITERIA)
    a = ap.parse_args()

    data, n_raw = read_file(a.path, a.stopping)
    print(f"Αρχείο: {a.path}")
    print(f"Εγγραφές στο αρχείο: {n_raw:,}")
    print(f"Μετά την κοπή στο stoppingCriteria: {len(data):,}\n")

    SA = data['SA'].astype(np.int8)
    SB = data['SB'].astype(np.int8)
    OA = (data['OA'] > 0).astype(np.int8)      # click / no-click
    OB = (data['OB'] > 0).astype(np.int8)

    print("Έλεγχοι λογικότητας:")
    print(f"  SA τιμές: {np.unique(SA)}   κατανομή: {np.bincount(SA)[1:]}")
    print(f"  SB τιμές: {np.unique(SB)}   κατανομή: {np.bincount(SB)[1:]}")
    print(f"  ρυθμός click Alice: {OA.mean()*100:.4f}%")
    print(f"  ρυθμός click Bob:   {OB.mean()*100:.4f}%")

    # Eberhard ως επαλήθευση ότι φορτώθηκε σωστά
    def N(sa, sb, oa, ob):
        return int(((SA == sa) & (SB == sb) & (OA == oa) & (OB == ob)).sum())
    J = N(1,1,1,1) - N(1,2,1,0) - N(2,1,0,1) - N(2,2,1,1)
    sd = np.sqrt(N(1,1,1,1) + N(1,2,1,0) + N(2,1,0,1) + N(2,2,1,1))
    print(f"  Eberhard J = {J:,}  ({J/sd:+.1f}σ)  -> "
          f"{'ΟΚ, παραβίαση Bell' if J > 0 else 'ΠΡΟΒΛΗΜΑ: καμία παραβίαση'}")

    np.savez_compressed(a.out, SA=SA, SB=SB, OA=OA, OB=OB)
    print(f"\nΑποθηκεύτηκε: {a.out}  ({len(data):,} trials)")
    print(f"Επόμενο: python3 lag_test.py {a.out}")


if __name__ == '__main__':
    main()
