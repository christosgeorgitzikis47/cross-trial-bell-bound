"""
Load raw Bell-test records from the CURBy beacon.

FORMAT (confirmed, see ANAFORA.md):
  base64 -> zlib -> dtype = [('SA','u1'),('SB','u1'),('OA','u1'),('OB','u1')]

  Source: trevisan_python_interface/python/extractor_server.py:7-11
          (aggregate=True is the DEFAULT -> 4 bytes per trial)

  NOT the older laboratory format [u1,u1,u8,u8] (18 bytes): it does not
  divide the CURBy files and there is no base64 in it. OA/OB here are
  ALREADY aggregated to click/no-click -- do NOT reapply a pulse mask.

  Cut at stoppingCriteria (15,000,000): extractor_jobs.py:67-73

Usage:
    python3 load_curby.py <file.bin> [--out curby_trials.npz]
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
    print(f"File: {a.path}")
    print(f"Records in the file: {n_raw:,}")
    print(f"After the cut at stoppingCriteria: {len(data):,}\n")

    SA = data['SA'].astype(np.int8)
    SB = data['SB'].astype(np.int8)
    OA = (data['OA'] > 0).astype(np.int8)      # click / no-click
    OB = (data['OB'] > 0).astype(np.int8)

    print("Sanity checks:")
    print(f"  SA values: {np.unique(SA)}   counts: {np.bincount(SA)[1:]}")
    print(f"  SB values: {np.unique(SB)}   counts: {np.bincount(SB)[1:]}")
    print(f"  click rate Alice: {OA.mean()*100:.4f}%")
    print(f"  click rate Bob:   {OB.mean()*100:.4f}%")

    # Eberhard, as a check that the file was loaded correctly
    def N(sa, sb, oa, ob):
        return int(((SA == sa) & (SB == sb) & (OA == oa) & (OB == ob)).sum())
    J = N(1,1,1,1) - N(1,2,1,0) - N(2,1,0,1) - N(2,2,1,1)
    sd = np.sqrt(N(1,1,1,1) + N(1,2,1,0) + N(2,1,0,1) + N(2,2,1,1))
    print(f"  Eberhard J = {J:,}  ({J/sd:+.1f} sigma)  -> "
          f"{'OK, Bell violation' if J > 0 else 'PROBLEM: no violation'}")

    np.savez_compressed(a.out, SA=SA, SB=SB, OA=OA, OB=OB)
    print(f"\nSaved: {a.out}  ({len(data):,} trials)")
    print(f"Next: python3 lag_test.py {a.out}")


if __name__ == '__main__':
    main()
