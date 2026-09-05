"""
The sharp test: correlation of an outcome with a FUTURE setting.

The hypothesis: if the future influences the present, the outcome at trial i
is correlated with the setting at trial i+k for k > 0.

k=0 in the pair OA vs SA MUST fire -- that is ordinary quantum mechanics. A
positive control: if it does not fire, the loading is wrong.

NOTE - DEADTIME: a spurious signal at lag +/-1 is expected from detector
deadtime (a click affects the next time slot). It is NOT a discovery. It is
interesting only if it persists at |lag| >= 3.

Usage:  python3 lag_test.py curby_28297.npz
"""
import argparse
import numpy as np

LAGS = [-1000, -100, -10, -3, -1, 0, 1, 3, 10, 100, 1000]
N_SHUFFLE = 100
NS = 3          # the settings take values {1,2} -> indices 0..2


def mi(o, s, ns=NS):
    """Mutual information in bits. bincount instead of np.add.at (9x faster,
    identical result)."""
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
    """We shuffle the SETTINGS -> every correlation is destroyed while the
    margins stay EXACTLY the same. The right kind of nothing."""
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
    print(f"Trials: {len(SA):,}   click rate A={OA.mean()*100:.4f}% "
          f"B={OB.mean()*100:.4f}%\n")

    # same_party=True  -> lag 0 MUST fire (ordinary quantum mechanics).
    # same_party=False -> lag 0 must NOT fire: that is no-signalling, one
    # party's outcome does not depend on the OTHER party's setting.
    pairs = [("OA vs SA  [POSITIVE CONTROL]", OA, SA, True),
             ("OA vs SB  [the real question]", OA, SB, False),
             ("OB vs SA  [the real question]", OB, SA, False),
             ("OB vs SB  [POSITIVE CONTROL]", OB, SB, True)]

    results = {}
    for label, out, st, same_party in pairs:
        print("=" * 78)
        print(label)
        print("=" * 78)
        print(f"{'lag':>6} {'MI (bits)':>12} {'null mean':>12} {'null max':>12} "
              f"{'sig above':>9} {'signal?':>8}")
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
                note = (" <- must be YES" if same_party
                        else " <- no-signalling: must be '-'")
            elif abs(k) == 1 and hit:
                note = " <- deadtime;"
            print(f"{k:>6} {m:12.4e} {nd.mean():12.4e} {nd.max():12.4e} "
                  f"{sig:>+9.1f} {('YES' if hit else '-'):>8}{note}")
            rows.append({"lag": k, "mi": m, "null_mean": float(nd.mean()),
                         "null_max": float(nd.max()), "null_sd": float(sd),
                         "sigma": float(sig), "hit": bool(hit)})
        results[label] = rows
        print()

    # --- deadtime verdict ---
    print("=" * 78)
    print("DEADTIME CHECK")
    print("=" * 78)
    for label, rows in results.items():
        if "POSITIVE" in label:
            continue
        near = [r for r in rows if abs(r["lag"]) == 1 and r["hit"]]
        far = [r for r in rows if abs(r["lag"]) >= 3 and r["hit"]]
        if not near and not far:
            print(f"  {label:34s} clean - no signal")
        elif near and not far:
            print(f"  {label:34s} signal ONLY at lag +/-1 -> deadtime, NOT a discovery")
        elif far:
            lags = [r["lag"] for r in far]
            print(f"  {label:34s} SIGNAL AT |lag|>=3: {lags}  <-- WORTH INVESTIGATING")

    print("\nHOW TO READ THIS:")
    print("  lag=0 in OA vs SA must show YES. If not -> a bug.")
    print("  A signal at lag +/-1 is detector deadtime. Expected, not a find.")
    print("  A signal at |lag| >= 3 between an outcome and a DISTANT setting")
    print("  would be a violation of measurement independence.")

    np.save("lag_results.npy", results, allow_pickle=True)
    print("\nSaved: lag_results.npy")


if __name__ == '__main__':
    main()
