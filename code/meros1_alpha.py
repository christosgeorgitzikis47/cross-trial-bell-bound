"""
PART 1 - CALIBRATING alpha FROM THE DATA

Model (A+C):
    lam(i) = lam0(i) + eps * sum_k W_tau(k) * S(i+k),  W_tau(k)=exp(-k^2/2tau^2)
with S encoded as +/-1.

AMPLITUDE CONVENTION (critical, otherwise a factor of 4 appears):
    S = +/-1  ->  lam = lam0 +/- delta  with  delta = alpha*eps*W_tau(k)
    So the MEASURABLE difference in click rate between the two settings is
        D = rate(S=+1) - rate(S=-1) = 2 delta,   that is  delta = D/2.

The second-order approximation under test:
    I ~ delta^2 / (2 ln2 * p0(1-p0))
It follows from the chi^2(1) of the 2x2 table with equal setting margins:
    chi^2 = D^2*n / (4 p0(1-p0)),  MI = chi^2/(2 n ln2)
          = D^2/(8 ln2 p0(1-p0)) = delta^2/(2 ln2 p0(1-p0))   -- consistent.

alpha is NOT free: we define eps = 1 to mean "as strong as the ordinary
quantum-mechanical outcome-setting dependence of the SAME party at lag 0".
Then (W_tau(0)=1 for every tau):
    alpha = delta(0)  [in units of click probability per unit of eps]

It is run for both parties (OA vs SA, OB vs SB) as a consistency check.
"""
import json, math, os
import numpy as np

from lag_test import mi

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2.0)


def exact_mi_2x2(o, s):
    """Exact MI in bits from the 2x2 table (setting in {1,2})."""
    c = np.zeros((2, 2), float)
    for oi in (0, 1):
        for si in (1, 2):
            c[oi, si - 1] = np.count_nonzero((o == oi) & (s == si))
    j = c / c.sum()
    po = j.sum(1, keepdims=True)
    ps = j.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = j * np.log2(j / (po * ps))
    return float(np.nansum(t)), c


def analyse(label, o, s):
    n = len(o)
    n1 = int(np.count_nonzero(s == 1))
    n2 = n - n1
    k1 = int(np.count_nonzero(o[s == 1]))
    k2 = int(np.count_nonzero(o[s == 2]))
    r1, r2 = k1 / n1, k2 / n2
    p0 = o.mean()

    # delta = half the difference in rate
    Delta = r2 - r1
    delta = Delta / 2.0
    # binomial errors
    se1 = math.sqrt(r1 * (1 - r1) / n1)
    se2 = math.sqrt(r2 * (1 - r2) / n2)
    se_D = math.hypot(se1, se2)
    se_delta = se_D / 2.0

    mi_meas, C = exact_mi_2x2(o, s)
    mi_fast = mi(o, s)                       # the same estimator as the report
    denom = 2 * LN2 * p0 * (1 - p0)
    mi_pred = delta ** 2 / denom
    ratio = mi_pred / mi_meas
    # error on mi_pred from the error on delta (linear propagation:
    # 2 delta sigma_delta / denom)
    se_mi_pred = abs(2 * delta * se_delta) / denom

    # constant C of the model: I(k) = C*eps^2*exp(-k^2/tau^2),
    # C = alpha^2/(2 ln2 p0(1-p0))
    alpha = abs(delta)          # the eps = 1 calibration
    se_alpha = se_delta
    Cmod = alpha ** 2 / denom

    print("=" * 78)
    print(label)
    print("=" * 78)
    print(f"  n = {n:,}   p0 = {p0:.8e}   p0(1-p0) = {p0*(1-p0):.8e}")
    print(f"  2x2 table (rows = click 0/1, columns = setting 1/2):")
    print(f"    no-click : {int(C[0,0]):>12,}  {int(C[0,1]):>12,}")
    print(f"    click    : {int(C[1,0]):>12,}  {int(C[1,1]):>12,}")
    print(f"  click rate | S=1 : {r1:.8e}  ({k1:,} / {n1:,})")
    print(f"  click rate | S=2 : {r2:.8e}  ({k2:,} / {n2:,})")
    print(f"  D = r2 - r1      = {Delta:.6e} +/- {se_D:.2e}   "
          f"({Delta/se_D:.0f} sigma)")
    print(f"  delta(0) = D/2   = {delta:.6e} +/- {se_delta:.2e}")
    print()
    print(f"  MI measured (exact 2x2) = {mi_meas:.6e} bits/trial")
    print(f"  MI measured (report's mi()) = {mi_fast:.6e}  "
          f"[difference {abs(mi_fast-mi_meas):.2e}]")
    print(f"  MI predicted delta^2/(2 ln2 p0(1-p0)) = {mi_pred:.6e} "
          f"+/- {se_mi_pred:.2e}")
    print(f"  ratio predicted/measured = {ratio:.4f}   "
          f"-> discrepancy {abs(ratio-1)*100:.2f}%")
    ok = abs(ratio - 1) <= 0.20
    print(f"  DOES THE APPROXIMATION HOLD (criterion <20%)? "
          f"{'YES' if ok else 'NO'}")
    print()
    print(f"  CALIBRATION (eps = 1 at the ordinary quantum dependence):")
    print(f"    alpha = {alpha:.6e} +/- {se_alpha:.2e}  (relative error "
          f"{se_alpha/alpha*100:.2f}%)")
    print(f"    C = alpha^2/(2 ln2 p0(1-p0)) = {Cmod:.6e} bits/trial per eps^2")
    print()
    return dict(label=label, n=n, n1=n1, n2=n2, k1=k1, k2=k2, r1=r1, r2=r2,
                p0=float(p0), Delta=Delta, se_Delta=se_D, delta=delta,
                se_delta=se_delta, mi_measured_exact=mi_meas,
                mi_measured_fast=mi_fast, mi_predicted=mi_pred,
                se_mi_predicted=se_mi_pred, ratio=ratio,
                approximation_ok=bool(ok), alpha=alpha, se_alpha=se_alpha,
                C=Cmod, counts=C.tolist())


def main():
    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    SA, SB, OA, OB = d['SA'], d['SB'], d['OA'], d['OB']

    res = {}
    res["OA vs SA"] = analyse("OA vs SA  [PRIMARY CALIBRATION]", OA, SA)
    res["OB vs SB"] = analyse("OB vs SB  [CONSISTENCY CHECK]", OB, SB)

    a = res["OA vs SA"]
    print("=" * 78)
    print("CONCLUSION OF PART 1")
    print("=" * 78)
    print(f"  alpha = {a['alpha']:.4e} +/- {a['se_alpha']:.1e}   "
          f"(Alice; Bob: {res['OB vs SB']['alpha']:.4e})")
    print(f"  C = {a['C']:.4e} bits/trial per eps^2")
    print(f"  second-order approximation: "
          f"{'YES' if a['approximation_ok'] else 'NO'} "
          f"(discrepancy {abs(a['ratio']-1)*100:.2f}%)")

    json.dump(res, open(os.path.join(HERE, "meros1_alpha.json"), "w"), indent=2)
    print("\nSaved: meros1_alpha.json")


if __name__ == "__main__":
    main()
