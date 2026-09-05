"""
PART 6.3 - GENERALISING THE KERNEL (peer review objection #3: "why Gaussian?")

CLAIM: the bound depends on the kernel ONLY through
    Q = sum_k W(k)^2   (with W normalised so that max W = 1)

because:
    eps_excl = |T|/(alpha Q) + z*sigma_T/(alpha Q)  and
    sigma_T = sqrt(sum W(k)^2 sigma_delta(k)^2) ~ mean(sigma_delta)*sqrt(Q)
    -> the noise term becomes z*mean(sigma_delta)/(alpha*sqrt(Q)),
       proportional to 1/sqrt(Q), AND NOTHING ELSE about the kernel survives.

If that is verified numerically over 4 kernels x 26 tau x 2 pairs, then the
Q -> eps_excl table can be read by ANYONE with a kernel shape of their own:
compute its Q, read off the bound.

The check is made on the NOISE term, not on eps_excl as a whole: |eps-hat| is
the random draw of this particular dataset (|z| <= 2.4), not a property of the
kernel. It is reported too, so that its size is visible.
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    taus = np.array(m5["taus"])
    z_thr = m5["z_thr"]
    print("=" * 78)
    print("PART 6.3 - IS eps_excl PROPORTIONAL TO 1/sqrt(Q)?")
    print("=" * 78)

    rows = []
    for pair, alpha in (("OA vs SB", m5["alpha_A"]), ("OB vs SA", m5["alpha_B"])):
        for kn in m5["kernels"]:
            P = m5["pairs"][pair][kn]
            Q = np.array(P["Q"]); sT = np.array(P["sigma_T"])
            T = np.array(P["T"])
            eps_noise = z_thr * sT / (alpha * Q)          # the noise term
            eps_hat = np.abs(T) / (alpha * Q)
            c = eps_noise * np.sqrt(Q)                    # should be ~constant
            for j in range(len(taus)):
                rows.append(dict(pair=pair, kernel=kn, tau=float(taus[j]),
                                 Q=float(Q[j]), eps_noise=float(eps_noise[j]),
                                 eps_hat=float(eps_hat[j]), c=float(c[j]),
                                 c_full=float(P["eps_excl"][j] *
                                              math.sqrt(Q[j])),
                                 eps_excl=float(P["eps_excl"][j])))

    c = np.array([r["c"] for r in rows])
    print(f"  c = eps_noise*sqrt(Q) at {len(rows)} points "
          f"(4 kernels x 26 tau x 2 pairs)")
    print(f"    mean {c.mean():.5f}   sd {c.std(ddof=1):.5f} "
          f"({100*c.std(ddof=1)/c.mean():.2f}%)   "
          f"range [{c.min():.5f}, {c.max():.5f}]")

    print("\n  per kernel and pair (mean c +/- sd):")
    for pair in ("OA vs SB", "OB vs SA"):
        for kn in m5["kernels"]:
            v = np.array([r["c"] for r in rows
                          if r["pair"] == pair and r["kernel"] == kn])
            v10 = np.array([r["c"] for r in rows
                            if r["pair"] == pair and r["kernel"] == kn
                            and r["tau"] >= 10])
            print(f"    {pair}  {kn:<11} {v.mean():.5f} +/- {v.std(ddof=1):.5f}"
                  f"    (tau >= 10: {v10.mean():.5f} +/- {v10.std(ddof=1):.5f})")

    # how much of the spread comes from small tau (undersampling the kernel)
    big = np.array([r["c"] for r in rows if r["tau"] >= 10])
    print(f"\n  tau >= 10 only ({len(big)} points): mean {big.mean():.5f}  "
          f"sd {100*big.std(ddof=1)/big.mean():.2f}%")
    sml = np.array([r["c"] for r in rows if r["tau"] < 10])
    print(f"  tau < 10 only  ({len(sml)} points): mean {sml.mean():.5f}  "
          f"sd {100*sml.std(ddof=1)/sml.mean():.2f}%   "
          f"(there the kernel rests on few points and sigma_delta(k) is not "
          f"constant)")

    # ---- the table for third parties ----
    # for the usable table: the FULL eps_excl*sqrt(Q) (including the |eps-hat|
    # term), taken at its maximum
    cf = np.array([r["c_full"] for r in rows])
    cf10 = np.array([r["c_full"] for r in rows if r["tau"] >= 10])
    c_ref = float(cf10.mean())
    c_hi = float(cf10.max())
    print(f"\n  c_full over ALL points: mean {cf.mean():.5f}  "
          f"max {cf.max():.5f}"
          f"   [the max comes from tau = 1, where the one-sided kernel has"
          f" Q = 0.37]")
    print(f"  c_full at tau >= 10 ({len(cf10)} points): mean {cf10.mean():.5f}"
          f"  max {cf10.max():.5f}  sd {100*cf10.std(ddof=1)/cf10.mean():.2f}%")
    print("\n" + "=" * 78)
    print("TABLE  Q -> eps_excl   (c = max of eps_excl*sqrt(Q) at tau >= 10)")
    print("=" * 78)
    print(f"  eps_excl(Q) = c/sqrt(Q)  with c = {c_hi:.5f}  "
          f"(mean {c_ref:.5f}; the MAXIMUM over 208 points is used,")
    print(f"  so that the table never promises a tighter bound than was "
          f"measured)")
    print(f"  NOTE 1: Q is computed with W normalised to max W = 1, and summed")
    print(f"  over the same window |k| <= 10,000.")
    print(f"  NOTE 2: for Q < 3 (a kernel on 1-2 lags) the 1/sqrt(Q) relation")
    print(f"  was not tested -- there one reads the measured map, not the "
          f"formula.\n")
    print(f"    {'Q':>10} {'eps_excl':>12}   {'example kernel':<38}")
    ex = {1: "delta(k-k0), a single lag",
          2: "two lags of equal weight",
          10: "Gaussian tau ~ 5.6 / exponential tau ~ 20",
          100: "Gaussian tau ~ 56 / square over 100 lags",
          1000: "Gaussian tau ~ 564 / exponential tau ~ 2,000",
          10000: "Gaussian tau ~ 5,642"}
    for Qv in (1, 2, 3, 10, 30, 100, 300, 1000, 3000, 10000):
        print(f"    {Qv:>10} {c_hi/math.sqrt(Qv):>12.4e}   {ex.get(Qv,''):<38}")

    # check: does the table reproduce the real eps_excl?
    print("\n  Check against the real map (OA vs SB, tau >= 10):")
    print(f"    {'kernel':<12} {'tau':>7} {'Q':>9} {'eps_excl(table)':>16} "
          f"{'eps_excl(measured)':>18} {'ratio':>7}")
    for kn in m5["kernels"]:
        for tv in (10, 100, 1000, 10000):
            r = [x for x in rows if x["pair"] == "OA vs SB"
                 and x["kernel"] == kn and abs(x["tau"] - tv) < 1e-9]
            if not r:
                continue
            r = r[0]
            pred = c_hi / math.sqrt(r["Q"])
            print(f"    {kn:<12} {r['tau']:>7.0f} {r['Q']:>9.1f} "
                  f"{pred:>16.4e} {r['eps_excl']:>18.4e} "
                  f"{pred/r['eps_excl']:>7.3f}")

    json.dump(dict(z_thr=z_thr, c_mean_tau_ge_10=c_ref, c_max_tau_ge_10=c_hi,
                   c_sd_rel_all=float(c.std(ddof=1) / c.mean()),
                   c_sd_rel_tau_ge_10=float(big.std(ddof=1) / big.mean()),
                   table={str(q): c_hi / math.sqrt(q)
                          for q in (1, 2, 3, 10, 30, 100, 300, 1000, 3000,
                                    10000)},
                   points=rows),
              open(os.path.join(HERE, "meros6_kernelQ.json"), "w"), indent=2)
    print("\nSaved: meros6_kernelQ.json")


if __name__ == "__main__":
    main()
