"""
ΜΕΡΟΣ 6.3 — ΓΕΝΙΚΕΥΣΗ ΠΥΡΗΝΑ (ένσταση peer review #3: «γιατί γκαουσιανός;»)

ΙΣΧΥΡΙΣΜΟΣ: το όριο εξαρτάται από τον πυρήνα ΜΟΝΟ μέσω του
    Q = Σ_k W(k)²   (με W κανονικοποιημένο ώστε max W = 1)

γιατί:
    ε_excl = |T|/(αQ) + z·σ_T/(αQ)   και   σ_T = √(Σ W(k)²σ_δ(k)²) ≈ σ̄_δ·√Q
    -> ο θορυβώδης όρος γίνεται  z·σ̄_δ/(α·√Q)  ∝ 1/√Q
       ΚΑΙ ΤΙΠΟΤΑ ΑΛΛΟ από τον πυρήνα δεν επιβιώνει.

Αν αυτό επαληθεύεται αριθμητικά στους 4 πυρήνες × 26 τ × 2 ζεύγη, τότε ο
πίνακας Q -> ε_excl διαβάζεται από ΟΠΟΙΟΝΔΗΠΟΤΕ με δικό του σχήμα πυρήνα:
υπολογίζει το Q του, διαβάζει το όριο.

Ο έλεγχος γίνεται στον ΘΟΡΥΒΩΔΗ όρο, όχι στο ε_excl συνολικά: το |ε̂| είναι
η τυχαία παρατήρηση αυτού του dataset (|z| ≤ 2,4), δεν είναι ιδιότητα του
πυρήνα. Δίνεται και αυτό, για να φαίνεται πόσο μετράει.
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    taus = np.array(m5["taus"])
    z_thr = m5["z_thr"]
    print("=" * 78)
    print("ΜΕΡΟΣ 6.3 — ε_excl ∝ 1/√Q ;")
    print("=" * 78)

    rows = []
    for pair, alpha in (("OA vs SB", m5["alpha_A"]), ("OB vs SA", m5["alpha_B"])):
        for kn in m5["kernels"]:
            P = m5["pairs"][pair][kn]
            Q = np.array(P["Q"]); sT = np.array(P["sigma_T"])
            T = np.array(P["T"])
            eps_noise = z_thr * sT / (alpha * Q)          # ο θορυβώδης όρος
            eps_hat = np.abs(T) / (alpha * Q)
            c = eps_noise * np.sqrt(Q)                    # πρέπει ≈ σταθερό
            for j in range(len(taus)):
                rows.append(dict(pair=pair, kernel=kn, tau=float(taus[j]),
                                 Q=float(Q[j]), eps_noise=float(eps_noise[j]),
                                 eps_hat=float(eps_hat[j]), c=float(c[j]),
                                 c_full=float(P["eps_excl"][j] *
                                              math.sqrt(Q[j])),
                                 eps_excl=float(P["eps_excl"][j])))

    c = np.array([r["c"] for r in rows])
    print(f"  c = ε_noise·√Q σε {len(rows)} σημεία (4 πυρήνες × 26 τ × 2 ζεύγη)")
    print(f"    μέσος {c.mean():.5f}   sd {c.std(ddof=1):.5f} "
          f"({100*c.std(ddof=1)/c.mean():.2f}%)   "
          f"εύρος [{c.min():.5f}, {c.max():.5f}]")

    print("\n  ανά πυρήνα και ζεύγος (μέσος c ± sd):")
    for pair in ("OA vs SB", "OB vs SA"):
        for kn in m5["kernels"]:
            v = np.array([r["c"] for r in rows
                          if r["pair"] == pair and r["kernel"] == kn])
            v10 = np.array([r["c"] for r in rows
                            if r["pair"] == pair and r["kernel"] == kn
                            and r["tau"] >= 10])
            print(f"    {pair}  {kn:<11} {v.mean():.5f} ± {v.std(ddof=1):.5f}"
                  f"    (τ ≥ 10: {v10.mean():.5f} ± {v10.std(ddof=1):.5f})")

    # πόσο από τη διασπορά οφείλεται στα μικρά τ (υποδειγματοληψία του πυρήνα)
    big = np.array([r["c"] for r in rows if r["tau"] >= 10])
    print(f"\n  μόνο τ ≥ 10 ({len(big)} σημεία): μέσος {big.mean():.5f}  "
          f"sd {100*big.std(ddof=1)/big.mean():.2f}%")
    sml = np.array([r["c"] for r in rows if r["tau"] < 10])
    print(f"  μόνο τ < 10  ({len(sml)} σημεία): μέσος {sml.mean():.5f}  "
          f"sd {100*sml.std(ddof=1)/sml.mean():.2f}%   "
          f"(εκεί ο πυρήνας έχει λίγα σημεία και το σ_δ(k) δεν είναι σταθερό)")

    # ---- ο πίνακας για χρήση από τρίτους ----
    # για τον πίνακα χρήσης: ΠΛΗΡΕΣ ε_excl·√Q (μαζί με τον όρο |ε̂|), max
    cf = np.array([r["c_full"] for r in rows])
    cf10 = np.array([r["c_full"] for r in rows if r["tau"] >= 10])
    c_ref = float(cf10.mean())
    c_hi = float(cf10.max())
    print(f"\n  c_full ΟΛΑ τα σημεία: μέσος {cf.mean():.5f}  max {cf.max():.5f}"
          f"   [το max έρχεται από τ = 1, όπου ο μονόπλευρος πυρήνας έχει"
          f" Q = 0,37]")
    print(f"  c_full σε τ ≥ 10 ({len(cf10)} σημεία): μέσος {cf10.mean():.5f}"
          f"  max {cf10.max():.5f}  sd {100*cf10.std(ddof=1)/cf10.mean():.2f}%")
    print("\n" + "=" * 78)
    print("ΠΙΝΑΚΑΣ  Q -> ε_excl   (c = max του ε_excl·√Q σε τ ≥ 10)")
    print("=" * 78)
    print(f"  ε_excl(Q) = c/√Q  με c = {c_hi:.5f}  "
          f"(μέσος {c_ref:.5f}· χρησιμοποιείται ο ΜΕΓΙΣΤΟΣ σε 208 σημεία,")
    print(f"  ώστε ο πίνακας να μην υπόσχεται ποτέ αυστηρότερο όριο από το "
          f"μετρημένο)")
    print(f"  ΠΡΟΣΟΧΗ 1: το Q υπολογίζεται με W κανονικοποιημένο σε max W = 1,")
    print(f"  και αθροίζεται στο ίδιο παράθυρο |k| ≤ 10.000.")
    print(f"  ΠΡΟΣΟΧΗ 2: για Q < 3 (πυρήνας σε 1-2 lag) η σχέση 1/√Q δεν")
    print(f"  ελέγχθηκε — εκεί διαβάζεται ο μετρημένος χάρτης, όχι ο τύπος.\n")
    print(f"    {'Q':>10} {'ε_excl':>12}   {'παράδειγμα πυρήνα':<38}")
    ex = {1: "δ(k−k₀), ένα μόνο lag",
          2: "δύο lag ίσου βάρους",
          10: "γκαουσιανός τ ≈ 5,6 / εκθετικός τ ≈ 20",
          100: "γκαουσιανός τ ≈ 56 / τετράγωνο 100 lag",
          1000: "γκαουσιανός τ ≈ 564 / εκθετικός τ ≈ 2.000",
          10000: "γκαουσιανός τ ≈ 5.642"}
    for Qv in (1, 2, 3, 10, 30, 100, 300, 1000, 3000, 10000):
        print(f"    {Qv:>10} {c_hi/math.sqrt(Qv):>12.4e}   {ex.get(Qv,''):<38}")

    # έλεγχος: ο πίνακας αναπαράγει τα πραγματικά ε_excl;
    print("\n  Έλεγχος στον πραγματικό χάρτη (OA vs SB, τ ≥ 10):")
    print(f"    {'πυρήνας':<12} {'τ':>7} {'Q':>9} {'ε_excl(πίνακας)':>16} "
          f"{'ε_excl(μετρημένο)':>18} {'λόγος':>7}")
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
    print("\nΑποθηκεύτηκε: meros6_kernelQ.json")


if __name__ == "__main__":
    main()
