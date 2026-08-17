"""
ΜΕΡΟΣ 9 — ΚΟΙΝΟ ΟΡΙΟ ΑΠΟ ΤΟΥΣ ΔΕΚΑ ΠΑΛΜΟΥΣ

ΔΕΝ συγκολλάμε δεδομένα. Οι παλμοί απέχουν μήνες και το α τους διαφέρει 20%·
μια ενιαία ακολουθία θα ήταν λάθος αντικείμενο. Αντί γι' αυτό:

    για κάθε παλμό p:   ε̂_p = T_p/(α_p Q),   σ_p = σ_T,p/(α_p Q)
    κοινή εκτίμηση:     ε̂_joint = Σ (ε̂_p/σ_p²) / Σ (1/σ_p²)
                        σ_joint  = 1/√( Σ 1/σ_p² )
    κοινό όριο:         ε_excl_joint = |ε̂_joint| + z_thr·σ_joint

Βάρη αντίστροφης διασποράς, με το ΔΙΚΟ ΤΟΥ α σε κάθε παλμό. Αυτό είναι η
βέλτιστη γραμμική συνδυαστική εκτίμηση όταν οι παλμοί είναι ανεξάρτητοι, και
είναι ανεκτικό στο ότι το α αλλάζει: παλμός με μικρό α έχει μεγάλο σ_p και
βαραίνει λιγότερο.

Αναμενόμενο κέρδος αν όλοι οι παλμοί ήταν ισοδύναμοι: √10 = 3,16. Στην πράξη
λιγότερο, γιατί το α (άρα η ευαισθησία) διαφέρει.

Το σ_T,p βγαίνει όπως και στον χάρτη: max(εμπειρικό από ανακατέματα, διωνυμικό).
"""
import argparse, json, math, os, sys, time
import numpy as np

from load_curby import read_file
from meros2_injection import scan, mi_and_delta
from meros3_map import sigma_delta
from meros5_asym import build_filters, KERNELS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "dedomena_curby")
ROUNDS = [1000, 15000, 22000, 23000, 26000, 28293, 28294, 28295, 28296, 28297]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=400)
    ap.add_argument("--K", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=5150)
    ap.add_argument("--rounds", type=int, nargs="+", default=ROUNDS)
    a = ap.parse_args()

    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    a10 = json.load(open(os.path.join(HERE, "meros6_alpha10.json")))
    alpha = {r["round"]: (r["alpha_A"], r["alpha_B"]) for r in a10["rounds"]}
    taus = np.array(m5["taus"])
    z_thr = m5["z_thr"]
    Wmats = {kn: build_filters(kn, taus, a.K) for kn in KERNELS}

    print("=" * 78)
    print("ΜΕΡΟΣ 9 — ΚΟΙΝΟ ΟΡΙΟ ΑΠΟ ΤΟΥΣ ΔΕΚΑ ΠΑΛΜΟΥΣ")
    print("=" * 78)
    print(f"  {len(a.rounds)} παλμοί × 2 ζεύγη × {a.shuffles} ανακατέματα")
    print(f"  z_thr = {z_thr:.3f}   τ: {len(taus)} τιμές   K = ±{a.K:,}\n")

    rng = np.random.default_rng(a.seed)
    per_pulse = {}          # [pair][kernel] -> λίστα ανά παλμό
    t_start = time.time()

    for ri, rnd in enumerate(a.rounds):
        t0 = time.time()
        data, _ = read_file(os.path.join(DATA, f"curby_round_{rnd}.bin"))
        SA = data['SA'].astype(np.int8); SB = data['SB'].astype(np.int8)
        OA = (data['OA'] > 0).astype(np.int8)
        OB = (data['OB'] > 0).astype(np.int8)
        del data
        n = len(SA)
        aA, aB = alpha[rnd]

        for label, O, S, al in (("OA vs SB", OA, SB, aA),
                                ("OB vs SA", OB, SA, aB)):
            s1 = (S == 1).astype(np.int8)
            _, n11, A1, B1, nk, _ = scan(O, s1, a.K)
            _, delta = mi_and_delta(n11, A1, B1, nk)
            sd = sigma_delta(A1, B1, nk)

            sh = s1.copy()
            Tnull = {kn: np.empty((a.shuffles, len(taus))) for kn in KERNELS}
            for i in range(a.shuffles):
                rng.shuffle(sh)
                _, n11s, A1s, B1s, nks, _ = scan(O, sh, a.K)
                _, ds = mi_and_delta(n11s, A1s, B1s, nks)
                for kn in KERNELS:
                    Tnull[kn][i] = Wmats[kn] @ ds

            for kn in KERNELS:
                Wm = Wmats[kn]
                Q = (Wm ** 2).sum(axis=1)
                T = Wm @ delta
                sT = np.maximum(Tnull[kn].std(axis=0, ddof=1),
                                np.sqrt(Wm ** 2 @ sd ** 2))
                eps_hat = T / (al * Q)
                sig = sT / (al * Q)
                per_pulse.setdefault(label, {}).setdefault(kn, []).append(
                    dict(round=rnd, alpha=al, eps_hat=eps_hat.tolist(),
                         sigma=sig.tolist(),
                         z=(T / sT).tolist()))
            del s1, sh, Tnull
        del SA, SB, OA, OB

        el = time.time() - t0
        left = (len(a.rounds) - ri - 1) * el / 60
        print(f"  γύρος {rnd}: {el/60:.1f} λεπτά   (απομένουν ~{left:.0f})",
              flush=True)

    # ---------------- συνδυασμός ----------------
    print("\n" + "=" * 78)
    print("ΣΥΝΔΥΑΣΜΟΣ ΜΕ ΒΑΡΗ ΑΝΤΙΣΤΡΟΦΗΣ ΔΙΑΣΠΟΡΑΣ")
    print("=" * 78)
    out = {"rounds": a.rounds, "taus": taus.tolist(), "z_thr": z_thr,
           "shuffles": a.shuffles, "per_pulse": per_pulse, "joint": {}}

    for label in per_pulse:
        out["joint"][label] = {}
        print(f"\n--- {label} ---")
        print(f"  {'πυρήνας':<12} {'τ':>7} {'ε_excl(28297)':>14} "
              f"{'ε_excl(joint)':>14} {'κέρδος':>8} {'z_joint':>8}")
        for kn in KERNELS:
            E = np.array([p["eps_hat"] for p in per_pulse[label][kn]])
            S = np.array([p["sigma"] for p in per_pulse[label][kn]])
            w = 1.0 / S ** 2
            e_joint = (E * w).sum(axis=0) / w.sum(axis=0)
            s_joint = 1.0 / np.sqrt(w.sum(axis=0))
            z_joint = e_joint / s_joint
            excl = np.abs(e_joint) + z_thr * s_joint
            i28 = a.rounds.index(28297)
            excl_single = (np.abs(E[i28]) + z_thr * S[i28])
            gain = excl_single / excl
            out["joint"][label][kn] = dict(
                eps_hat=e_joint.tolist(), sigma=s_joint.tolist(),
                z=z_joint.tolist(), eps_excl=excl.tolist(),
                eps_excl_28297=excl_single.tolist(), gain=gain.tolist(),
                n_above=int((np.abs(z_joint) > z_thr).sum()))
            for jt, tv in enumerate(taus):
                if tv in (1., 10., 100., 1000., 10000.):
                    print(f"  {kn:<12} {tv:>7g} {excl_single[jt]:>14.4e} "
                          f"{excl[jt]:>14.4e} {gain[jt]:>8.2f} "
                          f"{z_joint[jt]:>8.2f}")

    nab = sum(out["joint"][l][k]["n_above"] for l in out["joint"]
              for k in KERNELS)
    gains = np.array([g for l in out["joint"] for k in KERNELS
                      for g in out["joint"][l][k]["gain"]])
    zs = np.array([z for l in out["joint"] for k in KERNELS
                   for z in out["joint"][l][k]["z"]])
    print("\n" + "=" * 78)
    print(f"  |z_joint| > {z_thr:.3f} σε {nab} / "
          f"{2*len(KERNELS)*len(taus)} σημεία")
    print(f"  μέγιστο |z_joint| = {np.abs(zs).max():.2f}")
    print(f"  κέρδος ε_excl(28297)/ε_excl(joint): μέσο {gains.mean():.2f}  "
          f"εύρος [{gains.min():.2f}, {gains.max():.2f}]   "
          f"(ιδανικό √10 = 3,16)")
    out["n_above_total"] = nab
    out["gain_mean"] = float(gains.mean())
    out["max_abs_z"] = float(np.abs(zs).max())
    print(f"  συνολικός χρόνος: {(time.time()-t_start)/60:.0f} λεπτά")

    json.dump(out, open(os.path.join(HERE, "meros9_joint.json"), "w"),
              indent=2)
    print("\nΑποθηκεύτηκε: meros9_joint.json")


if __name__ == "__main__":
    main()
