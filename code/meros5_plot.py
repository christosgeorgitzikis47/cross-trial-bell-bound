"""ΜΕΡΟΣ 5 — το PNG: ε_excl(τ) και για τους τέσσερις πυρήνες μαζί."""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meros5_asym import kernel_W, KERNELS

HERE = os.path.dirname(os.path.abspath(__file__))

SURF = "#fcfcfb"
INK = "#1c1c1a"; INK2 = "#55534e"; INK3 = "#8b8983"; GRID = "#e6e4df"
COL = {"sym": "#3b6fd4", "future": "#c0392b",
       "past": "#1e8a6a", "exp_future": "#8b5cc7"}
LAB = {"sym":        "συμμετρικός  exp(−k²/2τ²)   [αναφορά #6]",
       "future":     "μόνο μέλλον  exp(−k²/2τ²), k > 0",
       "past":       "μόνο παρελθόν  exp(−k²/2τ²), k < 0",
       "exp_future": "μόνο μέλλον, εκθετικός  exp(−k/τ), k > 0"}
US_PER_TRIAL = 774.34 / 15_190_485 * 1e6          # ≈ 50,97 μs (άνω φράγμα)


def main():
    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    taus = np.array(m5["taus"])
    z_thr = m5["z_thr"]
    A = m5["pairs"]["OA vs SB"]; B = m5["pairs"]["OB vs SA"]
    try:
        ver = json.load(open(os.path.join(HERE, "meros5_verify.json")))
    except FileNotFoundError:
        ver = None

    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 10_000)

    allmin = min(np.array(A[k]["eps_excl"]).min() for k in KERNELS)
    ymin = allmin / 3.0
    ax.set_ylim(ymin, 3.0)

    # συντηρητικό περίβλημα: ο ΧΕΙΡΟΤΕΡΟΣ πυρήνας, και τα δύο ζεύγη
    env = np.max([np.maximum(np.array(A[k]["eps_excl"]),
                             np.array(B[k]["eps_excl"])) for k in KERNELS],
                 axis=0)
    ax.fill_between(taus, env, 3.0, color=INK3, alpha=0.09, lw=0, zorder=1)

    for k in KERNELS:
        eA = np.array(A[k]["eps_excl"]); eB = np.array(B[k]["eps_excl"])
        ax.plot(taus, eA, lw=2.1, color=COL[k], zorder=4, label=LAB[k])
        ax.plot(taus, eB, lw=1.0, color=COL[k], alpha=0.55, zorder=3,
                ls=(0, (4, 2)))

    # σημεία επαλήθευσης με ένεση
    if ver:
        pts = [(p["tau"], p["eps"]) for p in ver["points"] if p["frac"] == 1.0]
        pts2 = [(p["tau"], p["eps"]) for p in ver["points"] if p["frac"] == 2.0]
        if pts:
            ax.scatter(*zip(*pts), s=58, marker="o", color=SURF,
                       edgecolor=INK, linewidth=1.8, zorder=6,
                       label="ένεση στο ε_excl  (ισχύς ~50%, εξ ορισμού)")
            ax.scatter(*zip(*pts2), s=58, marker="^", color=INK,
                       edgecolor=SURF, linewidth=1.2, zorder=6,
                       label="ένεση στο 2·ε_excl  →  ανιχνεύθηκε 100%")

    ax.axhline(1.0, color=INK2, lw=1.2, ls=(0, (1, 2)), zorder=3)
    ax.text(1.15, 1.12, "ε = 1  (πλήρης ένταση της κανονικής lag-0 σύνδεσης)",
            color=INK2, fontsize=8.5, ha="left", va="bottom")
    ax.axvline(3000, color=INK3, lw=1, ls=(0, (2, 3)), zorder=2)
    ax.annotate("το παράθυρο ±10.000\nκόβει τις ουρές", xy=(3000, ymin * 1.9),
                xytext=(3400, ymin * 1.9), color=INK2, fontsize=8.5,
                va="center", ha="left")
    ax.text(1.15, 1.9, "ΑΠΟΚΛΕΙΕΤΑΙ  (για κάθε πυρήνα)", color=INK2,
            fontsize=12, fontweight="bold", va="bottom", ha="left", zorder=7)

    ax.set_xlabel("τ  —  χρονική έκταση του πυρήνα (trials)", fontsize=11,
                  color=INK)
    ax.set_ylabel("ε  —  ένταση της σύζευξης", fontsize=11, color=INK)
    ax.set_title("Ασύμμετροι πυρήνες: άνω όριο ε_excl(τ) για "
                 "λ(i) = λ₀(i) + ε·Σₖ W(k)·S(i+k)\n"
                 "CURBy γύρος 28297 · n = 15.000.000 · matched filter στο δ(k) "
                 f"· Bonferroni z > {z_thr:.2f}",
                 fontsize=12, color=INK, pad=14, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK3); ax.spines[sp].set_linewidth(0.9)
    ax.tick_params(colors=INK2, labelsize=9.5)
    leg = ax.legend(loc="lower left", frameon=True, fontsize=9,
                    facecolor=SURF, edgecolor=GRID, framealpha=0.95)
    for t in leg.get_texts():
        t.set_color(INK)

    ax2 = ax.secondary_xaxis("top", functions=(
        lambda t: t * US_PER_TRIAL / 1e3, lambda ms: ms * 1e3 / US_PER_TRIAL))
    ax2.set_xlabel("τ σε ms — ΑΝΩ ΦΡΑΓΜΑ (≤ 51,0 μs/trial στον γύρο 28297)",
                   fontsize=9, color=INK2, labelpad=8)
    ax2.tick_params(colors=INK2, labelsize=9)
    ax2.spines["top"].set_color(INK3)

    # --- ένθετο: το σχήμα των πυρήνων ---
    ins = fig.add_axes([0.638, 0.572, 0.222, 0.192])
    ins.set_facecolor(SURF)
    x = np.linspace(-3.2, 3.2, 641)
    for k in ("future", "past", "exp_future"):
        ins.plot(x, kernel_W(k, x, 1.0), lw=1.6, color=COL[k])
    ins.plot(x, kernel_W("sym", x, 1.0), lw=1.4, color=COL["sym"],
             ls=(0, (3, 2)))
    ins.axvline(0, color=INK3, lw=0.8, ls=(0, (2, 2)))
    ins.set_xlabel("k / τ   (k > 0 = ΜΕΛΛΟΝ)", fontsize=8, color=INK2)
    ins.set_ylabel("W(k)", fontsize=8, color=INK2)
    ins.set_title("σχήμα πυρήνα", fontsize=8.5, color=INK2)
    ins.tick_params(colors=INK3, labelsize=7)
    for sp in ("top", "right"):
        ins.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ins.spines[sp].set_color(GRID)

    fig.text(0.008, 0.012,
             "Ο μονόπλευρος πυρήνας χάνει τη μισή ισχύ του φίλτρου (Q → Q/2), "
             "άρα το όριο χαλαρώνει κατά √2· ο εκθετικός έχει Q ≈ τ/2 αντί "
             "τ√π, άρα κατά ≈1,9×.\nΤο k = 0 δεν ανήκει σε κανέναν μονόπλευρο "
             "πυρήνα. Συνεχής γραμμή = OA vs SB (αποτέλεσμα Alice ; ρύθμιση "
             "Bob) · διακεκομμένη = OB vs SA.",
             fontsize=8, color=INK3, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    out = os.path.join(HERE, "meros5_xartis_asymmetroi.png")
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("Αποθηκεύτηκε:", out)

    # ---- πίνακας ----
    want = [1, 3, 10, 32, 100, 316, 1000, 3162, 10000]
    print(f"\n{'τ':>7}" + "".join(f"{k:>13}" for k in KERNELS) +
          f"{'fut/sym':>9}{'past/sym':>9}{'exp/sym':>9}")
    for w in want:
        j = int(np.argmin(np.abs(taus - w)))
        row = [np.array(A[k]["eps_excl"])[j] for k in KERNELS]
        print(f"{taus[j]:>7.0f}" + "".join(f"{v:>13.4e}" for v in row) +
              f"{row[1]/row[0]:>9.3f}{row[2]/row[0]:>9.3f}"
              f"{row[3]/row[0]:>9.3f}")


if __name__ == "__main__":
    main()
