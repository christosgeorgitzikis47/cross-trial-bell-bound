"""ΜΕΡΟΣ 3 — το PNG του χάρτη αποκλεισμού (ε, τ)."""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))

SURF = "#fcfcfb"
BLUE = "#3b6fd4"        # OA vs SB
ORANGE = "#d1622a"      # OB vs SA
INK = "#1c1c1a"
INK2 = "#55534e"
INK3 = "#8b8983"
GRID = "#e6e4df"

# χρόνος ανά trial: ΑΝΩ ΦΡΑΓΜΑ, από τα μεταδεδομένα του 28297
# (request -> precommit = 774,34 s για 15.190.485 καταγεγραμμένα trials)
US_PER_TRIAL = 774.34 / 15_190_485 * 1e6          # ≈ 50,97 μs


def main():
    m3 = json.load(open(os.path.join(HERE, "meros3_map.json")))
    taus = np.array(m3["taus"])
    z_thr = m3["z_thr"]
    A = m3["pairs"]["OA vs SB"]
    B = m3["pairs"]["OB vs SA"]
    eA = np.array(A["eps_excl"]); eB = np.array(B["eps_excl"])
    env = np.maximum(eA, eB)                       # συντηρητικό περίβλημα

    try:
        ver = json.load(open(os.path.join(HERE, "meros3_verify.json")))
    except FileNotFoundError:
        ver = None

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 10_000)
    ymin = env.min() / 4.0
    ax.set_ylim(ymin, 3.0)

    # --- σκιασμένη = ΑΠΟΚΛΕΙΣΜΕΝΗ περιοχή (πάνω από τη γραμμή) ---
    ax.fill_between(taus, env, 3.0, color=BLUE, alpha=0.10, lw=0, zorder=1)
    ax.fill_between(taus, env, 3.0, facecolor="none", edgecolor=BLUE,
                    hatch="///", alpha=0.22, lw=0, zorder=1)

    ax.plot(taus, eA, lw=2, color=BLUE, zorder=4,
            label="OA vs SB  (αποτέλεσμα Alice ; ρύθμιση Bob)")
    ax.plot(taus, eB, lw=2, color=ORANGE, zorder=4, ls=(0, (5, 2)),
            label="OB vs SA  (αποτέλεσμα Bob ; ρύθμιση Alice)")

    # --- σημεία επαλήθευσης με ένεση ---
    if ver:
        pts_hi = [(p["tau"], p["eps"]) for p in ver["points"]
                  if p["frac"] == 1.0]
        pts_lo = [(p["tau"], p["eps"]) for p in ver["points"]
                  if p["frac"] == 0.5]
        if pts_hi:
            ax.scatter(*zip(*pts_hi), s=64, marker="o", color=SURF,
                       edgecolor=INK, linewidth=2, zorder=6,
                       label="ένεση στο ε_excl → ανιχνεύθηκε")
            ax.scatter(*zip(*pts_lo), s=64, marker="v", color=INK3,
                       edgecolor=SURF, linewidth=1.5, zorder=6,
                       label="ένεση στο ε_excl/2 → ΔΕΝ ανιχνεύθηκε")

    # --- η κάμψη των μεγάλων τ ---
    ax.axvline(3000, color=INK3, lw=1, ls=(0, (2, 3)), zorder=2)
    ax.annotate("το παράθυρο ±10.000\nκόβει τις ουρές του πυρήνα",
                xy=(3000, ymin * 2.0), xytext=(3450, ymin * 2.0),
                color=INK2, fontsize=8.5, va="center", ha="left")

    # --- ετικέτες περιοχών ---
    ax.text(1.6, 0.9, "ΑΠΟΚΛΕΙΕΤΑΙ", color=BLUE, fontsize=13,
            fontweight="bold", va="top", ha="left", zorder=7)
    ax.text(120, ymin * 1.35, "δεν αποκλείεται", color=INK2, fontsize=12,
            va="bottom", ha="center", zorder=7)

    # ε = 1 : πλήρης ένταση κανονικής κβαντομηχανικής
    ax.axhline(1.0, color=INK2, lw=1.2, ls=(0, (1, 2)), zorder=3)
    ax.text(9500, 1.12, "ε = 1  (πλήρης ένταση της κανονικής lag-0 σύνδεσης)",
            color=INK2, fontsize=8.5, ha="right", va="bottom")

    ax.set_xlabel("τ  —  χρονική έκταση του πυρήνα (trials)", fontsize=11,
                  color=INK)
    ax.set_ylabel("ε  —  ένταση της σύζευξης", fontsize=11, color=INK)
    ax.set_title("Χάρτης αποκλεισμού του μοντέλου  λ(i) = λ₀(i) + ε·Σₖ W_τ(k)·S(i+k)\n"
                 f"CURBy γύρος 28297 · n = 15.000.000 · matched filter στο δ(k) · "
                 f"Bonferroni z > {z_thr:.2f} (m = 40.002)",
                 fontsize=12, color=INK, pad=14, loc="left")

    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK3); ax.spines[sp].set_linewidth(0.9)
    ax.tick_params(colors=INK2, labelsize=9.5)

    leg = ax.legend(loc="lower left", frameon=True, fontsize=9.5,
                    bbox_to_anchor=(0.0, 0.0),
                    facecolor=SURF, edgecolor=GRID, framealpha=0.95)
    for t in leg.get_texts():
        t.set_color(INK)

    # --- δεύτερος άξονας: ο ΙΔΙΟΣ άξονας σε μονάδες χρόνου ---
    ax2 = ax.secondary_xaxis(
        "top", functions=(lambda t: t * US_PER_TRIAL / 1e3,
                          lambda ms: ms * 1e3 / US_PER_TRIAL))
    ax2.set_xlabel("τ σε ms  —  ΑΝΩ ΦΡΑΓΜΑ (≤ 51,0 μs/trial στον γύρο 28297· "
                   "ο ρυθμός ΔΕΝ είναι σταθερός στο dataset)",
                   fontsize=9, color=INK2, labelpad=8)
    ax2.tick_params(colors=INK2, labelsize=9)
    ax2.spines["top"].set_color(INK3)

    fig.text(0.008, 0.012,
             "Σκιασμένη = αποκλεισμένη. Η γραμμή είναι το άνω όριο ε_excl(τ) = |ε̂| + z·σ_ε. "
             "Η κάμψη μετά το τ ≈ 3.000 είναι πραγματική:\nτο παράθυρο σάρωσης ±10.000 lag κόβει τις ουρές του πυρήνα, "
             "οπότε η ευαισθησία σταματά να βελτιώνεται ως √τ.",
             fontsize=8, color=INK3, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    out = os.path.join(HERE, "meros3_xartis_apokleismou.png")
    fig.savefig(out, dpi=170, facecolor=SURF)
    print("Αποθηκεύτηκε:", out)

    # --------- πίνακας 8 τιμών τ ---------
    want = [1, 3, 10, 32, 100, 316, 1000, 3162, 10000]
    print(f"\n{'τ (trials)':>11} {'τ (ms, άνω φρ.)':>16} "
          f"{'ε_excl OA/SB':>13} {'ε_excl OB/SA':>13} {'z(OA/SB)':>9}")
    for w in want:
        j = int(np.argmin(np.abs(taus - w)))
        print(f"{taus[j]:>11.0f} {taus[j]*US_PER_TRIAL/1e3:>16.3f} "
              f"{eA[j]:>13.3e} {eB[j]:>13.3e} {A['z'][j]:>9.2f}")


if __name__ == "__main__":
    main()
