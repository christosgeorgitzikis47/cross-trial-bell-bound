"""
ΣΧΗΜΑΤΑ ΤΟΥ PAPER — αγγλικά, vector PDF + PNG.

Fig 1  MI(O ; S at lag k) για τα 20.001 lag, και τα δύο ζεύγη, με το κατώφλι.
       (Το κύριο αποτέλεσμα του §6.1 — έλειπε εντελώς.)
Fig 2  J(k)/σ του Eberhard: πλήρες εύρος (symlog) + ένθετο |k| ≤ 50.
Fig 3  ε_excl(τ) για τους τέσσερις πυρήνες, με την κάμψη σημειωμένη.

Κάθε σχήμα γράφεται δύο φορές: .pdf (vector, για δημοσίευση) και .png
(για preview μέσα στο markdown).
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures_en")
os.makedirs(OUT, exist_ok=True)

BLUE = "#2f5fb3"; ORANGE = "#c0622a"; GREEN = "#1e8a6a"; PURPLE = "#7d51b5"
RED = "#c0392b"; INK = "#1a1a1a"; INK2 = "#5a5a5a"; GRID = "#dcdcdc"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
})


def save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None)
    print("γράφτηκε:", os.path.join(OUT, name) + ".{pdf,png}")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1
def fig1():
    fs = json.load(open(os.path.join(HERE, "full_scan_results.json")))
    thr = fs["mi_threshold"]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    for ax, (f, label, col) in zip(axes, [
            ("full_scan_OA_vs_SB.npz", r"$I(O_A(i)\,;\,S_B(i+k))$", BLUE),
            ("full_scan_OB_vs_SA.npz", r"$I(O_B(i)\,;\,S_A(i+k))$", ORANGE)]):
        d = np.load(os.path.join(HERE, f))
        lag, mi = d["lags"], d["mi"]
        pair = "O_A vs S_B" if "OA" in f else "O_B vs S_A"
        st = fs["pairs"][pair.replace("_", "").replace(" vs ", " vs ")] \
            if False else fs["pairs"]["OA vs SB" if "OA" in f else "OB vs SA"]

        ax.plot(lag, mi, lw=0.3, color=col, alpha=0.9, rasterized=True)
        ax.axhline(thr, color=RED, lw=1.3, ls="--", zorder=5)
        ax.text(9800, thr * 1.03, f"Bonferroni threshold {thr:.3g} bits/trial "
                r"($\alpha_{FW}$ = 0.05, 40,002 hypotheses)",
                color=RED, fontsize=7.8, va="bottom", ha="right")
        j = int(np.argmax(mi))
        ax.plot(lag[j], mi[j], "o", ms=6, mfc="none", mec=INK, mew=1.1,
                zorder=6)
        ax.annotate(f"max {mi[j]:.3g} at $k$ = {lag[j]:,}  "
                    f"({100*mi[j]/thr:.0f}% of threshold)",
                    xy=(lag[j], mi[j]), xytext=(lag[j] + 700, thr * 0.80),
                    fontsize=8, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
        ax.set_ylim(0, thr * 1.16)
        ax.set_ylabel("mutual information\n(bits per trial)")
        ax.set_title(label + f"   —  0 of 20,001 lags above threshold",
                     loc="left", color=INK, pad=4)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlim(-10000, 10000)
    axes[1].set_xlabel(r"lag $k$ (trials);  $k>0$ = setting in the future "
                       "of the outcome")
    fig.suptitle("Figure 1 — no outcome–setting correlation at any lag "
                 "(round 28297, $n$ = 15,000,000)",
                 fontsize=10.5, y=1.005, x=0.02, ha="left")
    save(fig, "fig1_mi_vs_lag")


# ---------------------------------------------------------------- Figure 2
def fig2():
    d = json.load(open(os.path.join(HERE, "j_curve_28297_data.json")))
    k = np.array([r["k"] for r in d["rows"]], float)
    z = np.array([r["z"] for r in d["rows"]], float)
    nz = z[k != 0]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    for ax, full in ((axA, False), (axB, True)):
        m = np.ones_like(k, bool) if full else (np.abs(k) <= 50)
        ax.axhline(0, color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.plot(k[m], z[m], "-", lw=0.9, color=BLUE, marker="o", ms=2.4,
                mfc=BLUE, mec="none")
        ax.plot([0], [z[k == 0][0]], "o", ms=7, color=RED, zorder=6)
        ax.grid(True, color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_ylim(-150, 30)
        ax.set_xlabel(r"lag $k$ (trials)")
    axA.set_ylabel(r"Eberhard $J\,/\,\sigma$")
    axA.set_title(r"$|k| \leq 50$", loc="left", pad=4)
    axA.set_xlim(-52, 52)
    axA.annotate(f"$k$ = 0:  {z[k==0][0]:+.1f}$\\sigma$\n(Bell violation)",
                 xy=(0, z[k == 0][0]), xytext=(6, -30), fontsize=8.5,
                 color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    axB.set_xscale("symlog", linthresh=10)
    axB.set_xlim(-12000, 12000)
    axB.set_title(r"full range, $|k| \leq 10{,}000$ (symlog)", loc="left", pad=4)
    axB.annotate(f"$k \\neq 0$:  {nz.mean():.1f}$\\sigma$ mean\n"
                 f"(range {nz.min():.1f} to {nz.max():.1f}; 0 of 106 with $J>0$)",
                 xy=(300, nz.mean()), xytext=(15, -110), fontsize=8.5,
                 color=INK2)
    fig.suptitle("Figure 2 — the Bell violation exists only at zero lag "
                 "(round 28297)", fontsize=10.5, y=1.02, x=0.02, ha="left")
    save(fig, "fig2_j_curve")


# ---------------------------------------------------------------- Figure 3
def fig3():
    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    taus = np.array(m5["taus"]); z_thr = m5["z_thr"]
    A = m5["pairs"]["OA vs SB"]; B = m5["pairs"]["OB vs SA"]
    col = {"sym": BLUE, "future": RED, "past": GREEN, "exp_future": PURPLE}
    lab = {"sym": r"symmetric  $e^{-k^2/2\tau^2},\ k \neq 0$",
           "future": r"future-only  $k>0$",
           "past": r"past-only  $k<0$",
           "exp_future": r"exponential future  $e^{-k/\tau},\ k>0$"}

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 10000)
    ymin = min(np.array(A[k]["eps_excl"]).min() for k in m5["kernels"]) / 2.5
    ax.set_ylim(ymin, 3)
    env = np.max([np.maximum(A[k]["eps_excl"], B[k]["eps_excl"])
                  for k in m5["kernels"]], axis=0)
    ax.fill_between(taus, env, 3, color="#8b8983", alpha=0.10, lw=0)
    for k in m5["kernels"]:
        ax.plot(taus, A[k]["eps_excl"], lw=1.9, color=col[k], label=lab[k])
        ax.plot(taus, B[k]["eps_excl"], lw=0.9, color=col[k], alpha=0.5,
                ls=(0, (4, 2)))
    ax.axhline(1, color=INK2, lw=1, ls=(0, (1, 2)))
    ax.text(1.15, 1.1, r"$\varepsilon = 1$: as strong as the ordinary "
            "lag-0 quantum correlation", fontsize=8, color=INK2, va="bottom")
    ax.text(1.15, 1.75, "EXCLUDED", fontsize=12, fontweight="bold",
            color=INK2, va="bottom")
    ax.axvline(3000, color="#8b8983", lw=1, ls=(0, (2, 3)))
    ax.annotate("above $\\tau \\approx 3{,}000$ the $\\pm10{,}000$ window\n"
                "truncates the kernel tails ($Q/\\tau\\sqrt{\\pi}$ = 0.843):\n"
                "sensitivity stops improving as $\\sqrt{\\tau}$",
                xy=(3000, ymin * 1.6), xytext=(9600, ymin * 1.9),
                fontsize=8, color=INK2, va="center", ha="right")
    ax.set_xlabel(r"$\tau$ — temporal extent of the kernel (trials)")
    ax.set_ylabel(r"$\varepsilon$ — coupling strength")
    ax.grid(True, which="major", color=GRID, lw=0.7)
    ax.grid(True, which="minor", color=GRID, lw=0.35, alpha=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    leg = ax.legend(loc="lower left", frameon=True, edgecolor=GRID,
                    framealpha=0.95)
    leg.get_frame().set_linewidth(0.6)

    us = 774.34 / 15_190_485 * 1e6
    ax2 = ax.secondary_xaxis("top", functions=(lambda t: t * us / 1e3,
                                               lambda ms: ms * 1e3 / us))
    ax2.set_xlabel(r"$\tau$ in ms — UPPER BOUND only "
                   r"($\leq$ 51.0 µs/trial, round 28297; not a constant "
                   "of the dataset)", fontsize=8.5, color=INK2, labelpad=6)
    ax2.tick_params(labelsize=8, colors=INK2)

    fig.suptitle("Figure 3 — exclusion map: coupling strength $\\varepsilon$ "
                 "versus kernel width $\\tau$, four kernel shapes",
                 fontsize=10.5, y=1.06, x=0.02, ha="left")
    fig.text(0.02, -0.06, "Solid: $O_A$ vs $S_B$. Dashed: $O_B$ vs $S_A$. "
             "Shaded region excluded at family-wise $\\alpha$ = 0.05 "
             "(Bonferroni, 40,002 hypotheses, $z > 4.848$).",
             fontsize=8, color=INK2)
    save(fig, "fig3_exclusion_map")


# ---------------------------------------------------------------- Figure 4
def fig4():
    d = json.load(open(os.path.join(HERE, "meros7_power.json")))
    z_thr = d["z_thr"]
    pts = d["points"]
    col = {"future": RED, "past": GREEN, "exp_future": PURPLE}
    lab = {"future": "future-only", "past": "past-only",
           "exp_future": "exponential future"}
    mrk = {30.0: "o", 300.0: "s"}

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fracs = [0.0, 0.5, 1.0, 2.0]

    # η αναμενόμενη γραμμική κλιμάκωση (μέσος όρος των έξι σημείων)
    exp_slope = np.mean([r["z_mean"] for r in pts if r["frac"] == 1.0])
    ax.plot([0, 2.1], [0, 2.1 * exp_slope], color=INK2, lw=0.9,
            ls=(0, (5, 3)), zorder=1,
            label=r"linear in $\varepsilon$ (expected)")

    ax.axhline(z_thr, color=RED, lw=1.4, ls="--", zorder=2)
    ax.text(2.06, z_thr + 0.35, f"detection threshold  $z$ = {z_thr:.3f}",
            color=RED, fontsize=8.5, ha="right", va="bottom")
    ax.axhline(0, color=INK2, lw=0.7, ls=(0, (2, 3)), zorder=1)

    off = {("future", 30.0): -0.075, ("future", 300.0): -0.045,
           ("past", 30.0): -0.015, ("past", 300.0): 0.015,
           ("exp_future", 30.0): 0.045, ("exp_future", 300.0): 0.075}
    for kn in ("future", "past", "exp_future"):
        for tau in (30.0, 300.0):
            sel = sorted([r for r in pts if r["kernel"] == kn
                          and r["tau"] == tau], key=lambda r: r["frac"])
            x = np.array([r["frac"] for r in sel]) + off[(kn, tau)]
            y = np.array([r["z_mean"] for r in sel])
            e = np.array([r["z_sd"] for r in sel])
            ax.errorbar(x, y, yerr=e, fmt=mrk[tau], ms=4.5, lw=0,
                        elinewidth=0.9, capsize=2.4, color=col[kn],
                        mfc=col[kn] if tau == 30.0 else "white",
                        mec=col[kn], mew=1.1, zorder=4,
                        label=f"{lab[kn]}, " + r"$\tau$ = " + f"{tau:g}")

    # ποσοστά ανίχνευσης ανά επίπεδο
    for f in fracs:
        sel = [r for r in pts if r["frac"] == f]
        det = sum(r["n_pass"] for r in sel)
        tot = sum(r["reps"] for r in sel)
        ax.text(f, -2.4, f"{det}/{tot}", ha="center", va="center",
                fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc="#f0f2f5",
                          ec=GRID, lw=0.6))
    ax.text(1.03, -3.5, "runs detected out of 60 at each level",
            ha="center", va="center", fontsize=8.5, color=INK2)

    ax.set_xlim(-0.18, 2.2)
    ax.set_ylim(-4.2, 14)
    ax.set_xticks(fracs)
    ax.set_xticklabels(["0", "0.5", "1", "2"])
    ax.set_xlabel(r"injected coupling, in units of the stated bound  "
                  r"$\varepsilon / \varepsilon_{\mathrm{excl}}(\tau)$")
    ax.set_ylabel(r"matched-filter significance  $z = T/\sigma_T$")
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    leg = ax.legend(loc="upper left", frameon=True, edgecolor=GRID,
                    framealpha=0.95, fontsize=8, ncol=2)
    leg.get_frame().set_linewidth(0.6)
    fig.suptitle("Figure 4 — a signal at the stated bound is recovered; "
                 "twice the bound is always recovered",
                 fontsize=10.5, y=1.02, x=0.02, ha="left")
    save(fig, "fig4_injection_power")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("Όλα στο", OUT)
