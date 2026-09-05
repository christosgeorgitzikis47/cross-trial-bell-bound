"""
FIGURES OF THE PAPER - English, vector PDF + PNG.

Fig 1  MI(O ; S at lag k) over the 20,001 lags, both pairs, with the threshold.
       (The main result of section 6.1 - it was missing entirely.)
Fig 2  Eberhard J(k)/sigma: full range (symlog) plus an inset for |k| <= 50.
Fig 3  eps_excl(tau) for the four kernels, with the bend marked.

Every figure is written twice: .pdf (vector, for publication) and .png
(for preview inside the markdown).
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures_en")
os.makedirs(OUT, exist_ok=True)

# Colour-vision-safe qualitative palette (Wong / seaborn colorblind). The
# previous set failed for protanopia: the symmetric and exponential-future
# curves came within dE = 3.7, against a legibility threshold of 8. With these
# the worst pair anywhere is dE = 19.7 under protanopia, 27.3 under
# deuteranopia and 10.6 under tritanopia.
BLUE = "#0173b2"; ORANGE = "#de8f05"; GREEN = "#029e73"; PURPLE = "#cc78bc"
RED = "#c0392b"; INK = "#1a1a1a"; INK2 = "#5a5a5a"; GRID = "#dcdcdc"
DATA = "#4c4c4c"          # one colour for both panels of Figure 1

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    # 7 pt floor at final size: these figures are placed at their natural
    # width, so rcParams sizes are final sizes.
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.02)
    print("written:", os.path.join(OUT, name) + ".{pdf,png}")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1
def fig1():
    """Binned envelope rather than 20,001 overplotted spikes.

    Drawing every lag as a filled vertical in ~500 px of width means each
    pixel keeps the maximum of about 40 lags, so the crest of the ink reads
    as the typical value when the typical value is far lower. Instead: a
    light line for the per-bin maximum, a dark line for the per-bin median.
    An inset shows the histogram of sqrt(G) against the chi^2(1) curve the
    threshold assumes, and the same-pair k = 0 value sits in a broken strip
    above the scanned band, so the factor of 362 is visible rather than
    arithmetic the reader has to do across two pages.

    Units are 1e-6 bits per trial throughout, which keeps a shared exponent
    off the axis and out of the way of the broken strip.
    """
    fs = json.load(open(os.path.join(HERE, "full_scan_results.json")))
    cal = json.load(open(os.path.join(HERE, "meros1_alpha.json")))
    U = 1e-6                                   # display unit
    thr = fs["mi_threshold"] / U
    NB = 200                                   # 200 bins x ~100 lags
    n = fs["n"]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    fig.subplots_adjust(hspace=0.55)
    for ax, (f, label, same) in zip(axes, [
            ("full_scan_OA_vs_SB.npz", r"$I(O_A(i)\,;\,S_B(i+k))$", "OA vs SA"),
            ("full_scan_OB_vs_SA.npz", r"$I(O_B(i)\,;\,S_A(i+k))$", "OB vs SB")]):
        d = np.load(os.path.join(HERE, f))
        lag, mi = d["lags"], d["mi"] / U

        edges = np.linspace(lag.min(), lag.max() + 1, NB + 1)
        idx = np.clip(np.digitize(lag, edges) - 1, 0, NB - 1)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        hi = np.array([mi[idx == b].max() for b in range(NB)])
        md = np.array([np.median(mi[idx == b]) for b in range(NB)])

        ax.fill_between(ctr, 0, hi, color=DATA, alpha=0.14, lw=0)
        ax.plot(ctr, hi, lw=0.8, color=DATA, alpha=0.55,
                label="per-bin maximum (100 lags)")
        ax.plot(ctr, md, lw=1.6, color=DATA, label="per-bin median")
        ax.axhline(thr, color=RED, lw=1.3, ls="--", zorder=5)

        j = int(np.argmax(mi))
        ax.plot(lag[j], mi[j], "o", ms=6, mfc="none", mec=INK, mew=1.1,
                zorder=6)
        # keep the callout clear of the legend at the upper left
        xt = 4200 if lag[j] < 0 else -4200
        ax.annotate(f"largest single lag {mi[j]:.3g} at $k$ = {lag[j]:,}",
                    xy=(lag[j], mi[j]), xytext=(xt, thr * 0.90),
                    fontsize=7.5, color=INK, va="center", ha="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))

        # broken strip above: the same-pair k = 0 value of this same wing
        k0 = cal[same]["mi_measured_exact"] / U
        div = make_axes_locatable(ax)
        top = div.append_axes("top", size="20%", pad=0.05, sharex=ax)
        top.bar([0], [k0], width=2600, color=RED, alpha=0.8, lw=0)
        top.set_ylim(0, k0 * 1.7)
        top.set_yticks([k0]); top.set_yticklabels([f"{k0:.0f}"], fontsize=7)
        top.text(2200, k0 * 0.95,
                 f"same pair, $k$ = 0:  {k0/thr:,.0f}$\\times$ threshold",
                 fontsize=7.5, color=RED, va="top")
        top.tick_params(labelbottom=False, labelsize=7, length=2)
        top.grid(True, axis="y", color=GRID, lw=0.5)
        for sp in ("top", "right", "bottom"):
            top.spines[sp].set_visible(False)
        top.set_ylabel("same\ntrial", fontsize=7, color=INK2, labelpad=2)
        top.text(0.005, 1.30, label, transform=top.transAxes, fontsize=9,
                 color=INK, va="bottom")
        top.text(0.999, 1.30, "0 of 20,001 lags above threshold",
                 transform=top.transAxes, fontsize=8, color=INK2,
                 va="bottom", ha="right")
        top.set_xlim(-10000, 10000)

        # companion panel on the right: the chi^2(1) calibration the
        # threshold rests on. Kept outside the data axes so it never covers
        # the threshold line or the binned envelope.
        G = 2 * n * math.log(2) * mi * U
        ins = div.append_axes("right", size="24%", pad=0.26)
        ins.hist(np.sqrt(G), bins=55, range=(0, 4.2), density=True,
                 color=DATA, alpha=0.30, lw=0)
        xs = np.linspace(0.02, 4.2, 400)
        ins.plot(xs, 2 * norm.pdf(xs), color=RED, lw=1.3)
        ks_p = fs["pairs"]["OA vs SB" if "OA" in f else "OB vs SA"] \
            ["chi2_validation"]["ks_p"]
        ins.set_xlabel(r"$\sqrt{G}$", fontsize=7.5, labelpad=1)
        ins.set_title(r"$\chi^2(1)$ check" + "\n" + f"KS $p$ = {ks_p:.2f}",
                      fontsize=7, pad=3)
        ins.tick_params(labelsize=7, length=2)
        ins.set_xlim(0, 4.2); ins.set_yticks([])
        for sp in ("top", "right", "left"):
            ins.spines[sp].set_visible(False)

        ax.set_ylim(0, thr * 1.14)
        ax.set_ylabel("mutual information\n" r"($10^{-6}$ bits per trial)")
        ax.grid(True, which="major", color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlim(-10000, 10000)
        ax.set_xticks([-10000, -5000, 0, 5000, 10000])
        ax.legend(loc="upper left", frameon=False, fontsize=7.5,
                  handlelength=1.8, borderpad=0.2)

    axes[1].set_xlabel(r"lag $k$ (trials);  $k>0$ = setting in the future "
                       "of the outcome")
    save(fig, "fig1_mi_vs_lag")


# ---------------------------------------------------------------- Figure 2
def fig2():
    """Broken y axis: the baseline near -135 sigma is a fact about the
    statistic, not a result, and letting it own 88% of the height wastes the
    panel. A narrow strip carries it; the main axes carry the region where
    the question lives, -5 to +15. Left panel: |k| <= 50, every lag measured,
    points joined. Right panel: |k| > 50, sampled lags, points not joined."""
    d = json.load(open(os.path.join(HERE, "j_curve_28297_data.json")))
    k = np.array([r["k"] for r in d["rows"]], float)
    z = np.array([r["z"] for r in d["rows"]], float)
    nz = z[k != 0]
    z0 = z[k == 0][0]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2), sharex="col",
                             gridspec_kw={"height_ratios": [1, 2.6],
                                          "width_ratios": [1, 1.15],
                                          "hspace": 0.06, "wspace": 0.22})
    (bA, bB), (mA, mB) = axes                   # baseline strip, main axes

    # k = 0 is drawn on its own: joining it to its neighbours would put a
    # 147-sigma vertical straight through the axis break.
    near = (np.abs(k) <= 50) & (k != 0)
    far = np.abs(k) > 50
    for base, main, m, joined in ((bA, mA, near, True), (bB, mB, far, False)):
        style = dict(color=BLUE, marker="o", ms=3.0, mfc=BLUE, mec="none")
        for ax in (base, main):
            if joined:
                ax.plot(k[m], z[m], "-", lw=0.9, **style)
            else:
                ax.plot(k[m], z[m], linestyle="none", **style)
        if joined:
            main.plot([0], [z0], "o", ms=7, color=RED, zorder=6)
        base.set_ylim(-141, -130)
        main.set_ylim(-5, 15)
        main.axhline(0, color=INK2, lw=0.9, ls=(0, (3, 3)))
        base.grid(True, color=GRID, lw=0.6)
        main.grid(True, color=GRID, lw=0.6)
        base.spines["bottom"].set_visible(False)
        main.spines["top"].set_visible(False)
        for sp in ("top", "right"):
            base.spines[sp].set_visible(False)
        main.spines["right"].set_visible(False)
        base.tick_params(labelbottom=False, length=2)
        base.set_yticks([-135])
        main.set_yticks([-5, 0, 5, 10, 15])
        # the break marks
        kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=5,
                  linestyle="none", color=INK2, mec=INK2, mew=1, clip_on=False)
        base.plot([0, 1], [0, 0], transform=base.transAxes, **kw)
        main.plot([0, 1], [1, 1], transform=main.transAxes, **kw)

    mA.set_ylabel(r"Eberhard $J\,/\,\sigma$")
    bA.set_ylabel("baseline", fontsize=7.5, color=INK2)
    mA.set_xlim(-52, 52)
    mA.text(0.03, 0.95, r"$|k| \leq 50$, every lag measured",
            transform=mA.transAxes, fontsize=8, color=INK, va="top")
    mA.annotate(f"$k$ = 0:  {z0:+.1f}$\\sigma$\n(Bell violation)",
                xy=(0, z0), xytext=(14, 5.5), fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    mA.text(-50, 0.55, r"$J = 0$ — local realism", fontsize=7.5, color=INK2,
            va="bottom")

    mB.set_xscale("symlog", linthresh=100)
    mB.set_xlim(-12000, 12000)
    mB.text(0.03, 0.95, r"$|k| > 50$, sampled lags",
            transform=mB.transAxes, fontsize=8, color=INK, va="top")
    mB.axhline(0, color=INK2, lw=0.9, ls=(0, (3, 3)))
    mB.text(0.5, 0.42, f"all {len(nz)} nonzero lags sit at "
            f"{nz.mean():.0f}$\\sigma$,\nin the strip above;  none has "
            f"$J > 0$",
            transform=mB.transAxes, fontsize=8, color=INK2,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f6f6f4", ec=GRID,
                      lw=0.5))
    for ax in (mA, mB):
        ax.set_xlabel(r"lag $k$ (trials)")
    save(fig, "fig2_j_curve")


# ---------------------------------------------------------------- Figure 3
def fig3():
    """Two panels: the single pulse of 6.3 and the ten-pulse joint bound of
    6.4. The joint bound is the strongest statement the data support and used
    to appear only in a table, while the figure showed the weaker one."""
    m5 = json.load(open(os.path.join(HERE, "meros5_asym.json")))
    m9 = json.load(open(os.path.join(HERE, "meros9_joint.json")))
    taus = np.array(m5["taus"])
    A = m5["pairs"]["OA vs SB"]; B = m5["pairs"]["OB vs SA"]
    JA = m9["joint"]["OA vs SB"]; JB = m9["joint"]["OB vs SA"]
    col = {"sym": BLUE, "future": ORANGE, "past": GREEN, "exp_future": PURPLE}
    lab = {"sym": r"symmetric  $e^{-k^2/2\tau^2},\ k \neq 0$",
           "future": r"future-only  $k>0$",
           "past": r"past-only  $k<0$",
           "exp_future": r"exponential future  $e^{-k/\tau},\ k>0$"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 4.5), sharey=True)
    fig.subplots_adjust(top=0.80, wspace=0.06)
    ymin = min(min(np.array(J[k]["eps_excl"]).min() for k in m5["kernels"])
               for J in (JA, JB)) / 2.2

    for ax, (SA_, SB_, ttl) in zip(
            (ax1, ax2),
            ((A, B, "one pulse (round 28297)"),
             (JA, JB, "all ten pulses, inverse-variance"))):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1, 10000); ax.set_ylim(ymin, 3)
        for k in m5["kernels"]:
            ax.plot(taus, SA_[k]["eps_excl"], lw=1.9, color=col[k],
                    label=lab[k] if ax is ax1 else None)
            ax.plot(taus, SB_[k]["eps_excl"], lw=0.9, color=col[k],
                    alpha=0.55, ls=(0, (4, 2)))
        ax.axhline(1, color=INK2, lw=1, ls=(0, (1, 2)))
        ax.grid(True, which="major", color=GRID, lw=0.7)
        ax.grid(True, which="minor", color=GRID, lw=0.35, alpha=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel(r"$\tau$ — kernel width (trials)")
        ax.text(0.03, 0.965, ttl, transform=ax.transAxes, fontsize=8.5,
                color=INK, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.28", fc="white",
                          ec=GRID, lw=0.5))

    ax1.set_ylabel(r"$\varepsilon$ — coupling strength")
    # the excluded region: an arrow rather than a wash over the whole panel
    ax1.annotate("EXCLUDED\nabove the curves", xy=(1600, 0.30),
                 xytext=(1600, 0.024), fontsize=8.5, color=INK2, ha="center",
                 va="bottom",
                 arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0,
                                 shrinkA=3, shrinkB=3))
    ax1.text(1.15, 0.66, r"$\varepsilon = 1$: the ordinary lag-0 correlation",
             fontsize=7.5, color=INK2, va="top")

    # the truncation bend, annotated once, outside the curves
    for ax in (ax1, ax2):
        ax.axvline(3000, color="#8b8983", lw=1, ls=(0, (2, 3)))
    ax2.annotate("above $\\tau \\approx 3{,}000$ the $\\pm10{,}000$\n"
                 "window truncates the kernel tails:\n"
                 "sensitivity stops improving as $\\sqrt{\\tau}$",
                 xy=(3000, 0.30), xytext=(1.3, 0.30),
                 fontsize=7.5, color=INK2, va="center", ha="left",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))

    leg = ax1.legend(loc="lower left", frameon=True, edgecolor=GRID,
                     framealpha=0.96, fontsize=7.5, borderpad=0.4)
    leg.get_frame().set_linewidth(0.6)

    us = 774.34 / 15_190_485 * 1e6
    for ax in (ax1, ax2):
        sec = ax.secondary_xaxis("top", functions=(lambda t: t * us / 1e3,
                                                   lambda ms: ms * 1e3 / us))
        sec.tick_params(labelsize=7, colors=INK2)
    fig.text(0.5, 0.985, r"$\tau$ in ms — upper bound only "
             r"($\leq$ 51.0 µs per trial, round 28297)",
             fontsize=7.5, color=INK2, ha="center", va="top")
    save(fig, "fig3_exclusion_map")


# ---------------------------------------------------------------- Figure 4
def fig4():
    d = json.load(open(os.path.join(HERE, "meros7_power.json")))
    z_thr = d["z_thr"]
    pts = d["points"]
    col = {"future": ORANGE, "past": GREEN, "exp_future": PURPLE}
    lab = {"future": "future-only", "past": "past-only",
           "exp_future": "exponential future"}
    mrk = {30.0: "o", 300.0: "s"}

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fracs = [0.0, 0.5, 1.0, 2.0]

    # the expected linear scaling (mean over the six series)
    exp_slope = np.mean([r["z_mean"] for r in pts if r["frac"] == 1.0])
    # Anchored prediction, not a fit: the slope is fixed by the measured
    # z at eps = eps_excl and the line is forced through the origin. Nothing
    # is adjusted to the other three levels.
    ax.plot([0, 2.1], [0, 2.1 * exp_slope], color=INK2, lw=0.9,
            ls=(0, (5, 3)), zorder=1,
            label=r"prediction anchored at $\varepsilon_{\mathrm{excl}}$"
                  "\n(not a fit)")

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

    # detection rates per level
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
    save(fig, "fig4_injection_power")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("All written to", OUT)
