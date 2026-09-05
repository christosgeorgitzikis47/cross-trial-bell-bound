r"""
paper/paper_en.md  ->  paper/georgitzikis_2026_cross_trial_bell_bound.pdf
                       (arXiv-style preprint)

WHAT IT DOES
    1. Preprocess the markdown: every unicode symbol becomes REAL LaTeX maths
       ($\tau$, $\varepsilon$, $10^{-6}$, $\hat\delta$ ...), the display
       equations become numbered equation environments, and the figures become
       floats using the VECTOR pdf.
    2. pandoc: markdown -> LaTeX (tables in booktabs).
    3. pdflatex x3 (for references and floats).

THE CONTENT OF THE PAPER DOES NOT CHANGE. Only the typesetting.

Check: after the conversion, whatever is left outside ASCII is scanned and
reported. pdflatex (not xelatex) would choke on unicode, so the check is
mechanical as well.

Note: the Greek letters that appear below in the GREEK table, in the regular
expressions and in the EQUATIONS / INLINE_EXACT source strings are FUNCTIONAL.
They are the symbols of the paper and the exact source text to be matched, and
must not be translated.
"""
import os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "paper", "paper_en.md")
BUILD = os.path.join(ROOT, "paper", "_build")
os.makedirs(BUILD, exist_ok=True)

# --------------------------------------------------------------------------
# 1. The equations: an explicit source -> LaTeX mapping, so that the
#    automatic converter guesses nothing where it matters.
# --------------------------------------------------------------------------
EQUATIONS = [
    ("> **p(i) = p₀(S_own(i)) + α · ε · Σ_k W_τ(k) · S_other(i+k)**",
     r"p(i) \;=\; p_0\!\left(S_{\mathrm{own}}(i)\right) \;+\; "
     r"\alpha\,\varepsilon \sum_k W_\tau(k)\, S_{\mathrm{other}}(i+k)",
     "model"),
    ("**δ(k) = α · ε · W_τ(k)**",
     r"\delta(k) \;=\; \alpha\,\varepsilon\,W_\tau(k)", "delta"),
    ("**I(k) ≈ δ(k)² / (2 ln2 · p₀(1−p₀)) = C · ε² · W_τ(k)²,  "
     "C = α²/(2 ln2 · p₀(1−p₀))**",
     r"I(k) \;\approx\; \frac{\delta(k)^2}{2\ln 2\; p_0(1-p_0)} "
     r"\;=\; C\,\varepsilon^2 W_\tau(k)^2, \qquad "
     r"C = \frac{\alpha^2}{2\ln 2\; p_0(1-p_0)}", "mi"),
    ("> **J = N(11|a₁b₁) − N(10|a₁b₂) − N(01|a₂b₁) − N(11|a₂b₂) ≤ 0**",
     r"J \;=\; N(11|a_1b_1) - N(10|a_1b_2) - N(01|a_2b_1) - N(11|a_2b_2) "
     r"\;\le\; 0", "eberhard"),
    ("> **ε̂_joint = Σ_p (ε̂_p / σ_p²) / Σ_p (1 / σ_p²),  "
     "σ_joint = (Σ_p 1/σ_p²)^{−1/2}**",
     r"\hat\varepsilon_{\mathrm{joint}} = \frac{\sum_p "
     r"\hat\varepsilon_p/\sigma_p^2}{\sum_p 1/\sigma_p^2}, \qquad "
     r"\sigma_{\mathrm{joint}} = \Big(\sum_p 1/\sigma_p^2\Big)^{-1/2}",
     "joint"),
    ("> **δ_phantom(k) = δ_A · ρ(k)**",
     r"\delta_{\mathrm{phantom}}(k) \;=\; \delta_A\,\rho(k)", "phantom"),
    ("> **ε_phantom(τ) = Σ_k W_τ(k) ρ(k) / Q(τ)**",
     r"\varepsilon_{\mathrm{phantom}}(\tau) \;=\; "
     r"\frac{1}{Q(\tau)}\sum_k W_\tau(k)\,\rho(k)", "phantom-eps"),
    ("> **I = Σ_{s,o} P(s,o) log₂ [ P(s,o) / (P(s)P(o)) ]**",
     r"I \;=\; \sum_{s,o} P(s,o)\,\log_2\!\left[\frac{P(s,o)}"
     r"{P(s)P(o)}\right]", "mi-def"),
    ("> **P(o=1|s) log₂[P(o=1|s)/p₀] = (p₀±δ) log₂(1 ± δ/p₀) = "
     "(±δ + δ²/2p₀)/ln2 + O(δ³)**",
     r"P(o{=}1|s)\,\log_2\!\left[\frac{P(o{=}1|s)}{p_0}\right] "
     r"\;=\; (p_0 \pm \delta)\log_2\!\left(1 \pm \frac{\delta}{p_0}"
     r"\right) \;=\; \frac{\pm\delta + \delta^2/2p_0}{\ln 2} "
     r"+ O(\delta^3)", "expansion"),
    ("> **I = [ δ²/(2p₀) + δ²/(2(1−p₀)) ] / ln2 = δ² / (2 ln2 · p₀(1−p₀)) "
     "+ O(δ³)**",
     r"I \;=\; \frac{1}{\ln 2}\left[\frac{\delta^2}{2p_0} + "
     r"\frac{\delta^2}{2(1-p_0)}\right] \;=\; "
     r"\frac{\delta^2}{2\ln 2\; p_0(1-p_0)} + O(\delta^3)", "expansion2"),
    ("> **J = N(o=11 | a₁b₁) − N(o=10 | a₁b₂) − N(o=01 | a₂b₁) − "
     "N(o=11 | a₂b₂)**",
     r"J \;=\; N(o{=}11\,|\,a_1b_1) - N(o{=}10\,|\,a_1b_2) "
     r"- N(o{=}01\,|\,a_2b_1) - N(o{=}11\,|\,a_2b_2)", "eberhard2"),
    ("> **J → (n/4)[ P₁₁ − P₁₀ − P₀₁ − P₁₁ ] = −(n/4)(P₁₀ + P₀₁)**",
     r"J \;\to\; \frac{n}{4}\left[P_{11} - P_{10} - P_{01} - P_{11}"
     r"\right] \;=\; -\frac{n}{4}\left(P_{10} + P_{01}\right)",
     "eberhard-lag"),
]

# "boxed" results
BOXED = [
    ("> **I(O ; S at lag k) < 1.13 × 10⁻⁶ bits per trial for every |k| ≤ 10,000,\n"
     "> at family-wise α = 0.05**",
     r"\boxed{\;I\big(O ; S \text{ at lag } k\big) \;<\; 1.13\times10^{-6}"
     r"\ \text{bits per trial} \quad\text{for every } |k| \le 10{,}000,"
     r"\ \ \alpha_{\mathrm{FW}} = 0.05\;}", "bound1"),
    ("> **I(O_A(i) ; O_B(i+k)) < 1.066 × 10⁻⁶ bits per trial for every "
     "0 < |k| ≤ 10,000**",
     r"\boxed{\;I\big(O_A(i) ; O_B(i+k)\big) \;<\; 1.066\times10^{-6}"
     r"\ \text{bits per trial}\quad\text{for every } 0 < |k| \le 10{,}000\;}",
     "bound2"),
]

# multi-symbol inline expressions that need a hand-written rendering
INLINE_EXACT = [
    ("> **Q(τ) = Σ_k W_τ(k)²**, summed over the scan window |k| ≤ 10,000",
     "\\begin{equation}\nQ(\\tau) \\;=\\; \\sum_{|k|\\le 10{,}000} "
     "W_\\tau(k)^2\n\\label{eq:Q}\n\\end{equation}"),
    ("> **T(τ) = Σ_k W_τ(k) · δ̂(k)**,  with **E[T] = α ε Q(τ)** under the model,\n"
     "> **ε̂ = T/(αQ)**, and **z = T/σ_T**",
     "\\begin{equation}\nT(\\tau) \\;=\\; \\sum_k W_\\tau(k)\\,"
     "\\hat\\delta(k), \\qquad \\mathbb{E}[T] = \\alpha\\,"
     "\\varepsilon\\,Q(\\tau), \\qquad \\hat\\varepsilon = "
     "\\frac{T}{\\alpha Q}, \\qquad z = \\frac{T}{\\sigma_T}\n"
     "\\label{eq:T}\n\\end{equation}"),
    ("ε_excl(τ) = |ε̂| + z_thr σ_T/(αQ), for O_A vs S_B:",
     r"$\varepsilon_{\mathrm{excl}}(\tau) = |\hat\varepsilon| + "
     r"z_{\mathrm{thr}}\,\sigma_T/(\alpha Q)$, for $O_A$ vs $S_B$:"),
    ("σ_T = √(Σ W(k)² σ_δ(k)²)",
     r"$\sigma_T = \sqrt{\textstyle\sum_k W(k)^2\sigma_\delta(k)^2}$"),
    ("G = 2n ln2 · I", r"$G = 2n\ln 2\cdot I$"),
    ("ħ/t₀", r"$\hbar/t_0$"),
    ("Q(τ) = Σ_k W(k)²", r"$Q(\tau)=\sum_k W(k)^2$"),
    ("Q = Σ W(k)²", r"$Q=\sum_k W(k)^2$"),
    ("(m₁−m₂)(r₁−r₂)/2n", r"$(m_1-m_2)(r_1-r_2)/2n$"),
    ("p₀ = (c₁+c₂)/n", r"$p_0=(c_1+c_2)/n$"),
    ("(r₁+r₂)/2", r"$(r_1+r_2)/2$"),
    ("√(r₁r₂)", r"$\sqrt{r_1r_2}$"),
    ("ε̂ = T/(αQ)", r"$\hat\varepsilon=T/(\alpha Q)$"),
    ("T = Σ W(k) δ̂(k)", r"$T=\sum_k W(k)\hat\delta(k)$"),
    ("α = δ(0) = (r₂−r₁)/2", r"$\alpha=\delta(0)=(r_2-r_1)/2$"),
    ("Q = Σ W(k)² purely geometric", r"$Q=\sum_k W(k)^2$ purely geometric"),
    ("I(X,Y : Λ)", r"$I(X,Y:\Lambda)$"),
    ("Q(10⁴)/τ√π = 0.843", r"$Q(10^4)/\tau\sqrt{\pi} = 0.843$"),
    ("√(1/0.499) = 1.415", r"$\sqrt{1/0.499}=1.415$"),
    ("ε·√Q", r"$\varepsilon\cdot\sqrt{Q}$"),
    ("ε_excl(Q) = c/√Q", r"$\varepsilon_{\mathrm{excl}}(Q)=c/\sqrt{Q}$"),
    ("ε_excl ∝ 1/√Q", r"$\varepsilon_{\mathrm{excl}} \propto 1/\sqrt{Q}$"),
    ("ε_excl ∝ 1/α", r"$\varepsilon_{\mathrm{excl}} \propto 1/\alpha$"),
    ("width τ/√2 in I and τ in δ", r"width $\tau/\sqrt2$ in $I$ and $\tau$ in $\delta$"),
    ("the predicted τ/√2", r"the predicted $\tau/\sqrt2$"),
    ("improve as √τ", r"improve as $\sqrt\tau$"),
    ("N(o_A o_B | a b)", r"$N(o_A o_B \,|\, a\,b)$"),
    ("S_A(i), S_B(i) ∈ {1, 2}", r"$S_A(i), S_B(i) \in \{1,2\}$"),
    ("O_A(i), O_B(i) ∈ {0, 1}", r"$O_A(i), O_B(i) \in \{0,1\}$"),
    ("S(i) ∈ {−1, +1}", r"$S(i)\in\{-1,+1\}$"),
    ("round(logspace(0, 4, 25))", r"\texttt{round(logspace(0, 4, 25))}"),
    ("|k| ≤ 10,000", r"$|k| \le 10{,}000$"),
    ("0 < |k| ≤ 10,000", r"$0 < |k| \le 10{,}000$"),
    ("|k| ≤ 2", r"$|k| \le 2$"),
    ("|k| = 1", r"$|k| = 1$"),
    ("|k| > 10,000", r"$|k| > 10{,}000$"),
    ("for k > 0, else 0", r"for $k>0$, else 0"),
    ("for k < 0, else 0", r"for $k<0$, else 0"),
    ("), all k ", r"), all $k$ "),
    ("k > 0 ≡ future", r"$k>0 \equiv$ future"),
    ("at trial i+k influences the outcome at trial i.",
     r"at trial $i+k$ influences the outcome at trial $i$."),
    ("settings at trials i ± k", r"settings at trials $i \pm k$"),
    ("k = 0 belongs to neither", r"$k=0$ belongs to neither"),
    ("Writing p(i) for", r"Writing $p(i)$ for"),
    ("p(i) is an observable", r"$p(i)$ is an observable"),
    ("with c_s the click count", r"with $c_s$ the click count"),
    ("r_s = c_s/m_s", r"$r_s = c_s/m_s$"),
    ("with m_s the number", r"with $m_s$ the number"),
    ("four-term p log p sum", r"four-term $p\log p$ sum"),
    ("δ(k) for half", r"$\delta(k)$ for half"),
    ("N(11|·) terms", r"$N(11|\cdot)$ terms"),
    ("k ≠ 0 with J > 0", r"$k \ne 0$ with $J>0$"),
    ("at k = 0 and a mean", r"at $k=0$ and a mean"),
    ("τ takes 26 values", r"$\tau$ takes 26 values"),
    ("z > 4.848", r"$z > 4.848$"),
    ("z > 3.672", r"$z > 3.672$"),
    ("|z| = 2.36", r"$|z| = 2.36$"),
    ("α_A ≠ α_B", r"$\alpha_A \ne \alpha_B$"),
    ("Δk = 1, 2, 3, 5, 10", r"$\Delta k = 1, 2, 3, 5, 10$"),
    ("z_obs/z_pred = 1.035", r"$z_{\mathrm{obs}}/z_{\mathrm{pred}} = 1.035$"),
    ("σ_T(map)/σ_T(analytic) = 1.018",
     r"$\sigma_T(\mathrm{map})/\sigma_T(\mathrm{analytic}) = 1.018$"),
    ("σ_J = √(N₁ + N₂ + N₃ + N₄)", r"$\sigma_J = \sqrt{N_1+N_2+N_3+N_4}$"),
    ("r_{1,2} = p₀ ∓ δ", r"$r_{1,2} = p_0 \mp \delta$"),
    ("δ/p₀ and δ/(1−p₀)", r"$\delta/p_0$ and $\delta/(1-p_0)$"),
    ("P(s) = ½", r"$P(s) = \tfrac12$"),
    ("p₀ → 1−p₀", r"$p_0 \to 1-p_0$"),
    ("N(·|·)", r"$N(\cdot|\cdot)$"),
    ("δ/p₀ = 0.283", r"$\delta/p_0 = 0.283$"),
    ("P₁₀ ≈ P₀₁ ≈ 6.9 × 10⁻³ against P₁₁ ≈ 5 × 10⁻⁵",
     r"$P_{10} \approx P_{01} \approx 6.9\times10^{-3}$ against "
     r"$P_{11} \approx 5\times10^{-5}$"),
    ("n/4 trials per group", r"$n/4$ trials per group"),
    ("P₁₁, P₁₀, P₀₁", r"$P_{11}, P_{10}, P_{01}$"),
    ("trial i+k with k ≠ 0", r"trial $i+k$ with $k \ne 0$"),
    ("J ≤ 0 for any local", r"$J \le 0$ for any local"),
    ("I(O(i) ; S(i+k)) ≤ I(λ(i) ; S(i+k))",
     r"$I(O(i);S(i+k)) \le I(\lambda(i);S(i+k))$"),
    ("I(O ; S) at nonzero lag", r"$I(O;S)$ at nonzero lag"),
    ("λ(i) to conditions", r"$\lambda(i)$ to conditions"),
    ("whether J is small at k ≠ 0", r"whether $J$ is small at $k \ne 0$"),
    ("ε̂_p = T_p/(α_p Q)", r"$\hat\varepsilon_p = T_p/(\alpha_p Q)$"),
    ("σ_p = σ_Tp/(α_p Q)", r"$\sigma_p = \sigma_{T,p}/(\alpha_p Q)$"),
    ("Q_het = Σ_p (ε̂_p − ε̂_joint)²/σ_p²",
     r"$Q_{\mathrm{het}} = \sum_p (\hat\varepsilon_p - "
     r"\hat\varepsilon_{\mathrm{joint}})^2/\sigma_p^2$"),
    ("|ε̂_joint| + z_thr σ_joint",
     r"$|\hat\varepsilon_{\mathrm{joint}}| + z_{\mathrm{thr}}\sigma_{\mathrm{joint}}$"),
    ("|z_joint| = 2.10", r"$|z_{\mathrm{joint}}| = 2.10$"),
    ("√10 = 3.16", r"$\sqrt{10} = 3.16$"),
    ("√10", r"$\sqrt{10}$"),
    ("α ε √Q ≪ p₀", r"$\alpha\varepsilon\sqrt{Q} \ll p_0$"),
    ("E[O_A | S_B(i+k) = s] = p₀ + δ_A ρ(k) s",
     r"$\mathbb{E}[O_A \,|\, S_B(i+k)=s] = p_0 + \delta_A\,\rho(k)\,s$"),
    ("ρ(k) = Corr(S_A(i), S_B(i+k))", r"$\rho(k) = \mathrm{Corr}(S_A(i), S_B(i+k))$"),
    ("Corr(S_A(i), S_B(i+k))", r"$\mathrm{Corr}(S_A(i), S_B(i+k))$"),
    ("Corr(S_A(i), S_A(i+k)), k ≠ 0", r"$\mathrm{Corr}(S_A(i), S_A(i+k))$, $k \ne 0$"),
    ("Corr(S_B(i), S_B(i+k)), k ≠ 0", r"$\mathrm{Corr}(S_B(i), S_B(i+k))$, $k \ne 0$"),
    ("I(S_A(i) ; S_B(i+k))", r"$I(S_A(i) ; S_B(i+k))$"),
    ("δ_A = 1.96 × 10⁻³", r"$\delta_A = 1.96\times10^{-3}$"),
    ("ρ(k)", r"$\rho(k)$"),
    ("1/√n = 2.58 × 10⁻⁴", r"$1/\sqrt{n} = 2.58\times10^{-4}$"),
    ("1/√n", r"$1/\sqrt{n}$"),
    ("ε̂_A = T_A/(α_A Q)", r"$\hat\varepsilon_A = T_A/(\alpha_A Q)$"),
    ("ε̂_B = T_B/(α_B Q)", r"$\hat\varepsilon_B = T_B/(\alpha_B Q)$"),
    ("α ε √Q", r"$\alpha\varepsilon\sqrt{Q}$"),
    ("Σ W_f(k)W_t(k) / √(Σ W_f(k)² · Σ W_t(k)²)",
     r"$\sum_k W_f(k)W_t(k)\big/\sqrt{\sum_k W_f(k)^2 \cdot "
     r"\sum_k W_t(k)^2}$"),
    ("δ̂(k)/σ_δ(k)", r"$\hat\delta(k)/\sigma_\delta(k)$"),
    ("τ_filter / τ_true", r"$\tau_{\mathrm{filter}}/\tau_{\mathrm{true}}$"),
    ("τ_true = 3,000", r"$\tau_{\mathrm{true}} = 3{,}000$"),
    ("τ_true = 300", r"$\tau_{\mathrm{true}} = 300$"),
    ("τ_true = 30", r"$\tau_{\mathrm{true}} = 30$"),
    ("τ_true", r"$\tau_{\mathrm{true}}$"),
    ("τ_filter", r"$\tau_{\mathrm{filter}}$"),
    ("10^{4/24} = 1.468", r"$10^{4/24} = 1.468$"),
    ("4ε_excl", r"$4\varepsilon_{\mathrm{excl}}$"),
    ("2ε_excl", r"$2\varepsilon_{\mathrm{excl}}$"),
    ("ε_excl/2", r"$\varepsilon_{\mathrm{excl}}/2$"),
    ("ε_excl(τ)", r"$\varepsilon_{\mathrm{excl}}(\tau)$"),
    ("ε_excl(Q)", r"$\varepsilon_{\mathrm{excl}}(Q)$"),
    ("W_τ(k)²", r"$W_\tau(k)^2$"),
    ("W(k)²", r"$W(k)^2$"),
    ("δ(k)²", r"$\delta(k)^2$"),
    ("σ_δ(k)²", r"$\sigma_\delta(k)^2$"),
    ("W_τ(k)", r"$W_\tau(k)$"),
    ("δ̂(k) are uncorrelated", r"$\hat\delta(k)$ are uncorrelated"),
    ("χ²(1)", r"$\chi^2(1)$"),
    ("2×2", r"$2\times2$"),
    ("δ̂(k)", r"$\hat\delta(k)$"),
    ("δ̂", r"$\hat\delta$"),
    ("Î", r"$\hat I$"),
    ("|ε̂|", r"$|\hat\varepsilon|$"),
    ("ε̂", r"$\hat\varepsilon$"),
    ("The J(k) curve", r"The $J(k)$ curve"),
    ("δ(k)", r"$\delta(k)$"),
    ("I(k)", r"$I(k)$"),
    ("Q(τ)", r"$Q(\tau)$"),
    ("T(τ)", r"$T(\tau)$"),
]

GREEK = {"τ": r"\tau", "ε": r"\varepsilon", "ρ": r"\rho", "δ": r"\delta", "σ": r"\sigma",
         "α": r"\alpha", "λ": r"\lambda", "χ": r"\chi", "π": r"\pi",
         "Σ": r"\Sigma", "Λ": r"\Lambda", "Δ": r"\Delta", "µ": r"\mu"}
SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
       "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-"}
SUB = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6",
       "₇": "7", "₈": "8", "₉": "9"}


PROTECT = re.compile(r"(\$\$.*?\$\$|\$[^$\n]*\$|\\[a-zA-Z]+(?:\{[^{}]*\})*)")


def code_spans(t):
    """`code` -> \texttt{code}, escaping the special LaTeX characters."""
    def one(m):
        c = m.group(1)
        # inside \texttt maths is meaningless: the arrow goes back to ASCII
        c = c.replace(r"$\rightarrow$", "->").replace(r"$\to$", "->")
        for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("#", r"\#"),
                     ("%", r"\%"), ("&", r"\&"), ("$", r"\$")):
            c = c.replace(a, b)
        return r"\texttt{%s}" % c
    return re.sub(r"`([^`]+)`", one, t)


def apply_exact(t):
    for src, tex in INLINE_EXACT:
        t = t.replace(src, tex)
    return t


def convert_inline(t):
    """Convert ONLY the parts that are not already LaTeX.

    Anything already turned into maths by the explicit tables (INLINE_EXACT,
    EQUATIONS) passes through untouched -- otherwise the automatic rule would
    run again over its own output and break it."""
    return "".join(part if (part.startswith("$") or part.startswith("\\"))
                   else _convert_plain(part)
                   for part in PROTECT.split(t))


def _convert_plain(t):
    # 1. scientific notation:  1.13 x 10^-6
    def sci(m):
        mant, exp = m.group(1), "".join(SUP[c] for c in m.group(2))
        return rf"${mant} \times 10^{{{exp}}}$"
    t = re.sub(r"([0-9][0-9.,]*)\s*×\s*10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", sci, t)
    t = re.sub(r"(?<![$0-9])10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
               lambda m: "$10^{%s}$" % "".join(SUP[c] for c in m.group(1)), t)
    # 2. a symbol with a superscript or subscript:  k^2, tau^2, p_0, c_1
    t = re.sub(r"([A-Za-zτεδσαλχπρ])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
               lambda m: "$%s^{%s}$" % (GREEK.get(m.group(1), m.group(1)),
                                        "".join(SUP[c] for c in m.group(2))), t)
    t = re.sub(r"([A-Za-zτεδσαλχπρ])([₀₁₂₃₄₅₆₇₈₉]+)",
               lambda m: "$%s_{%s}$" % (GREEK.get(m.group(1), m.group(1)),
                                        "".join(SUB[c] for c in m.group(2))), t)
    # 3. letter subscripts: sigma_T, O_A, S_B, alpha_A, W_tau, eps_excl ...
    def _sub(m):
        base = GREEK.get(m.group(1), m.group(1))
        idx = m.group(3)
        idx = GREEK[idx] if idx in GREEK else r"\mathrm{%s}" % idx
        return "$%s_{%s}$" % (base, idx)
    t = re.sub(r"\b([A-Zσεδαz])_(\{)?([A-Za-zτ]+)(\})?", _sub, t)
    # 4. a square root with a simple argument
    t = re.sub(r"√\(([^)]+)\)", lambda m: "$\\sqrt{%s}$" % m.group(1), t)
    t = re.sub(r"√([A-Za-zτ0-9]+)",
               lambda m: "$\\sqrt{%s}$" % GREEK.get(m.group(1), m.group(1)), t)
    # 5. isolated Greek letters
    for g, tex in GREEK.items():
        t = t.replace(g, f"${tex}$")
    # 6a. a sign glued to a number: "$-$1" would break pandoc
    #     (a closing math delimiter followed by a digit does not count as maths)
    t = re.sub(r"−([0-9][0-9,\.]*)", lambda m: "$-%s$" % m.group(1), t)
    # 6. operators
    for a, b in (("≤", r"$\le$"), ("≥", r"$\ge$"), ("≈", r"$\approx$"),
                 ("≠", r"$\ne$"), ("≪", r"$\ll$"), ("≫", r"$\gg$"), ("≡", r"$\equiv$"), ("∝", r"$\propto$"),
                 ("∈", r"$\in$"), ("±", r"$\pm$"), ("×", r"$\times$"),
                 ("·", r"$\cdot$"), ("−", "$-$"), ("→", r"$\rightarrow$"),
                 ("§", r"\S"), ("—", "---"), ("–", "--")):
        t = t.replace(a, b)
    # 7. tidy up: $..$$..$ -> a single math span
    t = re.sub(r"\$\s*\$", " ", t)
    # 8. an operator IMMEDIATELY followed by a digit: pandoc does not close
    #    the maths before a digit, so the number is absorbed into the span
    for op in (r"\pm", "-", r"\le", r"\ge", r"\approx", r"\ne",
               r"\times", r"\sim", r"\propto"):
        t = re.sub(re.escape("$" + op + "$") + r"([0-9][0-9,\.]*)",
                   lambda m, o=op: "$%s %s$" % (o, m.group(1)), t)
    return t


def figure_block(png, caption, label):
    pdf = png.replace(".png", ".pdf").replace("../", "")
    return (
        "\n\\begin{figure}[htbp]\n\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{{pdf}}}\n"
        f"\\caption{{{caption}}}\n\\label{{fig:{label}}}\n"
        "\\end{figure}\n\n")


def preprocess(md):
    out = md
    frozen = []          # whatever is already final LaTeX is never touched again

    def freeze(block):
        frozen.append(block)
        return "\n@@FROZEN%d@@\n" % (len(frozen) - 1)

    # --- fenced code block -> verbatim (before anything else) ---
    def fence(m):
        return freeze("\\begin{verbatim}\n%s\\end{verbatim}"
                      % m.group(1))
    out = re.sub(r"```[a-z]*\n(.*?)```\n", fence, out, flags=re.S)

    # --- boxed results and equations ---
    for src, tex, lab in EQUATIONS + BOXED:
        assert src in out, f"equation not found: {src[:40]}"
        out = out.replace(src, freeze(
            "\\begin{equation}\n%s\n\\label{eq:%s}\n\\end{equation}"
            % (tex, lab)))
    for src, tex in INLINE_EXACT:
        if tex.startswith("\\begin{equation}"):
            out = out.replace(src, freeze(tex))
        else:
            out = out.replace(src, tex)

    # --- figures: the image plus the caption paragraph right after it ---
    fig_re = re.compile(
        r"!\[Figure (\d)\]\(([^)]+)\)\s*\n\s*\n\*\*Figure \d\.\*\* (.+?)\n\n",
        re.S)

    def fig_sub(m):
        cap = " ".join(m.group(3).split())
        cap = convert_inline(cap)          # το apply_exact έχει ήδη τρέξει
        cap = cap.replace("%", r"\%")
        cap = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", cap)
        cap = code_spans(cap)
        return figure_block(m.group(2), cap, m.group(1))
    out = fig_re.sub(fig_sub, out)

    # --- the remaining text ---
    lines = out.split("\n")
    res, in_raw = [], False
    for ln in lines:
        if ln.startswith(("\\begin{", "\\end{", "\\includegraphics",
                          "\\caption", "\\label", "\\centering")) or \
                ln.strip().startswith("$$"):
            res.append(ln); continue
        if ln.startswith("```"):
            in_raw = not in_raw; res.append(ln); continue
        if in_raw:
            res.append(ln); continue
        res.append(convert_inline(ln))
    out = "\n".join(res)
    # inline code -> \texttt
    out = code_spans(out)
    for i, block in enumerate(frozen):
        out = out.replace("@@FROZEN%d@@" % i, block)
    return out


PREAMBLE = r"""
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{etoolbox}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,justification=justified}
\AtBeginEnvironment{longtable}{\small}
\AtBeginEnvironment{tabular}{\small}
\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.15}
\usepackage{microtype}
\usepackage{needspace}
\AtBeginEnvironment{longtable}{\needspace{20\baselineskip}}
"""


def main():
    md = open(SRC, encoding="utf-8").read()

    # title / author / abstract are taken out of the text
    title = re.search(r"^# (.+)$", md, re.M).group(1)
    # the version line is optional (removed from v11 onwards)
    m_ver = re.search(r"^\*(Version .+?)\*", md, re.M | re.S)
    ver = " ".join(m_ver.group(1).split()) if m_ver else ""
    abstract = re.search(r"## Abstract\n(.+?)\n---", md, re.S).group(1).strip()
    body = md[md.index("## 1. Introduction"):]

    abstract = convert_inline(apply_exact(abstract))
    abstract = code_spans(abstract)
    body = preprocess(body)

    # sections: "## 1. Introduction" -> \section (numbered automatically)
    body = re.sub(r"^## (\d)\. (.+)$", r"\\section{\2}", body, flags=re.M)
    body = re.sub(r"^### (\d)\.(\d) (.+)$", r"\\subsection{\3}", body, flags=re.M)
    body = re.sub(r"^## Appendix A .*?$",
                  r"\\clearpage\n\\appendix\n\\section{Which script produces which number}",
                  body, flags=re.M)
    body = re.sub(r"^## Appendix B .*?$", r"\\section{Reproduction}",
                  body, flags=re.M)
    body = re.sub(r"^## Appendix C .*?$", r"\\section{Derivations}",
                  body, flags=re.M)
    body = re.sub(r"^### C\.(\d) (.+)$", r"\\subsection{\2}", body, flags=re.M)
    for h in ("Data and code availability", "Acknowledgements", "References"):
        body = body.replace(f"## {h}", f"\\section*{{{h}}}")
    body = body.replace("\n---\n", "\n")

    # bibliography -> thebibliography
    refs = re.search(r"\\section\*\{References\}\n(.+)$", body, re.S)
    items = re.findall(r"^\[(\d+)\] (.+?)(?=\n\[\d+\]|\Z)", refs.group(1),
                       re.S | re.M)
    def emph(t):
        t = " ".join(t.split())
        t = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", t)
        t = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", t)
        t = re.sub(r'"([^"]+)"', r"``\1''", t)      # typographic quotes
        return t
    bib = "\\begin{thebibliography}{9}\n" + "\n".join(
        "\\bibitem{r%s} %s" % (i, emph(t)) for i, t in items
    ) + "\n\\end{thebibliography}\n"
    body = body[:refs.start()] + bib

    pre = os.path.join(BUILD, "paper_pre.md")
    open(pre, "w", encoding="utf-8").write(body)

    left = sorted({c for c in body if ord(c) > 127})
    print("unicode left in the body:", left if left else "NONE")
    bad = re.findall(r"\$[^$\n]{0,40}\$[0-9]", body)
    if bad:
        print("WARNING, maths followed by a digit:", bad[:5])

    meta = os.path.join(BUILD, "meta.yaml")
    open(meta, "w", encoding="utf-8").write(
        "---\ntitle: |\n  " + title + "\nauthor: |\n"
        "  Christos Georgitzikis\\\n  \\normalsize Independent Researcher\n"
        + ("date: |\n  4 September 2026\\\n  \\normalsize\\textit{"
           + ver + "}\n" if ver else "date: 4 September 2026\n")
        + "abstract: |\n  " + abstract.replace("\n", "\n  ") + "\n"
        "---\n")

    tex = os.path.join(BUILD, "paper_en.tex")
    hdr = os.path.join(BUILD, "preamble.tex")
    open(hdr, "w", encoding="utf-8").write(PREAMBLE)
    cmd = ["pandoc", meta, pre, "-f", "markdown+raw_tex+tex_math_dollars",
           "-t", "latex", "--standalone", "-V", "documentclass=article",
           "-V", "fontsize=11pt", "-V", "geometry:margin=1in",
           "-V", "numbersections=true", "-V", "colorlinks=true", "-V", "linkcolor=black",
           "-V", "urlcolor=[rgb]{0.1,0.25,0.5}", "-H", hdr, "-o", tex]
    subprocess.run(cmd, check=True)
    print("wrote the .tex")

    for i in range(3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-halt-on-error", "-output-directory", BUILD,
                            tex], capture_output=True, text=True,
                           errors="replace", cwd=ROOT)
        if r.returncode != 0:
            print(r.stdout[-4000:])
            sys.exit("pdflatex failed on pass %d" % (i + 1))
    out = os.path.join(ROOT, "paper",
                       "georgitzikis_2026_cross_trial_bell_bound.pdf")
    shutil.copy(os.path.join(BUILD, "paper_en.pdf"), out)
    log = open(os.path.join(BUILD, "paper_en.log"), errors="ignore").read()
    m = re.search(r"Output written on .*?\((\d+) pages", log, re.S)
    pages = m.group(1) if m else "?"
    print(f"DONE: {out}  -  {pages} pages")


if __name__ == "__main__":
    main()
