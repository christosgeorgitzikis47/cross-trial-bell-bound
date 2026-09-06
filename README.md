# Bounding cross-trial temporal coupling in a loophole-free Bell test

Does the setting chosen at one Bell trial leave any trace in the outcome of a
*different* trial? This repository measures how large such a coupling could be
without having been seen, using ten public rounds of the CURBy randomness
beacon.

Standard quantum mechanics predicts a strong correlation at lag 0 and exactly
nothing at any other lag. Models that relax measurement independence, including
retrocausal ones, generically predict a small signal spread over neighbouring
trials. The result here is null, and its value is that the region is now
mapped rather than assumed empty.

## The result

Scanning every lag |k| ≤ 10,000 between one party's outcome and the other
party's setting, the mutual information is below **1.13 × 10⁻⁶ bits per
trial**, at family-wise α = 0.05 over 40,002 hypotheses. That is 1/362 of the
same-trial outcome–setting information the same analysis measures.

The map that follows is that bound re-expressed in the units of one model
class, not a second result.

In model parameters, for a coupling kernel of width τ trials with the
same-trial term k = 0 excluded, the excluded coupling strength ε falls from
**7.6 × 10⁻²** at τ = 1 to **4.4 × 10⁻⁴** at τ = 10,000 for a single pulse, and
to **1.9 × 10⁻²** and **1.2 × 10⁻⁴** when all ten pulses are combined by
inverse-variance weights, those weights being justified by a measured absence
of covariance between the pulses rather than assumed. Here ε = 1 is the strength of the ordinary lag-0
quantum correlation. Four kernel families were tested (symmetric, future-only,
past-only, one-sided exponential) at 26 widths each: 208 tests on the single
pulse and 208 on the joint estimate, zero detections in either.

Independently, the outcome–outcome mutual information at nonzero lag is below
**1/13,384** of the Bell correlation at k = 0, and by Pinsker's inequality no
outcome lets an adversary guess any other trial's setting with a bias above
**1.25 × 10⁻³**. A Lomb–Scargle frequency scan of all ten pulses covers the
one shape the matched filters cannot see, an oscillatory kernel, and finds
nothing above threshold in 200,000 frequencies. Every bound is verified by
injecting a signal of known strength into the real data and recovering it.

![Exclusion map](figures/fig3_exclusion_map.png)

## The paper

[`paper/georgitzikis_2026_cross_trial_bell_bound.pdf`](paper/georgitzikis_2026_cross_trial_bell_bound.pdf)
— 26 pages, 4 figures. The LaTeX source bundle prepared for arXiv is in
[`paper/arxiv/`](paper/arxiv/); the arXiv identifier will be added here once
the posting is announced.

Christos Georgitzikis, Independent Researcher.
[ORCID 0009-0009-5764-0600](https://orcid.org/0009-0009-5764-0600)

## Data

The CURBy quantum randomness beacon (NIST and University of Colorado Boulder)
publishes the full record of every Bell trial at <https://random.colorado.edu>.

Ten rounds are analysed: 1000, 15000, 22000, 23000, 26000, 28293, 28294, 28295,
28296 and 28297, spanning 2023-10-31 to 2025-08-22, twenty-two months of the
archive. Each is cut at the protocol stopping criterion of 15,000,000 trials.
Round 28297 carries the single-pulse map; all ten carry the joint bound.

The raw `.bin` rounds are about 9 MB each and are not committed here.
[`results/curby_manifest.json`](results/curby_manifest.json) records the URL,
byte count and SHA-256 of every round downloaded, so the input can be verified
bit for bit.

## Reproducing

Tested with Python 3.14.6, NumPy 2.5.1, SciPy 1.18.0 and Matplotlib 3.11.1.
NumPy and SciPy are required; Matplotlib only for the figure scripts.

```bash
pip install -r requirements.txt
cd code

# 1. download the ten raw rounds (rate-limited, hard cap of 5 files per run,
#    so it takes two invocations; about 90 MB in total)
python3 katevasma.py --rounds 28297 28296 28295 28294 28293
python3 katevasma.py --rounds 26000 23000 22000 15000 1000
mkdir -p ../dedomena_curby && mv curby_round_*.bin ../dedomena_curby/

# 2. unpack round 28297 into SA / SB / OA / OB arrays
python3 load_curby.py ../dedomena_curby/curby_round_28297.bin --out curby_28297.npz

# 3. single-pulse analysis (each reads curby_28297.npz from this directory)
python3 full_scan.py                 # +/-10,000 lag scan, settings vs outcomes
python3 j_curve.py curby_28297.npz --png j_curve_28297.png
                                     # Eberhard J(k), the shift-sign self-test;
                                     # the --png name is what figures_en.py reads
python3 lag_test.py curby_28297.npz  # sampled scan with an empirical null
python3 lag_dense.py curby_28297.npz # dense scan near k = 0, deadtime check
python3 meros1_alpha.py              # calibrate alpha, the epsilon = 1 convention
python3 meros2_injection.py          # injection test of the analytic relation
python3 meros3_map.py                # matched filter, symmetric exclusion map
python3 meros3_verify.py             # inject at the bound, half it, twice it
python3 meros4_outcome_outcome.py    # outcome-outcome scan and k = 0 control
python3 meros5_asym.py               # the four kernels
python3 meros5_verify.py             # injection test for the one-sided filters

# 4. audit of quantities quoted in the paper
python3 meros6_p0.py                 # p0 as a marginal; exact vs expanded MI
python3 meros6_kernelQ.py            # the 1/sqrt(Q) regularity and its table
python3 meros6_systematic.py         # origin of the 3-7% systematic
python3 meros6_alpha10.py            # alpha for each of the ten pulses
python3 meros11_optimality.py        # Gaussianity of delta-hat, kernel mismatch

# 5. ten-pulse extensions (these read the raw rounds from dedomena_curby/)
python3 meros7_power.py              # the remaining two injection levels
python3 meros8_settings.py           # setting correlations, the phantom signal
python3 meros9_joint.py              # inverse-variance joint bound
python3 meros10_settings10.py        # phantom check across all ten
python3 meros12_selfmemory.py        # single-party memory scan
python3 meros13_parity.py            # setting-pair parity scan
python3 meros14_oo10.py              # outcome-outcome scan, all ten
python3 meros15_homogeneity.py       # epsilon homogeneity across pulses
python3 meros16_crosspulse.py        # cross-pulse correlation, pulse independence
python3 meros17_periodogram.py       # frequency scan of delta-hat, oscillatory kernels
python3 meros18_covariance.py        # covariance of the per-pulse estimators
python3 stability10.py               # J across all ten pulses, the 0-of-1,060 count

# 6. figures
python3 figures_en.py                # Figures 1-4, vector PDF and PNG
```

Building the PDF and its checks run from the repository root, not from `code/`,
and need pandoc and pdflatex:

```bash
python3 code/build_paper_pdf.py      # paper_en.md -> the PDF
python3 code/check_refs.py           # every internal cross-reference resolves
python3 code/check_tables.py         # no table header orphaned from its table
```

Runtime on a laptop. A single lag scan is FFT-based and takes seconds, but
every script that calibrates against an empirical null is dominated by its
shuffles: `full_scan.py` uses 2,000 and takes about 12 minutes, the
single-pulse matched-filter scripts use 400 and take 7 to 8 minutes each, and
`meros9_joint.py` repeats that across ten pulses and takes about 70 minutes.
It is the longest step by far. A full rerun downstream of the raw data is
roughly two to three hours.

Every script writes a `.json` with the full numbers and a `.txt` transcript of
its console output, into the directory it is run from. The copies committed in
`results/` are exactly what the paper quotes, so a rerun can be compared
against them file by file.

## Layout

```
paper/      the paper: markdown source, built PDF, arXiv submission bundle
code/       every analysis script, flat; they import each other by bare name
results/    the .json, .txt and .npz output of every run quoted in the paper
figures/    Figures 1-4, vector .pdf and .png
```

Every script in `code/` belongs to this analysis; nothing unrelated is kept
here. Appendix A of the paper maps each one to the numbers it produces.

Script names are transliterated Greek: `meros` is part, `katevasma` is
download, `epalitheusi` is verification, `xartografisi` is mapping. The
docstring at the top of each file states what it computes and what it refuses
to assume. Those docstrings and the console output are in Greek; the paper,
this README and the figures are in English.

## Citing

Until the arXiv posting is announced, cite the repository and version:

```
C. Georgitzikis, "Bounding cross-trial temporal coupling in a loophole-free
Bell test", version 12 (2026).
https://github.com/christosgeorgitzikis47/cross-trial-bell-bound
```

## License

The code in `code/` is MIT; see [LICENSE](LICENSE).

The paper in `paper/`, and the figures reproduced in it, are licensed
CC BY 4.0: <https://creativecommons.org/licenses/by/4.0/>. Reuse them
freely, with attribution.
