"""
PART 17 - FREQUENCY SCAN OF delta-hat(k): CLOSING LIMITATION 2

Every kernel in the map is one-signed. An OSCILLATORY coupling -- the real
hazard for a systematic: 50 Hz mains pickup, pump modulation, any periodic
drive -- would be a sinusoid in k and is nearly orthogonal to all four
families. It would be missed by the matched filters of section 6.3.

Here a Lomb-Scargle periodogram of the standardised delta-hat(k)/sigma(k)
over the 20,000 lags k != 0 of round 28297, both channels.

WHY LOMB-SCARGLE: the lag grid is uniform except for the single missing
point k = 0; Lomb-Scargle handles the gap exactly and, in the Scargle
normalisation P/var, the power at each frequency is Exp(1) under Gaussian
white noise -- which the delta-hat were shown to be (section 6.3, KS p =
0.54 / 0.22).

FREQUENCIES: f_j = j/N cycles per lag, j = 1 ... N/2, N = 20,001 lags, so
M = 10,000 independent frequencies per channel. Bonferroni on Exp(1):
P(max > z) <= m exp(-z), so z_thr = ln(m/0.05). Two families are reported,
both pre-specified: the single-pulse one of round 28297 (m = 2M = 20,000,
z = 12.90) and the ten-pulse one that matches every other multi-pulse scan
in this paper (m = 20M = 200,000, z = 15.20). The threshold is ALSO checked
empirically by permuting the lags.

CONVERSION TO HZ is nominal only: f_Hz = f * 250,000 at the instrument's
stated ~4 us per trial (section 4.1); the metadata upper bound of 51 us per
trial would scale every frequency down by 12.7.

POSITIVE CONTROL: an oscillatory coupling is injected at the OUTCOME level,
lam(i) = lam0 + alpha*eps*F(i), F(i) = sum_{k != 0} cos(2 pi f0 k + phi) S_B(i+k),
scanned with the same machinery, and its peak must appear at f0.

SENSITIVITY: a sinusoid of amplitude A in white noise sigma gives LS power
~ N A^2 / (4 sigma^2). With A = alpha*eps the threshold maps to
eps_thr = sqrt(4 z_thr / N) * sigma_delta / alpha, the coupling an
oscillatory kernel of unit peak would need to reach the threshold.
"""
import argparse, json, math, os, time
import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.signal import lombscargle

from meros2_injection import scan, mi_and_delta
from meros3_map import sigma_delta

HERE = os.path.dirname(os.path.abspath(__file__))
K = 10_000
ALPHA_FW = 0.05
US_PER_TRIAL_NOMINAL = 4.0            # section 4.1, Kavuri et al.
US_PER_TRIAL_BOUND = 51.0             # section 4.1, metadata upper bound


def ls_power(k, z, freqs):
    """Lomb-Scargle power in the Scargle normalisation: Exp(1) under white
    Gaussian noise of unit variance. scipy returns (1/2)[...]; dividing by
    the sample variance gives the normalised form."""
    zc = z - z.mean()
    P = lombscargle(k.astype(np.float64), zc, 2 * np.pi * freqs)
    return P / zc.var()


def build_F_cos(S, f0, phi, K):
    """F(i) = sum_{0<|k|<=K} cos(2 pi f0 k + phi) S(i+k), linear FFT correlation."""
    k = np.arange(-K, K + 1, dtype=np.float64)
    W = np.cos(2 * np.pi * f0 * k + phi)
    W[K] = 0.0
    n = len(S)
    L = next_fast_len(n + 2 * K + 1)
    Sf = rfft(S.astype(np.float64), L)
    Wp = np.zeros(L); Wp[:len(W)] = W
    Ff = irfft(Sf * np.conjugate(rfft(Wp, L)), L)
    return np.roll(Ff, K)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1717)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    N = 2 * K + 1
    M = N // 2
    freqs = np.arange(1, M + 1) / N                  # cycles per lag
    z_thr = math.log(2 * M / ALPHA_FW)               # one pulse, two channels
    z_thr10 = math.log(20 * M / ALPHA_FW)            # ten pulses, two channels
    hz = lambda f: f * 1e6 / US_PER_TRIAL_NOMINAL

    print("=" * 78)
    print("PART 17 - LOMB-SCARGLE PERIODOGRAM OF delta-hat(k), ROUND 28297")
    print("=" * 78)
    print(f"  lags: {N - 1:,} (k != 0)   frequencies: {M:,} per channel, "
          f"{2 * M:,} hypotheses")
    print(f"  Bonferroni on Exp(1): single pulse (m = {2*M:,}) z_thr = "
          f"{z_thr:.3f};  ten pulses (m = {20*M:,}) z_thr = {z_thr10:.3f}")
    print(f"  nominal conversion: 1 cycle/lag = {hz(1):,.0f} Hz at "
          f"{US_PER_TRIAL_NOMINAL:g} us/trial; 50 Hz mains = period "
          f"{hz(1)/50:,.0f} lags")
    print(f"  frequency range covered: {hz(freqs[0]):.1f} Hz to "
          f"{hz(freqs[-1]):,.0f} Hz (nominal)\n")

    out = dict(N=N, M=M, alpha_fw=ALPHA_FW, z_thr=z_thr, z_thr_ten=z_thr10,
               us_per_trial_nominal=US_PER_TRIAL_NOMINAL,
               us_per_trial_bound=US_PER_TRIAL_BOUND, channels={})

    # ------------------------------------------------ the data, both channels
    deltas = {}
    power_28297 = {}
    for pair in ("OA_vs_SB", "OB_vs_SA"):
        d = np.load(os.path.join(HERE, f"meros5_delta_{pair}.npz"))
        lags, delta, sig = d["lags"], d["delta"], d["sigma"]
        nz = lags != 0
        k = lags[nz].astype(np.float64)
        z = delta[nz] / sig[nz]
        deltas[pair] = (k, z, float(sig[nz].mean()))
        t0 = time.time()
        P = ls_power(k, z, freqs)
        power_28297[pair.replace("_", " ")] = P
        j = int(np.argmax(P))
        n_above = int((P > z_thr).sum())
        label = pair.replace("_", " ")
        print(f"--- {label}  (periodogram in {time.time()-t0:.1f}s)")
        print(f"    max power = {P[j]:.3f} at f = {freqs[j]:.6f} cycles/lag, "
              f"period {1/freqs[j]:,.1f} lags, ~{hz(freqs[j]):,.1f} Hz nominal")
        p_ch = 1 - (1 - math.exp(-P[j])) ** M
        print(f"    single-pulse threshold {z_thr:.3f}: "
              f"{'ABOVE' if P[j] > z_thr else 'below'} "
              f"({P[j]/z_thr*100:.0f}%)   ten-pulse threshold {z_thr10:.3f}: "
              f"{'ABOVE' if P[j] > z_thr10 else 'below'}   "
              f"frequencies above the former: {n_above} / {M:,}")
        print(f"    p of this maximum within one channel = "
              f"1-(1-e^-P)^M = {p_ch:.4f}")
        print(f"    mean power {P.mean():.4f} (theory 1), "
              f"expected max under the null ln(M)+gamma = "
              f"{math.log(M)+0.5772:.2f}")
        # the mains line, specifically
        j50 = int(np.argmin(np.abs(hz(freqs) - 50.0)))
        print(f"    power at the nearest frequency to 50 Hz nominal "
              f"(f = {freqs[j50]:.6f}): {P[j50]:.3f}")
        top = np.argsort(P)[::-1][:5]
        out["channels"][label] = dict(
            max_power=float(P[j]), at_freq=float(freqs[j]),
            period_lags=float(1 / freqs[j]), hz_nominal=float(hz(freqs[j])),
            above=bool(P[j] > z_thr), above_ten=bool(P[j] > z_thr10),
            p_within_channel=float(p_ch), n_above=n_above,
            mean_power=float(P.mean()),
            power_at_50Hz_nominal=float(P[j50]), freq_50Hz=float(freqs[j50]),
            top5=[dict(f=float(freqs[i]), period=float(1/freqs[i]),
                       hz=float(hz(freqs[i])), power=float(P[i])) for i in top])

    # ------------------------------------------------ the other nine pulses
    # A device systematic must repeat. The frequency of the largest peak of
    # 28297 is therefore re-examined in every other pulse, and each of those
    # pulses is scanned in full on its own.
    from load_curby import read_file
    DATA = os.path.join(os.path.dirname(HERE), "dedomena_curby")
    OTHER = [1000, 15000, 22000, 23000, 26000, 28293, 28294, 28295, 28296]
    worst = max(out["channels"].items(), key=lambda kv: kv[1]["max_power"])
    f_peak = worst[1]["at_freq"]
    j_peak = int(round(f_peak * N)) - 1
    print(f"\n--- the other nine pulses, both channels; the peak frequency of "
          f"{worst[0]} (f = {f_peak:.6f}, period {1/f_peak:,.1f} lags) is "
          f"re-examined in each")
    print(f"    {'round':>6} {'channel':<9} {'P(f_peak)':>10} {'max P':>7} "
          f"{'at period':>10} {'>single':>8} {'>ten':>6}")
    for lab, P in power_28297.items():
        tag = " (the peak itself)" if lab == worst[0] else ""
        print(f"    {28297:>6} {lab:<9} {P[j_peak]:>10.2f}{tag}")
    kfull0 = np.arange(-K, K + 1)
    nz0 = kfull0 != 0
    kk0 = kfull0[nz0].astype(np.float64)
    ten = []
    for rnd in OTHER:
        data, _ = read_file(os.path.join(DATA, f"curby_round_{rnd}.bin"))
        SA = data["SA"].astype(np.int8); SB = data["SB"].astype(np.int8)
        OA = (data["OA"] > 0).astype(np.int8)
        OB = (data["OB"] > 0).astype(np.int8)
        del data
        for lab, O, S in (("OA vs SB", OA, SB), ("OB vs SA", OB, SA)):
            s1 = (S == 1).astype(np.int8)
            _, n11, A1, B1, nk, _ = scan(O, s1, K)
            _, dl = mi_and_delta(n11, A1, B1, nk)
            sd = sigma_delta(A1, B1, nk)
            P = ls_power(kk0, (dl / sd)[nz0], freqs)
            jm = int(np.argmax(P))
            print(f"    {rnd:>6} {lab:<9} {P[j_peak]:>10.2f} {P[jm]:>7.2f} "
                  f"{1/freqs[jm]:>10.1f} "
                  f"{'YES' if P[jm] > z_thr else 'no':>8} "
                  f"{'YES' if P[jm] > z_thr10 else 'no':>6}", flush=True)
            ten.append(dict(round=rnd, channel=lab,
                            power_at_peak_freq=float(P[j_peak]),
                            max_power=float(P[jm]),
                            at_freq=float(freqs[jm]),
                            period_lags=float(1 / freqs[jm]),
                            above_single=bool(P[jm] > z_thr),
                            above_ten=bool(P[jm] > z_thr10)))
        del SA, SB, OA, OB
    pk = np.array([r["power_at_peak_freq"] for r in ten])
    n_above10 = sum(r["above_ten"] for r in ten) + sum(
        c["above_ten"] for c in out["channels"].values())
    print(f"\n    power at f_peak across the other 18 pulse-channels: "
          f"mean {pk.mean():.2f} (Exp(1) mean 1), largest {pk.max():.2f}")
    print(f"    above the ten-pulse threshold, all 10 pulses x 2 channels: "
          f"{n_above10} / {20*M:,}")
    out["ten_pulse"] = dict(rows=ten, peak_freq=f_peak,
                            power_at_peak_28297={k: float(v[j_peak])
                                                 for k, v in power_28297.items()},
                            peak_channel=worst[0],
                            power_at_peak_mean=float(pk.mean()),
                            power_at_peak_max=float(pk.max()),
                            n_above_ten_total=int(n_above10))

    # ------------------------------------------------ empirical null
    print(f"\n--- empirical null: {a.perms} permutations of the lags, "
          f"both channels")
    mx = []
    for pair in deltas:
        k, z, _ = deltas[pair]
        zp = z.copy()
        for _ in range(a.perms):
            rng.shuffle(zp)
            mx.append(float(ls_power(k, zp, freqs).max()))
    mx = np.array(mx)
    p95 = float(np.percentile(mx, 95))
    frac = float((mx > z_thr).mean())
    print(f"    max power over one channel: mean {mx.mean():.2f}, "
          f"95th pct {p95:.2f}, largest {mx.max():.2f}")
    print(f"    fraction of permutations above z_thr = {z_thr:.2f}: "
          f"{frac:.3f}  (Bonferroni promises <= {ALPHA_FW/2:.3f} per channel)")
    out["null"] = dict(perms=len(mx), max_mean=float(mx.mean()),
                       max_p95=p95, max_max=float(mx.max()),
                       frac_above_thr=frac)

    # ------------------------------------------------ sensitivity in eps
    cal = json.load(open(os.path.join(HERE, "meros1_alpha.json")))["OA vs SA"]
    alpha, r1, r2 = cal["alpha"], cal["r1"], cal["r2"]
    sig_mean = deltas["OA_vs_SB"][2]
    eps_thr = math.sqrt(4 * z_thr / (N - 1)) * sig_mean / alpha
    print(f"\n--- sensitivity: a sinusoidal kernel of unit peak and amplitude "
          f"eps reaches the threshold at")
    print(f"    eps_thr = sqrt(4 z_thr/N) sigma_delta/alpha = {eps_thr:.3e}   "
          f"(sigma_delta = {sig_mean:.3e}, alpha = {alpha:.4e})")
    out["eps_threshold_oscillatory"] = eps_thr

    # ------------------------------------------------ positive control
    d = np.load(os.path.join(HERE, "curby_28297.npz"))
    SA, SB = d["SA"], d["SB"]
    n = len(SA)
    S = np.where(SB == 2, 1.0, -1.0)
    SB1 = (SB == 1).astype(np.int8)
    lam0 = np.where(SA == 1, r1, r2)
    kfull = np.arange(-K, K + 1)
    nzmask = kfull != 0
    print(f"\n--- positive control: outcome-level injection of "
          f"cos(2 pi f0 k + phi), {a.reps} phases per point")
    print(f"    {'period':>8} {'~Hz':>10} {'eps/eps_thr':>12} {'detected':>9} "
          f"{'peak at f0':>11} {'mean power':>11}")
    ctrl = []
    for period in (7.0, 137.0, 5000.0):
        f0 = 1.0 / period
        j0 = int(np.argmin(np.abs(freqs - f0)))
        for frac in (1.0, 2.0):
            eps = frac * eps_thr
            powers, hits, at_f0 = [], 0, 0
            for r in range(a.reps):
                phi = rng.uniform(0, 2 * np.pi)
                F = build_F_cos(S, f0, phi, K)
                lam = np.clip(lam0 + alpha * eps * F, 0, 1)
                O = (rng.random(n) < lam).astype(np.int8)
                _, n11, A1, B1, nk, _ = scan(O, SB1, K)
                _, dl = mi_and_delta(n11, A1, B1, nk)
                sd = sigma_delta(A1, B1, nk)
                zz = (dl / sd)[nzmask]
                P = ls_power(kfull[nzmask].astype(np.float64), zz, freqs)
                jm = int(np.argmax(P))
                powers.append(float(P[j0]))
                hits += int(P[jm] > z_thr)
                at_f0 += int(abs(jm - j0) <= 1)
                del F, lam, O
            print(f"    {period:>8g} {hz(f0):>10.1f} {frac:>12g} "
                  f"{hits:>6}/{a.reps} {at_f0:>8}/{a.reps} "
                  f"{np.mean(powers):>11.1f}")
            ctrl.append(dict(period=period, f0=f0, hz_nominal=hz(f0),
                             eps=eps, frac=frac, detected=hits,
                             peak_at_f0=at_f0, reps=a.reps,
                             mean_power_at_f0=float(np.mean(powers)),
                             powers=powers))
    out["positive_control"] = ctrl
    ok2 = all(c["detected"] == c["reps"] and c["peak_at_f0"] == c["reps"]
              for c in ctrl if c["frac"] == 2.0)

    # ------------------------------------------------ verdict
    clean = out["ten_pulse"]["n_above_ten_total"] == 0
    clean1 = all(not c["above"] for c in out["channels"].values())
    print("\n" + "=" * 78)
    print(f"  round 28297, single-pulse family: "
          f"{'nothing above the threshold' if clean1 else 'one frequency above the threshold'}")
    print(f"  all ten pulses, ten-pulse family: "
          f"{'NOTHING above the threshold' if clean else '*** PEAK ABOVE THE THRESHOLD - STOP ***'}")
    print(f"  positive control at 2 eps_thr: "
          f"{'found at f0 every time' if ok2 else '*** NOT always recovered ***'}")
    print("=" * 78)
    out["clean"] = bool(clean)
    out["clean_single_pulse"] = bool(clean1)
    out["control_ok"] = bool(ok2)
    json.dump(out, open(os.path.join(HERE, "meros17_periodogram.json"), "w"),
              indent=2)
    print("\nSaved: meros17_periodogram.json")


if __name__ == "__main__":
    main()
