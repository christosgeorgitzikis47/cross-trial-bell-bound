"""
A1 - Mapping the available CURBy Q rounds.

METADATA ONLY (`/api/curbyq/round/{N}`, ~10 KB), NEVER `/data` (9 MB).
The same limits as katevasma.py: a 2 second pause, 403/429 -> IMMEDIATE STOP.

Usage:
    python3 xartografisi.py --probe 28297 30000 40000        # plain probes
    python3 xartografisi.py --bisect-max 28297 100000        # upper end
    python3 xartografisi.py --bisect-min 1 28297             # lower end
Whatever it finds is stored in xartografisi_cache.json so that the server is
never asked about the same round twice.
"""
import argparse, json, os, subprocess, sys, time

BASE = "https://random.colorado.edu"
UA = "curby-research-client/1.0 (academic replication)"
PAUSE = 2.0
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "xartografisi_cache.json")


def load_cache():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    return {}


def save_cache(c):
    with open(CACHE, "w") as f:
        json.dump(c, f, indent=1, sort_keys=True)


def fetch_meta(rnd):
    cmd = ["curl", "-sS", "-A", UA, "--max-time", "120", "-w", "%{http_code}",
           f"{BASE}/api/curbyq/round/{rnd}"]
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    if r.returncode != 0:
        return None, r.stderr.decode("utf-8", "replace")[:200]
    st = int(r.stdout[-3:].decode("ascii", "replace"))
    return st, r.stdout[:-3]


def parse(body):
    """-> {stage: {index, timestamp}}, parameters"""
    meta = json.loads(body)
    stages, params = {}, {}
    for rec in meta:
        c = rec["data"]["content"]
        p = c.get("payload", {})
        stages[p.get("stage")] = {"index": c.get("index"),
                                  "timestamp": p.get("timestamp")}
        if p.get("stage") == "request":
            params = p.get("parameters", {})
    return stages, params


def probe(rnd, cache):
    key = str(rnd)
    if key in cache:
        return cache[key]
    st, body = fetch_meta(rnd)
    if st in (403, 429):
        save_cache(cache)
        sys.exit(f"STOPPING: the server returned {st} for round {rnd}. "
                 f"No attempt to work around it.")
    if st != 200:
        rec = {"round": rnd, "ok": False, "status": st}
    else:
        try:
            stages, params = parse(body)
            ts = (stages.get("request") or {}).get("timestamp")
            # NOTE: a nonexistent round -> HTTP 200 with an EMPTY array, not
            # 404. "Exists" means: it has a request timestamp AND a randomness
            # stage (that is, a completed round with data available).
            ok = bool(ts) and "randomness" in stages
            rec = {"round": rnd, "ok": ok, "status": 200, "timestamp": ts,
                   "stages": stages, "parameters": params}
        except Exception as e:
            rec = {"round": rnd, "ok": False, "status": st,
                   "error": f"{type(e).__name__}: {e}",
                   "head": body[:200].decode("utf-8", "replace")}
    cache[key] = rec
    save_cache(cache)
    time.sleep(PAUSE)
    return rec


def show(rec):
    if rec["ok"]:
        p = rec.get("parameters", {})
        print(f"  round {rec['round']:>7}: OK   {rec.get('timestamp')}   "
              f"isQuantum={p.get('isQuantum')} stop={p.get('stoppingCriteria')} "
              f"nTrialsNeeded={p.get('nTrialsNeeded')}")
    else:
        print(f"  round {rec['round']:>7}: -- status={rec['status']}"
              + (f"  {rec.get('error','')}" if rec.get("error") else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, nargs="+")
    ap.add_argument("--bisect-max", type=int, nargs=2, metavar=("GOOD", "BAD"))
    ap.add_argument("--bisect-min", type=int, nargs=2, metavar=("BAD", "GOOD"))
    ap.add_argument("--budget", type=int, default=25,
                    help="maximum number of requests in this run")
    a = ap.parse_args()

    cache = load_cache()
    used = [0]

    def P(r):
        if str(r) not in cache:
            if used[0] >= a.budget:
                print(f"  (budget of {a.budget} requests reached - stopping)")
                return None
            used[0] += 1
        rec = probe(r, cache)
        show(rec)
        return rec

    if a.probe:
        for r in a.probe:
            if P(r) is None:
                break

    if a.bisect_max:
        good, bad = a.bisect_max
        print(f"Bisecting the upper end: known OK={good}, testing up to {bad}")
        rec = P(bad)
        if rec and rec["ok"]:
            print(f"  ! {bad} exists - raise the upper limit")
        else:
            while bad - good > 1:
                mid = (good + bad) // 2
                rec = P(mid)
                if rec is None:
                    break
                if rec["ok"]:
                    good = mid
                else:
                    bad = mid
            print(f"\n  LARGEST ROUND = {good}   ({bad} does not exist)")

    if a.bisect_min:
        bad, good = a.bisect_min
        print(f"Bisecting the lower end: testing from {bad}, known OK={good}")
        rec = P(bad)
        if rec and rec["ok"]:
            print(f"  ! {bad} exists - lower the lower limit")
        else:
            while good - bad > 1:
                mid = (good + bad) // 2
                rec = P(mid)
                if rec is None:
                    break
                if rec["ok"]:
                    good = mid
                else:
                    bad = mid
            print(f"\n  SMALLEST ROUND = {good}   ({bad} does not exist)")

    print(f"\nRequests in this run: {used[0]}   "
          f"total in the cache: {len(cache)}")


if __name__ == "__main__":
    main()
