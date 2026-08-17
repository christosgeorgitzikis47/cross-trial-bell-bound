"""
Α1 — Χαρτογράφηση διαθέσιμων γύρων CURBy Q.

ΜΟΝΟ μεταδεδομένα (`/api/curbyq/round/{N}`, ~10 KB), ΠΟΤΕ `/data` (9 MB).
Ίδια όρια με το katevasma.py: 2 δευτ. παύση, 403/429 -> ΑΜΕΣΗ ΔΙΑΚΟΠΗ.

Χρήση:
    python3 xartografisi.py --probe 28297 30000 40000        # σκέτα probe
    python3 xartografisi.py --bisect-max 28297 100000        # πάνω άκρο
    python3 xartografisi.py --bisect-min 1 28297             # κάτω άκρο
Αποθηκεύει ό,τι βρίσκει στο xartografisi_cache.json ώστε να μην
ξαναρωτηθεί ο server για τον ίδιο γύρο.
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
        sys.exit(f"ΣΤΑΜΑΤΗΜΑ: ο server επέστρεψε {st} στον γύρο {rnd}. "
                 f"Καμία προσπάθεια παράκαμψης.")
    if st != 200:
        rec = {"round": rnd, "ok": False, "status": st}
    else:
        try:
            stages, params = parse(body)
            ts = (stages.get("request") or {}).get("timestamp")
            # ΠΡΟΣΟΧΗ: ανύπαρκτος γύρος -> HTTP 200 με ΚΕΝΟ πίνακα, όχι 404.
            # «Υπάρχει» σημαίνει: έχει request timestamp ΚΑΙ randomness stage
            # (δηλ. ολοκληρωμένος γύρος με διαθέσιμα δεδομένα).
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
        print(f"  γύρος {rec['round']:>7}: OK   {rec.get('timestamp')}   "
              f"isQuantum={p.get('isQuantum')} stop={p.get('stoppingCriteria')} "
              f"nTrialsNeeded={p.get('nTrialsNeeded')}")
    else:
        print(f"  γύρος {rec['round']:>7}: -- status={rec['status']}"
              + (f"  {rec.get('error','')}" if rec.get("error") else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, nargs="+")
    ap.add_argument("--bisect-max", type=int, nargs=2, metavar=("GOOD", "BAD"))
    ap.add_argument("--bisect-min", type=int, nargs=2, metavar=("BAD", "GOOD"))
    ap.add_argument("--budget", type=int, default=25,
                    help="μέγιστα αιτήματα σε αυτό το τρέξιμο")
    a = ap.parse_args()

    cache = load_cache()
    used = [0]

    def P(r):
        if str(r) not in cache:
            if used[0] >= a.budget:
                print(f"  (όριο {a.budget} αιτημάτων — σταμάτημα)")
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
        print(f"Δυαδική αναζήτηση πάνω άκρου: γνωστό OK={good}, δοκιμή έως {bad}")
        rec = P(bad)
        if rec and rec["ok"]:
            print(f"  ! ο {bad} υπάρχει — ανέβασε το πάνω όριο")
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
            print(f"\n  ΜΕΓΙΣΤΟΣ ΓΥΡΟΣ = {good}   (ο {bad} δεν υπάρχει)")

    if a.bisect_min:
        bad, good = a.bisect_min
        print(f"Δυαδική αναζήτηση κάτω άκρου: δοκιμή από {bad}, γνωστό OK={good}")
        rec = P(bad)
        if rec and rec["ok"]:
            print(f"  ! ο {bad} υπάρχει — κατέβασε το κάτω όριο")
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
            print(f"\n  ΕΛΑΧΙΣΤΟΣ ΓΥΡΟΣ = {good}   (ο {bad} δεν υπάρχει)")

    print(f"\nΑιτήματα σε αυτό το τρέξιμο: {used[0]}   "
          f"σύνολο στην cache: {len(cache)}")


if __name__ == "__main__":
    main()
