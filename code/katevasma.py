"""
Download raw CURBy beacon data - under STRICT limits.

  * one download at a time, a 2 second pause between them
  * a HARD LIMIT of 5 files per run (--max)
  * 403/429 -> IMMEDIATE STOP, no attempt to work around it
  * manifest.json: round, url, size, sha256, download date

Usage:  python3 katevasma.py --rounds 28297 28296 28295 28294 28293
"""
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone

BASE = "https://random.colorado.edu"
UA = "curby-research-client/1.0 (academic replication)"
PAUSE = 2.0
HARD_LIMIT = 5

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifest.json")


def decompress_if_needed(raw):
    """The server (Cloudflare) serves cached responses as zstd or gzip, and
    libcurl 8.7.1 does not decode zstd. We detect the format from the magic
    bytes and decompress it ourselves. The RAW DATA (/data) arrives
    uncompressed and does not pass through here."""
    if raw[:4] == b"\x28\xb5\x2f\xfd":                     # zstd
        r = subprocess.run(["zstd", "-dc"], input=raw, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError("zstd decompression failed: "
                               + r.stderr.decode("utf-8", "replace")[:200])
        return r.stdout
    if raw[:2] == b"\x1f\x8b":                             # gzip
        import gzip
        return gzip.decompress(raw)
    return raw


def fetch(path, out_path=None):
    """Returns (status, bytes|None). Writes to a file if out_path is given.

    --compressed is NOT used: libcurl advertises encodings it cannot decode
    (br/zstd) and then fails with error 56. Instead we take whatever the
    server sends and decompress it ourselves."""
    cmd = ["curl", "-sS", "-A", UA, "--max-time", "180",
           "-w", "%{http_code}"]
    if out_path:
        cmd += ["-o", out_path]
    cmd.append(BASE + path)
    r = subprocess.run(cmd, capture_output=True, timeout=240)
    if r.returncode != 0:
        return None, r.stderr.decode("utf-8", "replace")[:300]
    if out_path:
        return int(r.stdout.decode("ascii", "replace").strip()[-3:]), None
    status = int(r.stdout[-3:].decode("ascii", "replace"))
    return status, decompress_if_needed(r.stdout[:-3])


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest():
    if os.path.exists(MANIFEST):
        return json.load(open(MANIFEST))
    return {"source": BASE, "api": "CURBy Twine Spool 1.0.0", "pulses": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", required=True)
    ap.add_argument("--max", type=int, default=HARD_LIMIT)
    args = ap.parse_args()

    n = min(args.max, HARD_LIMIT)
    rounds = args.rounds[:n]
    if len(args.rounds) > n:
        print(f"NOTE: {len(args.rounds)} requested, the hard limit is {n}. "
              f"Only these will be downloaded: {rounds}")

    man = load_manifest()
    have = {p["round"] for p in man["pulses"]}

    for i, rnd in enumerate(rounds):
        if rnd in have:
            print(f"[{i+1}/{len(rounds)}] round {rnd}: already present, skipped")
            continue

        # --- 1. metadata (a small JSON) ---
        st, body = fetch(f"/api/curbyq/round/{rnd}")
        if st in (403, 429):
            sys.exit(f"\nSTOPPING: the server returned {st} for the metadata "
                     f"of round {rnd}. No attempt to work around it.")
        if st != 200:
            print(f"[{i+1}] round {rnd}: metadata status={st}, skipped")
            continue
        meta = json.loads(body)
        params, stages = {}, {}
        for rec in meta:
            c = rec["data"]["content"]
            p = c.get("payload", {})
            stages[p.get("stage")] = {"index": c.get("index"),
                                      "timestamp": p.get("timestamp")}
            if p.get("stage") == "request":
                params = p.get("parameters", {})
        time.sleep(PAUSE)

        # --- 2. raw data ---
        fname = f"curby_round_{rnd}.bin"
        fpath = os.path.join(HERE, fname)
        url = f"/api/curbyq/round/{rnd}/data"
        print(f"[{i+1}/{len(rounds)}] round {rnd}: downloading...", end=" ", flush=True)
        st, err = fetch(url, out_path=fpath)
        if st in (403, 429):
            if os.path.exists(fpath):
                os.remove(fpath)
            sys.exit(f"\nSTOPPING: the server returned {st} for round {rnd}. "
                     f"No attempt to work around it.")
        if st != 200:
            if os.path.exists(fpath):
                os.remove(fpath)
            print(f"status={st}, skipped")
            continue

        size = os.path.getsize(fpath)
        digest = sha256_of(fpath)
        print(f"{size:,} bytes  sha256={digest[:16]}…")

        man["pulses"].append({
            "round": rnd,
            "url": BASE + url,
            "file": fname,
            "bytes": size,
            "sha256": digest,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "stages": stages,
            "nTrialsNeeded": params.get("nTrialsNeeded"),
            "stoppingCriteria": params.get("stoppingCriteria"),
            "nBitsOut": params.get("nBitsOut"),
            "seedLength": params.get("seedLength"),
            "isQuantum": params.get("isQuantum"),
        })
        with open(MANIFEST, "w") as f:
            json.dump(man, f, indent=2)

        if i < len(rounds) - 1:
            time.sleep(PAUSE)

    total = sum(p["bytes"] for p in man["pulses"])
    print(f"\nTotal in the manifest: {len(man['pulses'])} pulses, "
          f"{total/1e6:.1f} MB")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
