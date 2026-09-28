"""Fetch a prefix from a PUBLIC GCS bucket over HTTPS (no gsutil auth).

Lists objects via the JSON API and downloads them with resume + size check.
Used for droid_100 (gs://gresearch/robotics/droid_100) and openpi
checkpoints (gs://openpi-assets/...), whose buckets allow anonymous reads.

Usage: python run/gcs_fetch.py --bucket gresearch --prefix robotics/droid_100 \
           --dest ${SIMANY_ROOT}/data/droid [--list-only] [--max-gb 50]
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
DATA = "https://storage.googleapis.com/{bucket}/{name}"


def list_objects(bucket, prefix):
    items, token = [], None
    while True:
        q = {"prefix": prefix, "maxResults": "1000"}
        if token:
            q["pageToken"] = token
        url = API.format(bucket=bucket) + "?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.load(r)
        items += page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            return items


def fetch(bucket, name, size, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have == size:
        return "ok"
    url = DATA.format(bucket=bucket, name=urllib.parse.quote(name))
    req = urllib.request.Request(url)
    mode = "wb"
    if 0 < have < size:
        req.add_header("Range", f"bytes={have}-")
        mode = "ab"
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, mode) as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    return "fetched" if dest.stat().st_size == size else "SHORT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--max-gb", type=float, default=100.0,
                    help="refuse to download more than this")
    args = ap.parse_args()

    items = list_objects(args.bucket, args.prefix)
    total = sum(int(i["size"]) for i in items)
    print(f"{len(items)} objects, {total/1e9:.2f} GB under "
          f"gs://{args.bucket}/{args.prefix}", flush=True)
    if args.list_only:
        for i in items[:20]:
            print(f"  {int(i['size'])/1e6:9.1f} MB  {i['name']}")
        return
    if total / 1e9 > args.max_gb:
        sys.exit(f"refusing: {total/1e9:.1f} GB > --max-gb {args.max_gb}")
    done = 0
    t0 = time.time()
    for k, it in enumerate(items):
        rel = it["name"][len(args.prefix):].lstrip("/")
        st = fetch(args.bucket, it["name"], int(it["size"]),
                   Path(args.dest) / rel)
        done += int(it["size"])
        if st != "ok" and (k % 20 == 0 or done == total):
            rate = done / max(time.time() - t0, 1) / 1e6
            print(f"  [{k+1}/{len(items)}] {done/1e9:.2f}/{total/1e9:.2f} GB "
                  f"({rate:.0f} MB/s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
