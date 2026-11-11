"""Print SHA-256 checksums of the datasets in a data folder, as Markdown rows.

Usage:  python scripts/checksums.py "D:\\path\\to\\Dataset"
Read-only. For a folder dataset (Five Cities, OpenAQ) the checksum covers every file
in it (relative path + file hash, sorted).
"""
import glob
import hashlib
import os
import sys

PATTERNS = {
    "beijing_multisite": "beijing+multi*.zip",
    "beijing_pm25": "beijing+pm2*.zip",
    "five_cities": "pm2+5+data+of+five+chinese+cities",
    "india_city_day": "archive.zip",
    "epa_daily": "daily_88101_*.zip",
    "openaq": "data",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(path):
    if os.path.isfile(path):
        return sha(path)
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True)):
        if os.path.isfile(p):
            h.update(os.path.relpath(p, path).replace("\\", "/").encode())
            h.update(sha(p).encode())
    return h.hexdigest()


root = sys.argv[1] if len(sys.argv) > 1 else "."
print("| dataset | file or folder | SHA-256 |\n| --- | --- | --- |")
for name, pat in PATTERNS.items():
    hits = sorted(glob.glob(os.path.join(root, pat)))
    if not hits:
        print(f"| {name} | (not found) | |")
        continue
    print(f"| {name} | {os.path.basename(hits[0])} | {digest(hits[0])} |")
