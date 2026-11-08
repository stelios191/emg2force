"""Checksum-verified download of the Hyser subsets the pipeline needs (random-task EMG + force, MVC force).

python -m emg2force.download --root ../hyser_raw --subset random mvc [--subjects 1 2] [--workers 8]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from emg2force.provenance import REPO

PHYSIONET_URL = "https://physionet.org/files/hd-semg/2.0.0/"
TIMEOUT_S = 60  # per request. A stalled download fails and is retried
# PhysioNet's AWS open-data mirror. ~100x faster here. Every file is still checked against PhysioNet's SHA256SUMS.
MIRROR_URL = "https://physionet-open.s3.amazonaws.com/hd-semg/2.0.0/"

SUBSET_PATTERNS = {
    "random": r"^random_dataset/subject\d+_session\d+/random_(raw|force)_sample\d+\.(dat|hea)$",
    "mvc": r"^mvc_dataset/subject\d+_session\d+/mvc_force_finger\d_(extension|flexion)\.(dat|hea)$",
}


CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(root: Path) -> dict[str, str]:
    sums = root / "meta" / "SHA256SUMS.txt"
    if not sums.exists():
        sums.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching the checksum list from {PHYSIONET_URL} (7 MB, can take minutes) ...", flush=True)
        tmp = sums.with_suffix(".part")
        urllib.request.urlretrieve(PHYSIONET_URL + "SHA256SUMS.txt", tmp)
        tmp.rename(sums)  # only a complete list is ever reused
    manifest = {}
    for line in sums.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        manifest[rel] = digest
    return manifest


def select(manifest: dict[str, str], subsets: list[str], subjects: list[int] | None) -> list[str]:
    pats = [re.compile(SUBSET_PATTERNS[s]) for s in subsets]
    out = []
    for rel in manifest:
        if not any(p.match(rel) for p in pats):
            continue
        if subjects is not None:
            m = re.search(r"subject(\d+)_", rel)
            if m is not None and int(m.group(1)) not in subjects:
                continue
        out.append(rel)
    return sorted(out)


def fetch(root: Path, rel: str, digest: str, base_url: str, retries: int = 3) -> tuple[str, str]:
    dst = root / rel
    if dst.exists() and sha256_file(dst) == digest:
        return rel, "cached"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    err = "no attempt made (retries < 1)"
    for _ in range(retries):
        try:
            with urllib.request.urlopen(base_url + rel, timeout=TIMEOUT_S) as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh)
            if sha256_file(tmp) != digest:
                raise OSError("checksum mismatch")
            tmp.replace(dst)
            return rel, "ok"
        except Exception as e:
            err = str(e)
    tmp.unlink(missing_ok=True)
    return rel, f"FAILED: {err}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO.parent / "hyser_raw", help="Hyser download folder (default: ../hyser_raw)")
    ap.add_argument("--subset", nargs="+", default=["random", "mvc"], choices=sorted(SUBSET_PATTERNS))
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base-url", default=MIRROR_URL, help=f"file source (PhysioNet itself: {PHYSIONET_URL})")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.root)
    files = select(manifest, args.subset, args.subjects)
    print(f"{len(files)} files selected", flush=True)
    failed = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, args.root, rel, manifest[rel], args.base_url) for rel in files]
        for i, fut in enumerate(as_completed(futs), 1):
            rel, status = fut.result()
            if status.startswith("FAILED"):
                failed.append(rel)
            if i % 50 == 0 or status.startswith("FAILED"):
                print(f"[{i}/{len(files)}] {rel}: {status}", flush=True)
    print(f"done, {len(failed)} failed", flush=True)
    for rel in failed:
        print("  FAILED", rel)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
