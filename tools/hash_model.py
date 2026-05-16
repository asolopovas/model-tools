#!/usr/bin/env python3
"""Hash large model artifacts in a streaming way."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def hash_file(path: Path, algorithm: str = "sha256", chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--algorithm", default="sha256")
    args = parser.parse_args()

    for file in args.files:
        print(f"{hash_file(file, args.algorithm)}  {file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
