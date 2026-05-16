#!/usr/bin/env python3
"""Small Hugging Face Hub helper for clean model conversion workspaces.

The goal is to search/list/download models without creating random files in the
repo root. Downloads go to input/hf/<sanitized-repo-id>/ by default and Hub
metadata folders are removed after a successful download unless requested.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError as exc:  # pragma: no cover - friendly CLI error
    raise SystemExit(
        "Missing dependency: huggingface_hub. Run through `just hf-search ...` "
        "or install with `uv add huggingface-hub`."
    ) from exc


MINIMAL_PATTERNS = [
    "README.md",
    "*.json",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "merges.txt",
    "vocab.*",
    "tokenizer.*",
    "preprocessor_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "normalizer.json",
    "added_tokens.json",
]

CONVERSION_PATTERNS = [
    *MINIMAL_PATTERNS,
    "*.safetensors",
    "*.onnx",
]

# Default conversion preset avoids legacy/alternate huge weight formats. Use
# --preset full or explicit --include/--exclude when you really want them.
CONVERSION_IGNORE = [
    "*.bin",
    "*.ckpt",
    "*.pt",
    "*.pth",
    "*.h5",
    "*.msgpack",
    "*.gguf",
    "*.tflite",
    "*.mlpackage/**",
]

PRESETS: dict[str, tuple[list[str] | None, list[str] | None]] = {
    "minimal": (MINIMAL_PATTERNS, None),
    "conversion": (CONVERSION_PATTERNS, CONVERSION_IGNORE),
    "full": (None, None),
}


def token_arg() -> str | bool | None:
    """Use HF_TOKEN if provided, otherwise let huggingface_hub use saved auth."""
    return os.environ.get("HF_TOKEN") or None


def sanitize_repo_id(repo_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id.strip())
    return safe.strip("._-") or "model"


def human_size(size: int | None) -> str:
    if size is None:
        return "?"
    value = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def print_rows(headers: list[str], rows: list[list[Any]]) -> None:
    widths = [len(h) for h in headers]
    string_rows = [[str(cell) for cell in row] for row in rows]
    for row in string_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in string_rows:
        print(fmt.format(*row))


def command_search(args: argparse.Namespace) -> int:
    api = HfApi(token=token_arg())
    models = list(
        api.list_models(
            search=args.query,
            filter=args.filter or None,
            sort=args.sort,
            direction=-1,
            limit=args.limit,
            full=False,
        )
    )
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": model.id,
                        "downloads": getattr(model, "downloads", None),
                        "likes": getattr(model, "likes", None),
                        "pipeline_tag": getattr(model, "pipeline_tag", None),
                        "private": getattr(model, "private", None),
                    }
                    for model in models
                ],
                indent=2,
            )
        )
        return 0

    rows = [
        [
            model.id,
            getattr(model, "pipeline_tag", "") or "",
            getattr(model, "downloads", "") or 0,
            getattr(model, "likes", "") or 0,
        ]
        for model in models
    ]
    print_rows(["model_id", "task", "downloads", "likes"], rows)
    return 0


def command_info(args: argparse.Namespace) -> int:
    api = HfApi(token=token_arg())
    info = api.model_info(args.repo_id, revision=args.revision, files_metadata=True)
    data = {
        "id": info.id,
        "sha": info.sha,
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "private": info.private,
        "gated": getattr(info, "gated", None),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "library_name": getattr(info, "library_name", None),
        "tags": getattr(info, "tags", None),
        "siblings": len(info.siblings or []),
    }
    print(json.dumps(data, indent=2, default=str))
    return 0


def command_files(args: argparse.Namespace) -> int:
    api = HfApi(token=token_arg())
    info = api.model_info(args.repo_id, revision=args.revision, files_metadata=True)
    siblings = sorted(info.siblings or [], key=lambda item: item.rfilename)
    rows: list[list[Any]] = []
    total = 0
    for sibling in siblings[: args.limit if args.limit else None]:
        size = getattr(sibling, "size", None)
        if isinstance(size, int):
            total += size
        rows.append([sibling.rfilename, human_size(size)])
    print(f"repo: {args.repo_id}")
    print(f"revision: {args.revision or info.sha}")
    print(f"files shown: {len(rows)} / {len(siblings)}")
    if rows:
        print_rows(["file", "size"], rows)
    if total:
        print(f"shown total: {human_size(total)}")
    return 0


def command_download(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    target = output_dir / sanitize_repo_id(args.repo_id)
    target.mkdir(parents=True, exist_ok=True)

    if args.include:
        allow_patterns = args.include
        ignore_patterns = args.exclude or None
        preset = "custom"
    else:
        allow_patterns, default_ignore = PRESETS[args.preset]
        ignore_patterns = args.exclude if args.exclude else default_ignore
        preset = args.preset

    api = HfApi(token=token_arg())
    info = api.model_info(args.repo_id, revision=args.revision, files_metadata=False)

    print(f"repo: {args.repo_id}")
    print(f"revision: {args.revision or info.sha}")
    print(f"target: {target}")
    print(f"preset: {preset}")
    if allow_patterns:
        print("include:", ", ".join(allow_patterns))
    if ignore_patterns:
        print("exclude:", ", ".join(ignore_patterns))

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=target,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        cache_dir=args.cache_dir,
        token=token_arg(),
        max_workers=args.max_workers,
    )

    metadata_dir = target / ".cache"
    if args.clean_metadata and metadata_dir.exists():
        shutil.rmtree(metadata_dir)

    manifest = {
        "repo_id": args.repo_id,
        "revision_requested": args.revision,
        "revision_resolved": info.sha,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "preset": preset,
        "include": allow_patterns,
        "exclude": ignore_patterns,
        "snapshot_path": str(snapshot_path),
        "cache_dir": args.cache_dir,
    }
    (target / "MODEL_SOURCE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    files = [p for p in target.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"downloaded files: {len(files)}")
    print(f"local size: {human_size(total)}")
    print(f"manifest: {target / 'MODEL_SOURCE.json'}")
    return 0


def command_local(args: argparse.Namespace) -> int:
    root = Path(args.output_dir)
    manifests = sorted(root.glob("*/MODEL_SOURCE.json"))
    if not manifests:
        print(f"No local Hugging Face model manifests found under {root}")
        return 0
    rows = []
    for manifest_path in manifests:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append(
            [
                manifest_path.parent.name,
                data.get("repo_id", ""),
                data.get("revision_resolved", "")[:12],
                data.get("preset", ""),
            ]
        )
    print_rows(["folder", "repo_id", "sha", "preset"], rows)
    return 0


def command_auth(args: argparse.Namespace) -> int:
    api = HfApi(token=token_arg())
    try:
        who = api.whoami()
    except Exception as exc:  # noqa: BLE001 - CLI diagnostic
        print("Not authenticated, or token is invalid.")
        print("Set HF_TOKEN or run: hf auth login")
        print(f"detail: {exc}")
        return 0
    print(json.dumps(who, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hugging Face Hub helpers for model conversion.")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="Search Hub models.")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--sort", default="downloads")
    search.add_argument("--filter", action="append", default=[])
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=command_search)

    info = sub.add_parser("info", help="Show model repository metadata.")
    info.add_argument("repo_id")
    info.add_argument("--revision", default=None)
    info.set_defaults(func=command_info)

    files = sub.add_parser("files", help="List repository files without downloading them.")
    files.add_argument("repo_id")
    files.add_argument("--revision", default=None)
    files.add_argument("--limit", type=int, default=80, help="0 means no limit.")
    files.set_defaults(func=command_files)

    download = sub.add_parser("download", help="Download a clean local model folder.")
    download.add_argument("repo_id")
    download.add_argument("--revision", default=None)
    download.add_argument("--output-dir", default="input/hf")
    download.add_argument("--preset", choices=sorted(PRESETS), default="conversion")
    download.add_argument("--include", action="append", default=[], help="Override preset; glob pattern to include. Repeatable.")
    download.add_argument("--exclude", action="append", default=[], help="Glob pattern to exclude. Repeatable.")
    download.add_argument("--cache-dir", default=None, help="Optional external HF cache dir. Default uses normal user cache.")
    download.add_argument("--max-workers", type=int, default=8)
    download.add_argument("--clean-metadata", dest="clean_metadata", action="store_true", default=True)
    download.add_argument("--keep-metadata", dest="clean_metadata", action="store_false")
    download.set_defaults(func=command_download)

    local = sub.add_parser("local", help="List downloaded local model folders.")
    local.add_argument("--output-dir", default="input/hf")
    local.set_defaults(func=command_local)

    auth = sub.add_parser("auth", help="Check Hugging Face authentication.")
    auth.set_defaults(func=command_auth)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - concise CLI failure
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
