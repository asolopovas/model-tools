#!/usr/bin/env python3
"""Cross-platform workspace helpers used by the justfile.

Keep shell-specific filesystem operations out of recipes so the same commands work
from Windows cmd/PowerShell, macOS, and Linux shells.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


STANDARD_DIRS = [
    "input/whisper-large-v3-turbo",
    "input/hf",
    "output/onnx",
    "output/static",
    "output/simplified",
    "output/quantized",
    "output/samsung",
    "output/reports",
    "configs",
    "docs",
]


def human_size(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def command_setup(args: argparse.Namespace) -> int:
    dirs = args.dirs or STANDARD_DIRS
    for item in dirs:
        Path(item).mkdir(parents=True, exist_ok=True)
    return 0


def command_require_file(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    return 0


def command_ensure_parent(args: argparse.Namespace) -> int:
    Path(args.path).parent.mkdir(parents=True, exist_ok=True)
    return 0


def command_ls_size(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Missing path: {path}")
    print(f"{human_size(path.stat().st_size):>10}  {path}")
    return 0


def command_remove(args: argparse.Namespace) -> int:
    for item in args.paths:
        path = Path(item)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    return 0


def command_upload_list(args: argparse.Namespace) -> int:
    print("Upload/convert this first in Samsung SDK Service:")
    print(f"  {args.static_model}")
    print()
    print("Keep tokenizer/preprocessor files with your app/runtime as needed; they are not NPU binaries:")
    print(f"  {args.input_dir}/config.json")
    print("  tokenizer.json tokenizer_config.json generation_config.json")
    print("  preprocessor_config.json special_tokens_map.json normalizer.json vocab.json merges.txt added_tokens.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-platform workspace helper commands.")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="Create standard workspace directories.")
    setup.add_argument("dirs", nargs="*")
    setup.set_defaults(func=command_setup)

    require_file = sub.add_parser("require-file", help="Fail if a file does not exist.")
    require_file.add_argument("path")
    require_file.set_defaults(func=command_require_file)

    ensure_parent = sub.add_parser("ensure-parent", help="Create a path's parent directory.")
    ensure_parent.add_argument("path")
    ensure_parent.set_defaults(func=command_ensure_parent)

    ls_size = sub.add_parser("ls-size", help="Print a compact file size line.")
    ls_size.add_argument("path")
    ls_size.set_defaults(func=command_ls_size)

    remove = sub.add_parser("remove", help="Remove files/directories if they exist.")
    remove.add_argument("paths", nargs="+")
    remove.set_defaults(func=command_remove)

    upload_list = sub.add_parser("upload-list", help="Print Samsung SDK upload guidance.")
    upload_list.add_argument("static_model")
    upload_list.add_argument("--input-dir", default="input/whisper-large-v3-turbo")
    upload_list.set_defaults(func=command_upload_list)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
