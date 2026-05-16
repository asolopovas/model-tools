#!/usr/bin/env python3
"""Documentation consistency checks.

This intentionally stays small: it catches the main drift that hurts this repo,
namely changing just recipes without updating the command reference.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REQUIRED_FILES = [
    Path("README.md"),
    Path("AGENTS.md"),
    Path("docs/README.md"),
    Path("docs/commands.md"),
    Path("docs/cross-platform.md"),
    Path("docs/huggingface.md"),
    Path("docs/model-conversion.md"),
    Path("docs/maintenance.md"),
]


def just_recipes() -> list[str]:
    output = subprocess.check_output(["just", "--summary"], text=True)
    return sorted(output.split())


def main() -> int:
    missing = [str(path) for path in REQUIRED_FILES if not path.is_file()]
    if missing:
        print("Missing documentation files:")
        for path in missing:
            print(f"  {path}")
        return 1

    commands_doc = Path("docs/commands.md").read_text(encoding="utf-8")
    missing_recipes = [recipe for recipe in just_recipes() if f"`just {recipe}" not in commands_doc]
    if missing_recipes:
        print("docs/commands.md is missing just recipes:")
        for recipe in missing_recipes:
            print(f"  just {recipe}")
        return 1

    print("Docs check OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
