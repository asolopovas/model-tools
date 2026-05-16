# Cross-platform notes

Supported developer platforms:

```text
Windows 10/11, macOS, Linux
```

## Required tools

- `uv`
- `just`
- Python 3.11 or 3.12 available to `uv`
- Optional: Hugging Face CLI for `hf auth login`

## Shell policy

The `justfile` is intentionally shell-light:

- Unix uses `sh`.
- Windows uses Windows PowerShell via `set windows-shell`.
- Filesystem operations are delegated to `tools/workspace.py` instead of `mkdir`, `rm`, `test`, or shell-specific path logic.

This keeps recipes usable from PowerShell, cmd, Git Bash, macOS Terminal, and Linux shells.

## Path policy

Use forward slashes in project docs and configs:

```text
output/static/encoder_model_fp16_static.onnx
```

Python and `uv` handle these paths correctly on Windows/macOS/Linux.

## Large files

Large artifacts live under `input/` and `output/` and are ignored by git. Keep source code, configs, docs, and manifests small and reviewable.

## Official references used

- `just` supports `windows-shell` for platform-specific shell selection.
- `uv run` creates/updates the project `.venv` and runs commands in that environment.
