# AGENTS.md

Guidance for coding assistants and maintainers working in this repo.

## Project purpose

This is a clean local model-conversion workspace. Keep it focused on:

```text
search/download -> inspect -> convert/prep -> validate -> vendor upload
```

Current main target: Whisper large-v3-turbo encoder ONNX prepared for Samsung SDK Service.

## Rules

- Keep the repo root clean. Put model inputs under `input/` and generated artifacts under `output/`.
- Do not commit large model files. Respect `.gitignore` and `.gitattributes`.
- Prefer small Python utilities in `tools/` over shell-specific logic.
- Keep `justfile` cross-platform: no `rm`, `mkdir -p`, `test`, bash-only syntax, or PowerShell-only syntax in recipes.
- Use forward-slash paths in docs/configs.
- Update docs in the same change as code or task changes.

## Required checks after edits

Run the narrowest relevant checks, and before handoff prefer:

```bash
just docs-check
just test-smoke
```

For full validation:

```bash
just test-all
```

If a new `just` recipe is added, update `docs/commands.md`; `just docs-check` enforces this.

## Official docs consulted for current tooling

- `just`: `windows-shell` and shell configuration.
- `uv`: `uv run` project environment behavior and pyproject dependency management.
- Hugging Face Hub: `HfApi`, `model_info`, `snapshot_download`, `allow_patterns`, `ignore_patterns`, and `local_dir` metadata behavior.
- ONNX Runtime: `make_dynamic_shape_fixed` for mobile/fixed-shape conversion prep.
- ONNX: `onnx.checker.check_model` validation.
