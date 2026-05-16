# Maintenance rules

Keep the repo boring, tidy, and repeatable.

## When changing commands

1. Update `justfile`.
2. Update `docs/commands.md`.
3. Run:

```bash
just docs-check
just test-smoke
```

`docs-check` fails when a recipe exists in `just --summary` but is missing from `docs/commands.md`.

## When changing workflows

Update all relevant docs in the same change:

- `README.md` for quick start changes.
- `docs/model-conversion.md` for conversion steps/paths/shapes.
- `docs/huggingface.md` for Hub search/download behavior.
- `docs/cross-platform.md` for platform/tooling assumptions.
- `AGENTS.md` for assistant/developer rules.

## When adding generated artifacts

Do not commit large model outputs. Put them under ignored folders:

```text
input/hf/
output/
```

If a tiny fixture is truly needed, document why and make sure `.gitattributes` marks it correctly.

## Before handing off

Run:

```bash
just test-all
```

If Hugging Face is offline or blocked, run at least:

```bash
just docs-check
just test-current
just test-smoke
```
