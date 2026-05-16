# Documentation index

Compact docs for this model conversion workspace.

- [Command reference](commands.md) — every `just` task and what it does.
- [Model conversion workflow](model-conversion.md) — current ONNX/static/Samsung path.
- [Hugging Face workflow](huggingface.md) — search, inspect, and clean downloads.
- [Cross-platform notes](cross-platform.md) — Windows/macOS/Linux expectations.
- [Maintenance rules](maintenance.md) — how to keep docs and code in sync.

VS Code tasks mirror the common `just` workflows and are defined in `.vscode/tasks.json`.

Run this after changing tasks or docs:

```bash
just docs-check
```
