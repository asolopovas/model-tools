# model-tools

Clean, cross-platform workspace for model search, download, inspection, and conversion prep.

Current primary workflow:

```text
Hugging Face / safetensors -> ONNX FP16 -> static-shape ONNX -> compact FP32 LayerNorm -> Samsung SDK Service
```

## Quick start

```bash
just install
just test-current
just prepare-samsung
```

Final Samsung upload folder:

```text
output/final/
```

Upload only this ONNX file from that folder:

```text
output/final/SAMSUNG_UPLOAD_WHISPER_ENCODER.onnx
```

This candidate is simplified with ONNX Simplifier's `fuse_qkv` optimization disabled and
Whisper query-scale `Mul` folded into `q_proj` weights for Samsung EAIS compatibility.

`just prepare-samsung` refreshes this final folder and removes any older `.onnx` files from it.

For local Samsung Exynos AI Studio CLI testing, create a reproducible `eais` workspace:

```bash
just eais-workspace
just eais-command conversion
# when Samsung's eais CLI is installed in the current shell/WSL:
just eais-conversion
```

Current Whisper encoder shape:

```text
input_features: [1, 128, 3000]
```

## Layout

```text
input/          source models and clean HF downloads
output/         generated ONNX/static/simplified/vendor artifacts
configs/        conversion target configs
tools/          small Python utilities
docs/           compact project docs
.vscode/        VS Code tasks
```

Large artifacts are ignored by git.

## Hugging Face CLI examples

Use the short `just` commands for the common find -> preview -> download loop:

```bash
# Find models by text query; quote queries with spaces.
just search "whisper turbo" 10
just search "onnx speech recognition" 20

# Preview repository files and sizes before downloading weights.
just files openai/whisper-large-v3-turbo

# Start small: download configs, tokenizer, model card, and other metadata only.
just download openai/whisper-large-v3-turbo minimal

# Download conversion-friendly files with Hub progress bars.
# Default preset is "conversion": configs/tokenizers plus *.safetensors and *.onnx,
# while skipping legacy/checkpoint formats such as *.bin, *.pt, *.ckpt, and *.h5.
just download openai/whisper-large-v3-turbo

# See local downloads and their pinned source revisions.
just local
```

Downloads go to `input/hf/<repo>/` with a `MODEL_SOURCE.json` manifest. For gated/private
models, run `hf auth login` or set `HF_TOKEN`, then check with `just auth`.

## Documentation

Start here: [`docs/README.md`](docs/README.md)

Important docs:

- [`docs/commands.md`](docs/commands.md)
- [`docs/model-conversion.md`](docs/model-conversion.md)
- [`docs/huggingface.md`](docs/huggingface.md)
- [`docs/cross-platform.md`](docs/cross-platform.md)
- [`docs/maintenance.md`](docs/maintenance.md)

VS Code users can run the same workflows from **Terminal > Run Task...**; tasks are defined in `.vscode/tasks.json`.

Keep docs in sync:

```bash
just docs-check
```
