# model-tools

Clean, cross-platform workspace for model search, download, inspection, and conversion prep.

Current primary workflow:

```text
Hugging Face / safetensors -> ONNX FP16 -> static-shape ONNX -> Samsung SDK Service
```

## Quick start

```bash
just setup
just test-current
just prepare
just upload-list
```

Current Samsung upload candidate:

```text
output/static/encoder_model_fp16_static.onnx
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

## Hugging Face examples

```bash
just hf-search whisper 10
just hf-files openai/whisper-large-v3-turbo
just hf-download-minimal openai/whisper-large-v3-turbo
just hf-download openai/whisper-large-v3-turbo
```

Downloads go to `input/hf/<repo>/` with a `MODEL_SOURCE.json` manifest.

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
