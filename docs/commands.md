# Command reference

Run `just --list` for live help. This page is checked by `just docs-check`; update it whenever recipes change.

## Setup and documentation

| Command | Purpose |
|---|---|
| `just help` | Show available recipes. |
| `just setup` | Create standard workspace folders. |
| `just docs-check` | Verify required docs exist and all recipes are documented here. |

## Hugging Face

| Command | Purpose |
|---|---|
| `just hf-search whisper 10` | Search Hub models. |
| `just hf-info openai/whisper-tiny` | Show model metadata. |
| `just hf-files openai/whisper-tiny` | List repo files and sizes without downloading. |
| `just hf-download openai/whisper-large-v3-turbo` | Download conversion-friendly files to `input/hf/`. |
| `just hf-download-minimal openai/whisper-large-v3-turbo` | Download only config/tokenizer/card style files. |
| `just hf-local` | List local HF downloads with manifests. |
| `just hf-auth` | Check auth status; use `HF_TOKEN` or `hf auth login` for gated models. |
| `just hf-smoke` | Network smoke test; does not download weights. |

## ONNX inspection and preparation

| Command | Purpose |
|---|---|
| `just inspect` | Inspect the dynamic ONNX model. |
| `just inspect-static` | Inspect the static ONNX model. |
| `just check` | Run ONNX checker on a model. |
| `just check-static` | Run ONNX checker on the static model. |
| `just static` | Convert configured dynamic input shape to fixed shape. |
| `just verify-static` | Fail if graph metadata still contains unknown/dynamic shapes. |
| `just prepare` | Run `setup`, `static`, `check-static`, and `verify-static`. |
| `just simplify` | Run `onnxsim` on the static model. |
| `just upload-list` | Print the Samsung SDK Service upload target. |
| `just hash input/whisper-large-v3-turbo/config.json` | Stream-hash one or more files. |
| `just clean-generated` | Delete generated static/simplified outputs. Use explicit paths for tests. |

## Tests

| Command | Purpose |
|---|---|
| `just test-current` | Non-destructive checks against current real model artifacts. |
| `just test-smoke` | Test conversion tasks on a tiny generated ONNX model. |
| `just test-all` | Run docs, current model checks, smoke tests, and HF smoke test. |

## Useful overrides

All paths/settings can be overridden with environment variables:

```bash
MODEL=output/onnx/my.onnx STATIC_MODEL=output/static/my_static.onnx just prepare
INPUT_NAME=input_features INPUT_SHAPE=1,128,3000 just static
HF_MODEL_DIR=D:/models/hf just hf-download openai/whisper-tiny
```
