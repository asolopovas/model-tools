# Command reference

Run `just --list` for live help. This page is checked by `just docs-check`; update it whenever recipes change.

## Setup and documentation

| Command | Purpose |
|---|---|
| `just help` | Show available recipes. |
| `just install` | Install default Python dependencies and create standard workspace folders. |
| `just install-export` | Install optional PyTorch/Optimum export dependencies too; this is large. |
| `just setup` | Create standard workspace folders. |
| `just docs-check` | Verify required docs exist and all recipes are documented here. |

## Hugging Face

Short commands for everyday model discovery:

| Command | Purpose |
|---|---|
| `just search "whisper turbo" 10` | Search Hub models by query. |
| `just files openai/whisper-tiny` | List repo files and sizes without downloading. |
| `just download openai/whisper-tiny` | Download conversion-friendly files to `input/hf/` with Hub progress bars. |
| `just local` | List local HF downloads with manifests. |
| `just auth` | Check auth status; use `HF_TOKEN` or `hf auth login` for gated models. |

Full Hugging Face recipes:

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
| `just verify-samsung` | Run Samsung SDK Service documented preflight checks on an upload candidate. |
| `just prepare` | Run `setup`, `static`, `check-static`, and `verify-static`. |
| `just prepare-samsung` | Build and validate the Samsung candidate and refresh the single final upload folder. |
| `just final-samsung` | Create/refresh `output/final/` with exactly one ONNX upload file. |
| `just simplify` | Run `onnxsim` on the static model. |
| `just upload-list` | Print the Samsung SDK Service upload target from `output/final/`. |
| `just hash input/whisper-large-v3-turbo/config.json` | Stream-hash one or more files. |
| `just clean-generated` | Delete generated static/simplified outputs. Use explicit paths for tests. |

## Samsung EAIS CLI

These recipes wrap Samsung Exynos AI Studio's `eais` CLI when it is installed in the current shell/WSL environment.

| Command | Purpose |
|---|---|
| `just eais-check` | Check whether `eais` is on `PATH` and print package/help information. |
| `just eais-workspace` | Create/update `output/eais/whisper-large-v3-turbo` with the prepared ONNX and EAIS config files. |
| `just eais-command conversion` | Print the recommended `eais conversion` command without running it. |
| `just eais-init` | Run `eais init` in the local EAIS workspace to let Samsung create native templates. |
| `just eais-generation` | Run `eais generation` in the local EAIS workspace. |
| `just eais-conversion` | Run `eais conversion` with the conservative `safe` profile (`onnx_simplify=false`, `profile_batchsize=1`). |
| `just eais-conversion output/eais/whisper-large-v3-turbo no-quant` | Run conversion with quantization disabled for isolation. |
| `just eais-compile` | Run `eais compile` in the local EAIS workspace. |
| `just eais-profiling` | Run `eais profiling` in the local EAIS workspace. |

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
EAIS_WORKSPACE=output/eais/test EAIS_DEVICE=Gen-8 just eais-workspace
FINAL_DIR=output/final FINAL_MODEL_NAME=SAMSUNG_UPLOAD_WHISPER_ENCODER.onnx just final-samsung
```
