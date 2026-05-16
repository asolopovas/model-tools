# Hugging Face workflow

Goal: search and download models without polluting the repo root.

## Search first

```bash
just search "whisper turbo" 10
just hf-info openai/whisper-large-v3-turbo
just files openai/whisper-large-v3-turbo
```

Quote queries with spaces. Increase the limit only when you need a broader scan:

```bash
just search "onnx speech recognition" 20
```

## Download safely

Start small with metadata only:

```bash
just download openai/whisper-large-v3-turbo minimal
```

Then download conversion-friendly files. Hugging Face Hub shows download progress bars, and the helper prints the selected preset, target folder, file count, local size, and manifest path.

```bash
just download openai/whisper-large-v3-turbo
```

Default download folder:

```text
input/hf/<repo-id-with-__-instead-of-slashes>/
```

Each download writes:

```text
MODEL_SOURCE.json
```

The default `conversion` preset includes common config/tokenizer files plus `*.safetensors` and `*.onnx`, while excluding legacy/checkpoint formats like `*.bin`, `*.pt`, `*.ckpt`, and `*.h5`.

## Auth

For gated/private models:

```bash
hf auth login
# or
export HF_TOKEN=hf_...
```

Then:

```bash
just hf-auth
```

## Cleanliness note

Official `huggingface_hub.snapshot_download(local_dir=...)` creates `.cache/huggingface` metadata inside the local folder. `tools/hf_models.py` removes that metadata by default after successful downloads and keeps only the model files plus `MODEL_SOURCE.json`.

## Official references used

- `huggingface_hub.HfApi.list_models` for search.
- `huggingface_hub.HfApi.model_info` for metadata/file listing.
- `huggingface_hub.snapshot_download` with `allow_patterns` and `ignore_patterns` for controlled downloads.
