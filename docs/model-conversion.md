# Model conversion workflow

Current supported path:

```text
Hugging Face / safetensors -> ONNX FP16 -> static-shape ONNX -> compact FP32 LayerNorm -> Samsung SDK Service
```

Samsung's service-side quantizer profiles the converted graph on CPU. The current upload
candidate is still under service testing. It replaces expanded Whisper LayerNorm subgraphs
with compact FP32 ONNX `LayerNormalization` ops, then pre-simplifies the graph to reduce
EAIS old-SNC optimizer depth while avoiding FP16 CPU LayerNorm profiling. The simplifier
must skip `fuse_qkv`: Samsung Gen-6/EAIS 8.15.5.20 can mis-profile fused QKV attention
and fail at the first attention reshape with an invalid `[1, 1500, 20, 64]` shape.
The final prep step also folds Whisper's query-scale `Mul` into each `q_proj` weight and
bias, avoiding a standalone unsupported `Mul` immediately before the query reshape in EAIS.
`just verify-samsung` checks Samsung's documented SDK Service constraints before upload:
ONNX opset 13-17, static input/output shapes, file size below 2 GB, no Dropout/BatchNorm,
no QDQ nodes, and no remaining standalone Whisper self-attention query `Mul` nodes.

## Current Whisper target

Configured files:

```text
source config/weights: input/whisper-large-v3-turbo/
dynamic ONNX:          output/onnx/encoder_model_fp16.onnx
static ONNX:           output/static/encoder_model_fp16_static.onnx
Samsung upload ONNX:   output/simplified/encoder_model_fp16_static_compact_ln_sim.onnx
config:                configs/whisper-large-v3-turbo-samsung.json
```

Whisper large-v3 / large-v3-turbo encoder shape:

```text
input_features: [1, 128, 3000]
```

Prepare for Samsung SDK Service:

```bash
just prepare-samsung
```

Upload this file first:

```text
output/simplified/encoder_model_fp16_static_compact_ln_sim.onnx
```

## Validation policy

Before uploading a model, run:

```bash
just inspect
just check
just inspect-static
just check-static
just verify-static
```

`verify-static` is stricter than ONNX checker: it fails if exposed graph metadata still has symbolic or unknown dimensions.

## Samsung EAIS CLI workflow

If Samsung's `eais` CLI is installed in Ubuntu/WSL, generate a local CLI workspace from the
same upload candidate used for SDK Service:

```bash
just eais-check
just eais-workspace
just eais-command conversion
# optional when eais is installed: let Samsung generate/refresh native templates
just eais-init
```

The generated workspace defaults to `output/eais/whisper-large-v3-turbo` and hardlinks
`model.onnx` to avoid duplicating the 1.2 GB artifact when possible. The conservative
`safe` profile disables Samsung-side ONNX simplification because this repo already applies
ONNX simplification with Whisper-specific safeguards:

```bash
just eais-conversion
```

If conversion still fails during quantization/profiling, isolate conversion and compile with:

```bash
just eais-conversion output/eais/whisper-large-v3-turbo no-quant
```

## Official references used

- ONNX Runtime dynamic shape fixer: `python -m onnxruntime.tools.make_dynamic_shape_fixed` for mobile/NNAPI/CoreML-style fixed-shape needs.
- ONNX checker: `onnx.checker.check_model` for model validity checks.
