# Model conversion workflow

Current supported path:

```text
Hugging Face / safetensors -> ONNX FP16 -> static-shape ONNX -> compact FP32 LayerNorm -> Samsung SDK Service
```

Samsung's service-side quantizer profiles the converted graph on CPU. The current upload
candidate is still under service testing. It replaces expanded Whisper LayerNorm subgraphs
with compact FP32 ONNX `LayerNormalization` ops, then pre-simplifies the graph to reduce
EAIS old-SNC optimizer depth while avoiding FP16 CPU LayerNorm profiling.

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

## Official references used

- ONNX Runtime dynamic shape fixer: `python -m onnxruntime.tools.make_dynamic_shape_fixed` for mobile/NNAPI/CoreML-style fixed-shape needs.
- ONNX checker: `onnx.checker.check_model` for model validity checks.
