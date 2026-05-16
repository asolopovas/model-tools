#!/usr/bin/env python3
"""Preflight checks for Samsung Exynos AI Studio / SDK Service ONNX uploads."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import onnx
from onnx import TensorProto

MAX_SDK_SERVICE_BYTES = 2 * 1024**3
SUPPORTED_OPSET_MIN = 13
SUPPORTED_OPSET_MAX = 17

# Subset from Samsung's published supported-operator table, plus only ops used by
# the current Whisper encoder candidate. Unknown ops fail so new exports do not
# silently move outside the documented SDK Service surface.
KNOWN_SUPPORTED_ONNX_OPS = {
    "Add",
    "Cast",
    "Conv",
    "Div",
    "Erf",
    "Gelu",
    "LayerNormalization",
    "MatMul",
    "Mul",
    "Reshape",
    "Softmax",
    "Transpose",
}

DISALLOWED_OPS = {
    "BatchNormalization": "SDK Service user guide says Batch Normalization layers are not supported.",
    "Dropout": "SDK Service user guide says Dropout layers are not supported.",
    "QuantizeLinear": "Samsung error docs say QDQ input models need converter handling, not direct upload.",
    "DequantizeLinear": "Samsung error docs say QDQ input models need converter handling, not direct upload.",
}


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    shape: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return shape


def is_static_shape(shape: Iterable[int | str | None]) -> bool:
    return all(isinstance(dim, int) and dim > 0 for dim in shape)


def elem_type_name(value_info: onnx.ValueInfoProto) -> str:
    elem_type = value_info.type.tensor_type.elem_type
    return TensorProto.DataType.Name(elem_type)


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"FAIL: {message}")


def warn(message: str) -> None:
    print(f"WARN: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--input-name", default=None)
    parser.add_argument("--input-shape", default=None, help="Comma-separated expected input shape, e.g. 1,128,3000")
    parser.add_argument("--max-bytes", type=int, default=MAX_SDK_SERVICE_BYTES)
    args = parser.parse_args()

    failures: list[str] = []
    model_path = args.model
    if not model_path.is_file():
        fail(f"model file not found: {model_path}", failures)
        return 1

    size = model_path.stat().st_size
    if size >= args.max_bytes:
        fail(f"model is {size} bytes, at/above SDK Service 2GB limit", failures)
    else:
        print(f"OK: model size {size} bytes ({size / 1024**3:.3f} GiB) is below 2GB")

    model = onnx.load(model_path, load_external_data=False)
    onnx.checker.check_model(model)
    print("OK: ONNX checker passed")

    ai_onnx_opsets = [opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")]
    if not ai_onnx_opsets:
        fail("missing ai.onnx opset import", failures)
    else:
        opset = max(ai_onnx_opsets)
        if not (SUPPORTED_OPSET_MIN <= opset <= SUPPORTED_OPSET_MAX):
            fail(f"ai.onnx opset {opset} outside documented Samsung range 13-17", failures)
        else:
            print(f"OK: ai.onnx opset {opset} is in documented Samsung range 13-17")

    expected_shape = None
    if args.input_shape:
        expected_shape = [int(part) for part in args.input_shape.split(",")]

    for value_info in list(model.graph.input) + list(model.graph.output):
        shape = tensor_shape(value_info)
        if not is_static_shape(shape):
            fail(f"{value_info.name} has non-static shape metadata: {shape}", failures)
        else:
            print(f"OK: {value_info.name} {elem_type_name(value_info)} shape {shape}")

    if args.input_name:
        inputs = {item.name: item for item in model.graph.input}
        if args.input_name not in inputs:
            fail(f"expected input {args.input_name!r} not found; inputs are {list(inputs)}", failures)
        elif expected_shape is not None:
            actual = tensor_shape(inputs[args.input_name])
            if actual != expected_shape:
                fail(f"input {args.input_name!r} shape {actual} != expected {expected_shape}", failures)
            else:
                print(f"OK: expected input {args.input_name!r} shape matches {expected_shape}")

    op_types = [node.op_type for node in model.graph.node]
    for op_type, reason in DISALLOWED_OPS.items():
        count = op_types.count(op_type)
        if count:
            fail(f"contains {count} {op_type} node(s): {reason}", failures)

    unknown_ops = sorted(set(op_types) - KNOWN_SUPPORTED_ONNX_OPS)
    if unknown_ops:
        fail(f"contains op(s) not in current Samsung preflight allowlist: {unknown_ops}", failures)
    else:
        print("OK: all op types are in current Samsung preflight allowlist")

    attention_query_muls = [node.name for node in model.graph.node if node.op_type == "Mul" and "/self_attn/Mul" in node.name]
    if attention_query_muls:
        fail(f"contains standalone Whisper self-attention query Mul nodes: {attention_query_muls[:5]}", failures)
    else:
        print("OK: no standalone Whisper self-attention query Mul nodes")

    for node in model.graph.node:
        if node.op_type == "Softmax":
            axis_attrs = [attr for attr in node.attribute if attr.name == "axis"]
            axis = axis_attrs[0].i if axis_attrs else -1
            if axis not in (-1, 1, 3):
                warn(f"Softmax {node.name!r} uses axis={axis}; Samsung docs say Softmax axis should map to C or W")

    if failures:
        print(f"Samsung ONNX preflight failed with {len(failures)} issue(s).")
        return 1
    print(f"Samsung ONNX preflight OK: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
