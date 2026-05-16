#!/usr/bin/env python3
"""Replace Whisper LayerNorm subgraphs with compact FP32 ONNX LayerNormalization ops.

This is intended for Samsung SDK Service conversion when the fully-expanded
Whisper encoder graph is too deep for EAIS old-SNC optimizer, but fused FP16
ChannelLayerNorm fails during CPU quantizer profiling.

For each exported LayerNorm subgraph:

    ReduceMean -> Sub -> Pow -> ReduceMean -> Add(eps) -> Sqrt -> Div -> Mul(gamma) -> Add(beta)

write:

    Cast(FP32) -> LayerNormalization(axis=-1, epsilon=eps) -> Cast(FP16)

The large Conv/MatMul weights remain FP16. Only LayerNorm scale/bias are promoted
to FP32 because the LayerNormalization input is FP32.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

LN_SUFFIXES = [
    "ReduceMean",
    "Sub",
    "Pow",
    "ReduceMean_1",
    "Add",
    "Sqrt",
    "Div",
    "Mul",
    "Add_1",
]


def set_value_info_dtype(model: onnx.ModelProto, name: str, elem_type: int) -> None:
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in collection:
            if value.name == name and value.type.HasField("tensor_type"):
                value.type.tensor_type.elem_type = elem_type


def initializer_by_name(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {tensor.name: tensor for tensor in model.graph.initializer}


def as_float_initializer(model: onnx.ModelProto, name: str) -> bool:
    for tensor in model.graph.initializer:
        if tensor.name != name:
            continue
        if tensor.data_type == TensorProto.FLOAT:
            return False
        if tensor.data_type != TensorProto.FLOAT16:
            raise ValueError(f"expected FLOAT16 initializer for {name}, got {tensor.data_type}")
        arr = numpy_helper.to_array(tensor).astype(np.float32)
        tensor.CopyFrom(numpy_helper.from_array(arr, name=name))
        return True
    raise KeyError(f"initializer not found: {name}")


def scalar_initializer_value(tensors: dict[str, onnx.TensorProto], name: str, default: float = 1e-5) -> float:
    tensor = tensors.get(name)
    if tensor is None:
        return default
    arr = numpy_helper.to_array(tensor)
    if arr.size != 1:
        return default
    return float(arr.reshape(-1)[0])


def axes_attr(node: onnx.NodeProto) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == "axes":
            return list(attr.ints)
    return None


def is_layernorm_start(
    node: onnx.NodeProto, node_by_name: dict[str, onnx.NodeProto]
) -> tuple[str, dict[str, onnx.NodeProto]] | None:
    if node.op_type != "ReduceMean" or not node.name.endswith("/ReduceMean"):
        return None
    prefix = node.name[: -len("/ReduceMean")]
    group: dict[str, onnx.NodeProto] = {}
    for suffix in LN_SUFFIXES:
        candidate = node_by_name.get(f"{prefix}/{suffix}")
        if candidate is None:
            return None
        group[suffix] = candidate
    if [group[s].op_type for s in LN_SUFFIXES] != [
        "ReduceMean",
        "Sub",
        "Pow",
        "ReduceMean",
        "Add",
        "Sqrt",
        "Div",
        "Mul",
        "Add",
    ]:
        return None
    if group["ReduceMean"].input[0] != group["Sub"].input[0]:
        return None
    if group["Sub"].output[0] != group["Pow"].input[0]:
        return None
    if group["Pow"].output[0] != group["ReduceMean_1"].input[0]:
        return None
    if group["ReduceMean_1"].output[0] != group["Add"].input[0]:
        return None
    if group["Add"].output[0] != group["Sqrt"].input[0]:
        return None
    if group["Sqrt"].output[0] != group["Div"].input[1]:
        return None
    if group["Div"].output[0] != group["Mul"].input[0]:
        return None
    if group["Mul"].output[0] != group["Add_1"].input[0]:
        return None
    axes = axes_attr(group["ReduceMean"])
    if axes not in (None, [-1]):
        return None
    return prefix, group


def ensure_opset(model: onnx.ModelProto, version: int) -> None:
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            if opset.version < version:
                opset.version = version
            return
    model.opset_import.append(helper.make_opsetid("", version))


def rewrite(model: onnx.ModelProto) -> tuple[int, int]:
    node_by_name = {node.name: node for node in model.graph.node}
    tensor_by_name = initializer_by_name(model)
    layernorms: dict[str, tuple[str, dict[str, onnx.NodeProto]]] = {}
    skip_names: set[str] = set()

    for node in model.graph.node:
        found = is_layernorm_start(node, node_by_name)
        if found is None:
            continue
        prefix, group = found
        start_name = group["ReduceMean"].name
        layernorms[start_name] = (prefix, group)
        skip_names.update(n.name for n in group.values())

    if not layernorms:
        return 0, 0

    converted_initializers = 0
    new_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in layernorms:
            prefix, group = layernorms[node.name]
            original_input = group["ReduceMean"].input[0]
            original_output = group["Add_1"].output[0]
            gamma = group["Mul"].input[1]
            beta = group["Add_1"].input[1]
            epsilon = scalar_initializer_value(tensor_by_name, group["Add"].input[1])

            for init_name in (gamma, beta):
                if as_float_initializer(model, init_name):
                    converted_initializers += 1

            safe = prefix.strip("/").replace("/", "_")
            fp32_input = f"{original_input}__{safe}_compact_ln_fp32_input"
            fp32_output = f"{original_output}__compact_ln_fp32_output"

            new_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[original_input],
                    outputs=[fp32_input],
                    name=f"{prefix}/CompactLayerNorm/CastToFloat",
                    to=TensorProto.FLOAT,
                )
            )
            new_nodes.append(
                helper.make_node(
                    "LayerNormalization",
                    inputs=[fp32_input, gamma, beta],
                    outputs=[fp32_output],
                    name=f"{prefix}/CompactLayerNorm",
                    axis=-1,
                    epsilon=epsilon,
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[fp32_output],
                    outputs=[original_output],
                    name=f"{prefix}/CompactLayerNorm/CastToHalf",
                    to=TensorProto.FLOAT16,
                )
            )
            set_value_info_dtype(model, fp32_input, TensorProto.FLOAT)
            set_value_info_dtype(model, fp32_output, TensorProto.FLOAT)
            set_value_info_dtype(model, original_output, TensorProto.FLOAT16)
            continue

        if node.name in skip_names:
            continue
        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    ensure_opset(model, 17)
    return len(layernorms), converted_initializers


def main() -> int:
    parser = argparse.ArgumentParser(description="Compact Whisper LayerNorm subgraphs to FP32 ONNX LayerNormalization.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true", help="Run ONNX checker after writing.")
    args = parser.parse_args()

    model = onnx.load(args.input, load_external_data=False)
    layernorms, initializers = rewrite(model)
    if layernorms == 0:
        raise SystemExit("No Whisper-style LayerNorm subgraphs found; refusing to write unchanged model.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)

    if args.check:
        checked = onnx.load(args.output, load_external_data=False)
        onnx.checker.check_model(checked)

    print(f"compacted layernorms: {layernorms}")
    print(f"converted fp16 initializers to fp32: {initializers}")
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
