#!/usr/bin/env python3
"""Fold Whisper self-attention query scaling into q_proj weights.

Samsung EAIS/ENNQuantizer may fail when profiling Whisper attention if the query
scaling Mul remains as a standalone op before the query Reshape. This pass folds
that scalar Mul into each q_proj MatMul weight and bias, then removes the Mul node.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper


def replace_initializer(graph: onnx.GraphProto, name: str, array: np.ndarray) -> None:
    for index, initializer in enumerate(graph.initializer):
        if initializer.name == name:
            graph.initializer[index].CopyFrom(numpy_helper.from_array(array, name=name))
            return
    raise KeyError(f"initializer not found: {name}")


def remove_value_info(graph: onnx.GraphProto, names: set[str]) -> None:
    kept = [item for item in graph.value_info if item.name not in names]
    del graph.value_info[:]
    graph.value_info.extend(kept)


def consumers_by_input(nodes: Iterable[onnx.NodeProto]) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    return consumers


def producer_by_output(nodes: Iterable[onnx.NodeProto]) -> dict[str, onnx.NodeProto]:
    return {output_name: node for node in nodes for output_name in node.output}


def initializer_map(graph: onnx.GraphProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in graph.initializer}


def array_dtype_for_tensor(tensor: onnx.TensorProto) -> np.dtype:
    if tensor.data_type == TensorProto.FLOAT16:
        return np.dtype(np.float16)
    if tensor.data_type == TensorProto.FLOAT:
        return np.dtype(np.float32)
    raise TypeError(f"unsupported q_proj initializer dtype for {tensor.name}: {tensor.data_type}")


def get_scalar_initializer(init: dict[str, onnx.TensorProto], name: str) -> float | None:
    tensor = init.get(name)
    if tensor is None:
        return None
    array = numpy_helper.to_array(tensor)
    if array.size != 1:
        return None
    return float(array.reshape(()))


def fold_query_scales(model: ModelProto) -> int:
    graph = model.graph
    nodes = list(graph.node)
    init = initializer_map(graph)
    producer = producer_by_output(nodes)
    consumers = consumers_by_input(nodes)
    nodes_to_remove: set[str] = set()
    removed_outputs: set[str] = set()
    folded = 0

    for node in nodes:
        if node.op_type != "Mul" or "/self_attn/Mul" not in node.name:
            continue
        if len(node.input) != 2 or len(node.output) != 1:
            continue

        lhs, rhs = node.input
        scale = get_scalar_initializer(init, rhs)
        data_input = lhs
        if scale is None:
            scale = get_scalar_initializer(init, lhs)
            data_input = rhs
        if scale is None:
            continue

        add = producer.get(data_input)
        if add is None or add.op_type != "Add" or "/self_attn/q_proj/Add" not in add.name:
            continue

        bias_inputs = [name for name in add.input if name in init]
        matmul_outputs = [name for name in add.input if name not in init]
        if len(bias_inputs) != 1 or len(matmul_outputs) != 1:
            continue

        matmul = producer.get(matmul_outputs[0])
        if matmul is None or matmul.op_type != "MatMul" or "/self_attn/q_proj/MatMul" not in matmul.name:
            continue

        weight_inputs = [name for name in matmul.input if name in init]
        if len(weight_inputs) != 1:
            continue

        weight_name = weight_inputs[0]
        bias_name = bias_inputs[0]
        weight_tensor = init[weight_name]
        bias_tensor = init[bias_name]
        weight_dtype = array_dtype_for_tensor(weight_tensor)
        bias_dtype = array_dtype_for_tensor(bias_tensor)

        weight = numpy_helper.to_array(weight_tensor).astype(np.float32) * scale
        bias = numpy_helper.to_array(bias_tensor).astype(np.float32) * scale
        replace_initializer(graph, weight_name, weight.astype(weight_dtype))
        replace_initializer(graph, bias_name, bias.astype(bias_dtype))

        mul_output = node.output[0]
        for consumer in consumers.get(mul_output, []):
            for idx, input_name in enumerate(consumer.input):
                if input_name == mul_output:
                    consumer.input[idx] = data_input

        nodes_to_remove.add(node.name or mul_output)
        removed_outputs.add(mul_output)
        folded += 1

    if folded:
        kept_nodes = [node for node in graph.node if (node.name or (node.output[0] if node.output else "")) not in nodes_to_remove]
        del graph.node[:]
        graph.node.extend(kept_nodes)
        remove_value_info(graph, removed_outputs)

    return folded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument("--check", action="store_true", help="Run ONNX checker after writing.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = onnx.load(args.input_model, load_external_data=False)
    folded = fold_query_scales(model)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    if args.check:
        onnx.checker.check_model(model)
    print(f"folded query scale Mul nodes: {folded}")
    print(f"wrote: {args.output_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
