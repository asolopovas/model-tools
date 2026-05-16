#!/usr/bin/env python3
"""Print ONNX model shape and operator summary."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import onnx


def dims(value_info: onnx.ValueInfoProto) -> list[str | int]:
    shape = value_info.type.tensor_type.shape
    out: list[str | int] = []
    for dim in shape.dim:
        if dim.HasField("dim_value"):
            out.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            out.append(dim.dim_param)
        else:
            out.append("?")
    return out


def has_dynamic_dim(value_info: onnx.ValueInfoProto) -> bool:
    ttype = value_info.type.tensor_type
    if not ttype.HasField("shape"):
        return False
    return any(not d.HasField("dim_value") for d in ttype.shape.dim)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--top-ops", type=int, default=25)
    args = parser.parse_args()

    model = onnx.load(args.model, load_external_data=False)

    print(f"model: {args.model}")
    print(f"ir_version: {model.ir_version}")
    print("opsets:", [(op.domain or "ai.onnx", op.version) for op in model.opset_import])
    print(f"nodes: {len(model.graph.node)}")
    print(f"initializers: {len(model.graph.initializer)}")

    print("\ninputs:")
    for item in model.graph.input:
        print(f"  {item.name}: {dims(item)}")

    print("\noutputs:")
    for item in model.graph.output:
        print(f"  {item.name}: {dims(item)}")

    all_infos = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    dynamic = [item.name for item in all_infos if has_dynamic_dim(item)]
    print(f"\ndynamic shape metadata tensors: {len(dynamic)}")
    for name in dynamic[:50]:
        print(f"  {name}")
    if len(dynamic) > 50:
        print(f"  ... {len(dynamic) - 50} more")

    print(f"\ntop {args.top_ops} op types:")
    for op, count in Counter(node.op_type for node in model.graph.node).most_common(args.top_ops):
        print(f"  {op}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
