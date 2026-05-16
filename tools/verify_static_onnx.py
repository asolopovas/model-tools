#!/usr/bin/env python3
"""Fail if an ONNX graph still exposes symbolic/unknown shape metadata."""
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.model, load_external_data=False)
    bad: list[tuple[str, list[str | int]]] = []

    for item in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        ttype = item.type.tensor_type
        if not ttype.HasField("shape"):
            continue
        if any(not dim.HasField("dim_value") for dim in ttype.shape.dim):
            bad.append((item.name, dims(item)))

    if bad:
        print(f"ERROR: {args.model} still contains dynamic/unknown shape metadata:")
        for name, shape in bad[:100]:
            print(f"  {name}: {shape}")
        if len(bad) > 100:
            print(f"  ... {len(bad) - 100} more")
        return 1

    print(f"OK: {args.model} has static shape metadata for graph inputs/outputs/value_info.")
    for item in model.graph.input:
        print(f"  input {item.name}: {dims(item)}")
    for item in model.graph.output:
        print(f"  output {item.name}: {dims(item)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
