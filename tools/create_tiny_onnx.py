#!/usr/bin/env python3
"""Create a tiny dynamic ONNX model used by justfile smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])
    node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph([node], "tiny_dynamic_identity", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
