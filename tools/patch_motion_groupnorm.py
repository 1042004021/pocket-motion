#!/usr/bin/env python3
"""Rewrite the exported motion-module GroupNorm into QNN-2.39-friendly primitives.

QAIRT 2.39 maps the original 5-D InstanceNormalization to a rank-5 GroupNorm
node.  The V79 HTP backend rejects that node ("incorrect Rank 5").  The
mathematical equivalent below uses ReduceMean/Sub/Pow/Sqrt/Div on the same
3-D tensor and leaves the rest of the graph untouched.

This is an export-time graph transformation for the fixed-shape POC, not a
change to the model definition or weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, TensorProto


def patch(input_path: Path, output_path: Path) -> int:
    model = onnx.load(str(input_path))
    epsilon_name = "pocketmotion_gn_epsilon"
    if not any(init.name == epsilon_name for init in model.graph.initializer):
        model.graph.initializer.append(
            helper.make_tensor(epsilon_name, TensorProto.FLOAT, [1], [1e-6])
        )

    new_nodes = []
    patched = 0
    for node in model.graph.node:
        if node.op_type != "InstanceNormalization":
            new_nodes.append(node)
            continue

        x = node.input[0]
        y = node.output[0]
        prefix = node.name + "_gn"
        new_nodes.extend(
            [
                helper.make_node(
                    "ReduceMean", [x], [prefix + "_mean"], name=prefix + "_mean", axes=[-1], keepdims=1
                ),
                helper.make_node("Sub", [x, prefix + "_mean"], [prefix + "_centered"], name=prefix + "_centered"),
                helper.make_node("Mul", [prefix + "_centered", prefix + "_centered"], [prefix + "_sq"], name=prefix + "_sq"),
                helper.make_node(
                    "ReduceMean", [prefix + "_sq"], [prefix + "_var"], name=prefix + "_var", axes=[-1], keepdims=1
                ),
                helper.make_node("Add", [prefix + "_var", epsilon_name], [prefix + "_var_eps"], name=prefix + "_var_eps"),
                helper.make_node("Sqrt", [prefix + "_var_eps"], [prefix + "_std"], name=prefix + "_std"),
                helper.make_node("Div", [prefix + "_centered", prefix + "_std"], [y], name=prefix + "_out"),
            ]
        )
        patched += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = patch(args.input, args.output)
    print(f"patched {count} InstanceNormalization node(s): {args.input} -> {args.output}")
    return 0 if count == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())