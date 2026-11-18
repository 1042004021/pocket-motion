#!/usr/bin/env python3
"""Compare on-device QNN outputs with the CPU ONNX reference.

The QNN context uses NHWC tensors while the exported ONNX model uses NCHW.
This script transposes both sides before comparing, then prints the same
metrics used in the Task01 verification record.

Usage:
    .venv/bin/python tools/compare_qnn_outputs.py
    .venv/bin/python tools/compare_qnn_outputs.py --kit dist/qnn239-android
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KIT = PROJECT_ROOT / "dist" / "qnn239-android"

MODULES = [
    ("down_blocks_0_motion_modules_0_8f_32x32", 320, 32),
    ("down_blocks_1_motion_modules_0_8f_16x16", 640, 16),
    ("down_blocks_2_motion_modules_0_8f_8x8", 1280, 8),
]


def compare(kit: Path) -> tuple[list[dict], bool]:
    results = []
    ok = True
    for name, channels, size in MODULES:
        input_path = kit / "models" / name / "input_0.raw"
        output_path = kit / "results" / name / "Result_0" / "out.raw"
        onnx_path = PROJECT_ROOT / "models" / "onnx" / f"{name}.onnx"

        if not input_path.exists() or not output_path.exists():
            raise FileNotFoundError(f"missing input/output for {name}")
        if not onnx_path.exists():
            raise FileNotFoundError(f"missing ONNX reference: {onnx_path}")

        # NHWC on device -> NCHW for the exported PyTorch model.
        x = np.fromfile(input_path, dtype=np.float32).reshape(8, size, size, channels)
        x = x.transpose(0, 3, 1, 2)
        y_qnn = np.fromfile(output_path, dtype=np.float32).reshape(8, size, size, channels)
        y_qnn = y_qnn.transpose(0, 3, 1, 2)

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        y_cpu = session.run(None, {"latent": x})[0]

        diff = y_qnn - y_cpu
        allclose = bool(np.allclose(y_qnn, y_cpu, atol=2e-2, rtol=2e-2))
        ok = ok and allclose
        results.append(
            {
                "module": name,
                "channels": channels,
                "latent": f"{size}x{size}",
                "max_abs_diff": float(np.abs(diff).max()),
                "mean_abs_diff": float(np.abs(diff).mean()),
                "corr": float(np.corrcoef(y_qnn.reshape(-1), y_cpu.reshape(-1))[0, 1]),
                "allclose": allclose,
            }
        )
    return results, ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    results, ok = compare(args.kit)
    for row in results:
        print(
            f"{row['module']}: max_abs={row['max_abs_diff']:.6g} "
            f"mean_abs={row['mean_abs_diff']:.6g} "
            f"corr={row['corr']:.10f} allclose={row['allclose']}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())