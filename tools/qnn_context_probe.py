#!/usr/bin/env python3
"""Compile fixed-shape ONNX graphs to QNN HTP context binaries and verify them.

This uses the QNN EP shipped by ``onnxruntime-qnn`` in "prepare only" mode.
That is a host-side compiler step: it does not claim that the binary is
loadable by the phone's QNN runtime, which still has to be checked separately.

The verification runs the generated EPContext graph through the host QNN
backend (HTP emulation) and compares it with the CPU ONNX Runtime result.  A
large mismatch is a hard failure and is reported as such.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as oq


def setup_qnn() -> tuple[list[Any], dict[str, str]]:
    provider_lib = oq.get_library_path()
    lib_dir = os.path.dirname(provider_lib)
    os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["ADSP_LIBRARY_PATH"] = lib_dir
    ort.register_execution_provider_library(oq.EP_NAME, provider_lib)
    devices = [d for d in ort.get_ep_devices() if d.ep_name == oq.EP_NAME]
    if not devices:
        raise RuntimeError("QNN execution provider did not register a device")
    options = {
        "backend_path": oq.get_qnn_htp_path(),
        "htp_arch": "79",
        "soc_model": "69",
    }
    return devices, options


def compile_and_check(
    onnx_path: Path,
    out_dir: Path,
    devices: list[Any],
    options: dict[str, str],
    seed: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    context_path = out_dir / "model.qnn"
    context_bin = out_dir / "model_qnn.bin"
    context_path.unlink(missing_ok=True)
    context_bin.unlink(missing_ok=True)

    start = time.time()
    so = ort.SessionOptions()
    so.log_severity_level = 2
    so.add_session_config_entry("ep.context_enable", "1")
    so.add_session_config_entry("ep.context_file_path", str(context_path))
    # Fail instead of silently putting unsupported nodes on CPU.
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_provider_for_devices(
        devices,
        {
            **options,
            "enable_htp_prepare_only": "1",
            "ep.context_enable": "1",
            "ep.context_file_path": str(context_path),
        },
    )
    ort.InferenceSession(str(onnx_path), so)
    compile_seconds = time.time() - start

    if not context_path.exists() or not context_bin.exists():
        raise RuntimeError(
            f"QNN did not produce both context files: {context_path}, {context_bin}"
        )

    # Load the generated EPContext and run it on the host QNN backend.
    load_so = ort.SessionOptions()
    load_so.log_severity_level = 2
    load_so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    load_so.add_provider_for_devices(devices, options)
    qnn_session = ort.InferenceSession(str(context_path), load_so)
    cpu_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    input_meta = cpu_session.get_inputs()[0]
    shape = [dim if isinstance(dim, int) else 1 for dim in input_meta.shape]
    rng = np.random.RandomState(seed)
    x = rng.randn(*shape).astype(np.float32)

    start = time.time()
    y_qnn = qnn_session.run(None, {input_meta.name: x})[0]
    run_seconds = time.time() - start
    y_cpu = cpu_session.run(None, {input_meta.name: x})[0]

    diff = y_qnn.astype(np.float64) - y_cpu.astype(np.float64)
    max_abs = float(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())
    corr = float(np.corrcoef(y_qnn.reshape(-1), y_cpu.reshape(-1))[0, 1])
    allclose = bool(np.allclose(y_qnn, y_cpu, atol=2e-2, rtol=2e-2))

    return {
        "onnx": str(onnx_path),
        "context": str(context_path),
        "context_bin": str(context_bin),
        "compile_seconds": round(compile_seconds, 3),
        "run_seconds": round(run_seconds, 3),
        "context_bytes": context_path.stat().st_size,
        "context_bin_bytes": context_bin.stat().st_size,
        "input_shape": shape,
        "providers": qnn_session.get_providers(),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "correlation": corr,
        "allclose_2e-2": allclose,
        "finite": bool(np.isfinite(y_qnn).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/qnn"))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    devices, options = setup_qnn()
    results = []
    for onnx_path in sorted(args.onnx_dir.glob("*.onnx")):
        print(f"compiling {onnx_path} ...", flush=True)
        result = compile_and_check(
            onnx_path,
            args.out_dir / onnx_path.stem,
            devices,
            options,
            args.seed,
        )
        results.append(result)
        print(
            f"  {result['compile_seconds']:.1f}s compile, "
            f"{result['context_bin_bytes'] / 1024 / 1024:.1f} MiB binary, "
            f"max_abs={result['max_abs_diff']:.4g}, allclose={result['allclose_2e-2']}",
            flush=True,
        )

    report = {
        "qnn_provider_version": oq.__version__,
        "target": {"soc_model": 69, "htp_arch": 79, "note": "SM8750 / V79"},
        "host_compatibility_warning": (
            "Generated with onnxruntime-qnn 2.8.0 (QNN 2.52). "
            "The phone app currently ships QNN 2.39; context binary "
            "compatibility with the phone runtime is not proven."
        ),
        "modules": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "qnn_context_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {report_path}")
    return 0 if all(r["allclose_2e-2"] and r["finite"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())