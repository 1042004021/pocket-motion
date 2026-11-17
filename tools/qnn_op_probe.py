#!/usr/bin/env python3
"""Report the QNN toolchain status and the motion-module operator set.

The actual HTP context generation lives in ``qnn_context_probe.py``.  This
script only summarizes what is present, so it can be run cheaply after a
compile.  It never treats a CPU/GPU fallback as an NPU result.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


CONVERTER_NAMES = ("qnn-onnx-converter", "qnn_onnx_converter")
CONTEXT_NAMES = ("qnn-context-binary-generator", "qnn_context_binary_generator")


def find_tool(names: tuple[str, ...], sdk_root: Path | None) -> Path | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    if sdk_root:
        for host in ("x86_64-linux-clang", "aarch64-linux-clang", "x86_64-linux-gcc"):
            for name in names:
                candidate = sdk_root / "bin" / host / name
                if candidate.exists():
                    return candidate
    return None


def sdk_search(sdk_root: Path | None) -> dict[str, Any]:
    roots = []
    if sdk_root:
        roots.append(sdk_root)
    env = os.environ.get("QNN_SDK_ROOT")
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            Path.home() / "qairt",
            Path("/opt/qairt"),
            Path("/opt/qualcomm/qairt"),
            Path("/usr/local/qairt"),
        ]
    )
    seen = []
    for root in roots:
        if root in seen:
            continue
        seen.append(root)
        if root.exists():
            return {"found": True, "root": str(root)}
    return {"found": False, "searched": [str(p) for p in seen]}


def onnx_ops(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path))
    ops = Counter(node.op_type for node in model.graph.node)
    inputs = []
    for tensor in model.graph.input:
        dims = [d.dim_value for d in tensor.type.tensor_type.shape.dim]
        inputs.append({"name": tensor.name, "shape": dims})
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "op_count": len(model.graph.node),
        "ops": dict(sorted(ops.items())),
        "inputs": inputs,
    }


def host_qnn_ep_check() -> dict[str, Any]:
    result: dict[str, Any] = {"available": False}
    try:
        import onnxruntime as ort
        import onnxruntime_qnn as oq
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["available"] = True
    result["onnxruntime_version"] = ort.__version__
    result["onnxruntime_qnn_version"] = oq.__version__
    try:
        from onnxruntime_qnn import build_and_package_info

        result["qnn_version"] = build_and_package_info.qnn_version
    except Exception:
        pass

    provider_lib = oq.get_library_path()
    result["provider_lib"] = provider_lib
    try:
        ort.register_execution_provider_library(oq.EP_NAME, provider_lib)
    except Exception as exc:
        result["register_error"] = f"{type(exc).__name__}: {exc}"
        return result

    devices = []
    for device in ort.get_ep_devices():
        if device.ep_name != oq.EP_NAME:
            continue
        devices.append(
            {
                "ep_name": device.ep_name,
                "ep_vendor": device.ep_vendor,
                "device_type": str(device.device.type),
                "vendor_id": device.device.vendor_id,
                "device_id": device.device.device_id,
            }
        )
    result["devices"] = devices
    result["providers"] = ort.get_available_providers()
    return result


def read_context_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    modules = report.get("modules", [])
    return {
        "path": str(path),
        "module_count": len(modules),
        "all_pass": bool(modules)
        and all(m.get("allclose_2e-2") and m.get("finite") for m in modules),
        "modules": [
            {
                "onnx": m.get("onnx"),
                "input_shape": m.get("input_shape"),
                "context_bin_bytes": m.get("context_bin_bytes"),
                "compile_seconds": m.get("compile_seconds"),
                "max_abs_diff": m.get("max_abs_diff"),
                "mean_abs_diff": m.get("mean_abs_diff"),
                "correlation": m.get("correlation"),
                "allclose_2e-2": m.get("allclose_2e-2"),
                "finite": m.get("finite"),
            }
            for m in modules
        ],
        "host_compatibility_warning": report.get("host_compatibility_warning"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qnn-sdk-root", type=Path)
    parser.add_argument("--onnx-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument(
        "--context-report",
        type=Path,
        default=Path("models/qnn/qnn_context_report.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("models/qnn_op_report.json"))
    args = parser.parse_args()

    converter = find_tool(CONVERTER_NAMES, args.qnn_sdk_root)
    context_generator = find_tool(CONTEXT_NAMES, args.qnn_sdk_root)
    context = read_context_report(args.context_report)

    report: dict[str, Any] = {
        "status": "host_qnn_context_verified" if context and context["all_pass"] else "blocked",
        "phone_verified": False,
        "sdk": sdk_search(args.qnn_sdk_root),
        "converter": str(converter) if converter else None,
        "context_generator": str(context_generator) if context_generator else None,
        "host_qnn_ep": host_qnn_ep_check(),
        "models": [],
        "host_context_report": context,
        "blockers": [],
        "next_step": "Install QAIRT/QNN 2.39 and verify the generated context on the phone.",
    }

    for path in sorted(args.onnx_dir.glob("*.onnx")):
        report["models"].append(onnx_ops(path))

    if not context:
        report["blockers"].append(
            "no host QNN context report; run tools/qnn_context_probe.py first"
        )
    elif not context["all_pass"]:
        report["blockers"].append("host QNN context verification failed; inspect the report")

    report["blockers"].append(
        "host context was generated with QNN 2.52; the phone app ships QNN 2.39, "
        "so context binary compatibility is not proven"
    )
    report["blockers"].append(
        "the full SD1.5 UNetMotionModel has not yet been exported/compiled; "
        "only the motion modules were covered"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "host_qnn_context_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())