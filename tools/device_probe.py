#!/usr/bin/env python3
"""Inspect the host toolchain and the target Android device for the NPU POC.

This script intentionally uses only the Python standard library. It does not
download models, install packages, or modify the device.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


QNN_ENV_KEYS = (
    "QNN_SDK_ROOT",
    "QAIRT_ROOT",
    "QNN_HOME",
    "SNPE_ROOT",
)

PYTHON_MODULES = (
    "torch",
    "numpy",
    "onnx",
    "onnxruntime",
    "qai_appbuilder",
)


def run(command: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (completed.stdout or completed.stderr).strip()
        return completed.returncode, output
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def adb_devices() -> list[dict[str, str]]:
    code, output = run(["adb", "devices"])
    if code != 0:
        return []
    devices = []
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            devices.append({"serial": parts[0], "state": parts[1]})
    return devices


def shell(serial: str, command: str, timeout: int = 20) -> str:
    code, output = run(["adb", "-s", serial, "shell", command], timeout=timeout)
    return output if code == 0 else ""


def getprop(serial: str, name: str) -> str:
    return shell(serial, f"getprop {name}")


def read_memory_mb(serial: str) -> int | None:
    raw = shell(serial, "grep MemTotal /proc/meminfo")
    for token in raw.split():
        if token.isdigit():
            return round(int(token) / 1024)
    return None


def module_status() -> dict[str, bool]:
    return {
        name: importlib.util.find_spec(name) is not None
        for name in PYTHON_MODULES
    }


def qnn_environment() -> dict[str, object]:
    env = {name: os.environ.get(name, "") for name in QNN_ENV_KEYS}
    candidates = [
        env.get("QNN_SDK_ROOT", ""),
        env.get("QAIRT_ROOT", ""),
        env.get("QNN_HOME", ""),
        env.get("SNPE_ROOT", ""),
        "/opt/qualcomm/qairt",
        "/opt/qairt",
        str(Path.home() / "qairt"),
    ]
    existing = sorted({
        str(Path(value).expanduser())
        for value in candidates
        if value and Path(value).expanduser().exists()
    })
    return {
        "env": env,
        "existing_paths": existing,
        "converter": shutil.which("qnn-onnx-converter") or "",
        "qnn_context_binary_generator": shutil.which("qnn-context-binary-generator") or "",
        "qnn_net_run": shutil.which("qnn-net-run") or "",
    }


def choose_serial(requested: str | None, devices: list[dict[str, str]]) -> str | None:
    if requested:
        return requested
    online = [item["serial"] for item in devices if item["state"] == "device"]
    return online[0] if online else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe host and Android device for NPU video work")
    parser.add_argument("--serial", help="adb serial; defaults to the first online device")
    args = parser.parse_args()

    devices = adb_devices()
    serial = choose_serial(args.serial, devices)
    report: dict[str, object] = {
        "host": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "android_home": os.environ.get("ANDROID_HOME", os.environ.get("ANDROID_SDK_ROOT", "")),
            "adb": shutil.which("adb") or "",
            "python_modules": module_status(),
            "qnn": qnn_environment(),
        },
        "adb_devices": devices,
        "device": None,
        "warnings": [],
    }

    if not serial:
        report["warnings"].append("没有在线 Android 设备，跳过设备探测")
    else:
        abi = getprop(serial, "ro.product.cpu.abi")
        memory_mb = read_memory_mb(serial)
        device = {
            "serial": serial,
            "manufacturer": getprop(serial, "ro.product.manufacturer"),
            "model": getprop(serial, "ro.product.model"),
            "android_release": getprop(serial, "ro.build.version.release"),
            "sdk": getprop(serial, "ro.build.version.sdk"),
            "board_platform": getprop(serial, "ro.board.platform"),
            "soc_model": getprop(serial, "ro.soc.model"),
            "abi": abi,
            "memory_mb": memory_mb,
        }
        report["device"] = device
        if "arm64" not in abi:
            report["warnings"].append(f"目标设备 ABI 不是 arm64: {abi or 'unknown'}")
        if memory_mb is not None and memory_mb < 8000:
            report["warnings"].append(f"设备内存偏低: {memory_mb} MB")
        if not device["soc_model"] and not device["board_platform"]:
            report["warnings"].append("未能读取 SoC 信息，无法确认 NPU 代际")

    missing_modules = [name for name, present in module_status().items() if not present]
    if missing_modules:
        report["warnings"].append("缺少 Python 模块: " + ", ".join(missing_modules))
    qnn = qnn_environment()
    if not qnn["existing_paths"] and not qnn["converter"]:
        report["warnings"].append("未找到 QNN SDK/转换器路径，需要先准备 QAIRT 环境")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())