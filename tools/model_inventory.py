#!/usr/bin/env python3
"""Inventory local PocketMotion model assets without extracting or modifying them."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))

    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    dtypes: dict[str, int] = {}
    params = 0
    for info in tensors.values():
        dtypes[info["dtype"]] = dtypes.get(info["dtype"], 0) + 1
        shape = info.get("shape", [])
        n = 1
        for dim in shape:
            n *= int(dim)
        params += n

    prefixes = {
        "down_blocks": 0,
        "mid_block": 0,
        "up_blocks": 0,
    }
    for key in tensors:
        for prefix in prefixes:
            if key.startswith(prefix + "."):
                prefixes[prefix] += 1
                break

    return {
        "tensor_count": len(tensors),
        "dtypes": dtypes,
        "parameter_count": params,
        "key_prefixes": prefixes,
        "sample_keys": sorted(tensors)[:5],
    }


def zip_inventory(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
    total = sum(i.file_size for i in infos)
    return {
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "entry_count": len(infos),
        "uncompressed_bytes": total,
        "entries": [
            {
                "name": i.filename,
                "compressed_bytes": i.compress_size,
                "uncompressed_bytes": i.file_size,
            }
            for i in infos
            if not i.is_dir()
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/animatediff_lightning_4step_diffusers.safetensors"),
    )
    parser.add_argument("--sd15-dir", type=Path, default=Path("/home/elder/下载/sd15"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/inventory.json"),
    )
    args = parser.parse_args()

    result: dict[str, Any] = {
        "model": {
            "name": "AnimateDiff-Lightning 4-step",
            "repo": "ByteDance/AnimateDiff-Lightning",
            "file": str(args.checkpoint),
            "license": "creativeml-openrail-m",
            "source_url": "https://huggingface.co/ByteDance/AnimateDiff-Lightning",
        },
        "target": {
            "device": "Xiaomi 24129PN74C",
            "soc": "SM8750",
            "htp_arch": "V79",
            "android": "16",
        },
    }

    if args.checkpoint.exists():
        result["model"].update(safetensors_header(args.checkpoint))
        result["model"]["size_bytes"] = args.checkpoint.stat().st_size
        result["model"]["sha256"] = sha256(args.checkpoint)
    else:
        result["model"]["error"] = "checkpoint not found"

    sd15_zips = sorted(args.sd15_dir.glob("*.zip")) if args.sd15_dir.exists() else []
    result["sd15_qnn_packages"] = []
    for path in sd15_zips:
        try:
            item = {"file": str(path)}
            item.update(zip_inventory(path))
            result["sd15_qnn_packages"].append(item)
        except Exception as exc:  # keep inventory useful when one archive is bad
            result["sd15_qnn_packages"].append(
                {"file": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())