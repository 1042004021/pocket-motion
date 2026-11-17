#!/usr/bin/env python3
"""Export AnimateDiff-Lightning motion modules to fixed-shape ONNX graphs.

The exported graphs are the temporal modules that are inserted into the SD1.5
UNet.  Each graph has a fixed input shape:

    [num_frames, channels, latent_height, latent_width]

For the POC baseline (256x256, 8 frames, batch=1) the latent sizes are:

    320 channels -> 32x32
    640 channels -> 16x16
    1280 channels -> 8x8

Only representative modules are exported by default.  The same module
architecture repeats in the UNet, so exporting one module per
(channel, latent-size) pair is enough for QNN operator coverage.  Use
``--all`` to export every motion module in the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
import torch
from diffusers import MotionAdapter
from safetensors.torch import load_file


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class MotionModuleWrapper(torch.nn.Module):
    """Bind ``num_frames`` so the ONNX graph has no dynamic frame input."""

    def __init__(self, module: torch.nn.Module, num_frames: int) -> None:
        super().__init__()
        self.module = module
        self.num_frames = num_frames

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.module(latent, num_frames=self.num_frames)


def module_specs(adapter: MotionAdapter, all_modules: bool) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    def add(name: str, module: torch.nn.Module, latent_size: int) -> None:
        specs.append(
            {
                "name": name,
                "module": module,
                "channels": int(module.in_channels),
                "latent_size": latent_size,
            }
        )

    # 256x256 image -> 32x32 latent -> 16x16 -> 8x8.
    sizes = [32, 16, 8, 8]
    if all_modules:
        for block_idx, size in enumerate(sizes):
            for module_idx, module in enumerate(adapter.down_blocks[block_idx].motion_modules):
                add(f"down_blocks.{block_idx}.motion_modules.{module_idx}", module, size)
        for module_idx, module in enumerate(adapter.mid_block.motion_modules):
            add(f"mid_block.motion_modules.{module_idx}", module, 8)
        for block_idx, module_list in enumerate(adapter.up_blocks):
            size = [8, 8, 16, 32][block_idx]
            for module_idx, module in enumerate(module_list.motion_modules):
                add(f"up_blocks.{block_idx}.motion_modules.{module_idx}", module, size)
    else:
        # One representative module for each channel/latent-size pair.
        add("down_blocks.0.motion_modules.0", adapter.down_blocks[0].motion_modules[0], 32)
        add("down_blocks.1.motion_modules.0", adapter.down_blocks[1].motion_modules[0], 16)
        add("down_blocks.2.motion_modules.0", adapter.down_blocks[2].motion_modules[0], 8)

    return specs


def export_one(
    spec: dict[str, Any],
    out_dir: Path,
    frames: int,
    opset: int,
    check: bool,
) -> dict[str, Any]:
    channels = spec["channels"]
    latent_size = spec["latent_size"]
    name = spec["name"]
    safe_name = name.replace(".", "_")
    out_path = out_dir / f"{safe_name}_{frames}f_{latent_size}x{latent_size}.onnx"

    module = spec["module"].eval()
    wrapper = MotionModuleWrapper(module, frames).eval()
    dummy = torch.randn(frames, channels, latent_size, latent_size, dtype=torch.float32)

    with torch.no_grad():
        torch_output = wrapper(dummy)
        torch.onnx.export(
            wrapper,
            dummy,
            out_path,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["latent"],
            output_names=["out"],
            dynamo=False,
        )

    model = onnx.load(out_path)
    onnx.checker.check_model(model)
    ops = Counter(node.op_type for node in model.graph.node)

    result: dict[str, Any] = {
        "name": name,
        "path": str(out_path),
        "input_shape": [frames, channels, latent_size, latent_size],
        "size_bytes": out_path.stat().st_size,
        "sha256": file_sha256(out_path),
        "opset": opset,
        "ops": dict(sorted(ops.items())),
    }

    if check:
        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        ort_output = session.run(None, {"latent": dummy.numpy()})[0]
        diff = (torch_output.numpy() - ort_output).astype("float32")
        result["cpu_check"] = {
            "max_abs_diff": float(abs(diff).max()),
            "mean_abs_diff": float(abs(diff).mean()),
            "finite": bool(torch.isfinite(torch.from_numpy(ort_output)).all()),
        }

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/animatediff_lightning_4step_diffusers.safetensors"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--all", action="store_true", help="export every motion module")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.checkpoint} ...")
    adapter = MotionAdapter()
    adapter.load_state_dict(load_file(str(args.checkpoint)), strict=True)
    adapter.eval()

    specs = module_specs(adapter, args.all)
    print(f"exporting {len(specs)} module(s), frames={args.frames}, opset={args.opset}")
    results = []
    for spec in specs:
        print(f"  - {spec['name']} ({spec['channels']}ch, {spec['latent_size']}x{spec['latent_size']})")
        results.append(
            export_one(spec, args.out_dir, args.frames, args.opset, not args.skip_check)
        )

    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "frames": args.frames,
        "opset": args.opset,
        "image_size": 256,
        "modules": results,
    }
    manifest_path = args.out_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {manifest_path}")
    for result in results:
        print(
            f"{result['name']}: {result['size_bytes'] / 1024 / 1024:.1f} MiB, "
            f"ops={len(result['ops'])}, path={result['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())