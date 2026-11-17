#!/usr/bin/env bash
# Build QAIRT 2.39 V79 QNN contexts for the three representative motion modules.
#
# Prerequisites:
#   - QAIRT 2.39 SDK at QNN_SDK_ROOT (default shown below)
#   - Docker image: pocket-motion-qairt:2.39
#     docker build -t pocket-motion-qairt:2.39 -f tools/qairt/Dockerfile tools/qairt
#   - Python venv .venv-qairt with onnx==1.14.1 (used for the graph patch)
#
# Generated files stay in models/ and are ignored by Git.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SDK_ROOT=${QNN_SDK_ROOT:-/home/elder/下载/qairt/qairt/2.39.0.250925}
IMAGE=${QAIRT_DOCKER_IMAGE:-pocket-motion-qairt:2.39}
PY310="$PROJECT_ROOT/.venv-qairt/bin/python"

if [[ ! -x "$SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter" ]]; then
  echo "QAIRT 2.39 not found at $SDK_ROOT; set QNN_SDK_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PY310" ]]; then
  echo "missing $PY310; create the Python 3.10 conversion venv first" >&2
  exit 2
fi

# module name, latent dimensions, channel count
MODULES=(
  "down_blocks_0_motion_modules_0_8f_32x32 8,320,32,32 320"
  "down_blocks_1_motion_modules_0_8f_16x16 8,640,16,16 640"
  "down_blocks_2_motion_modules_0_8f_8x8 8,1280,8,8 1280"
)

for spec in "${MODULES[@]}"; do
  read -r name dim channels <<<"$spec"
  onnx="$PROJECT_ROOT/models/onnx/${name}.onnx"
  patched="$PROJECT_ROOT/models/onnx/${name}_qnn239.onnx"
  out="$PROJECT_ROOT/models/qnn239/$name"

  if [[ ! -f "$onnx" ]]; then
    echo "missing ONNX: $onnx; run tools/export_motion_onnx.py first" >&2
    exit 2
  fi

  echo "=== $name"
  "$PY310" "$PROJECT_ROOT/tools/patch_motion_groupnorm.py" \
    --input "$onnx" --output "$patched"
  mkdir -p "$out"

  docker run --rm \
    -v "$SDK_ROOT:/qairt:ro" \
    -v "$PROJECT_ROOT:/work" \
    -w /work \
    "$IMAGE" \
    bash -lc "
      set -euo pipefail
      export PYTHONPATH=/qairt/lib/python
      export LD_LIBRARY_PATH=/qairt/lib/x86_64-linux-clang:\${LD_LIBRARY_PATH:-}
      rm -f models/qnn239/$name/model_qnn.cpp models/qnn239/$name/model_qnn.bin
      /qairt/bin/x86_64-linux-clang/qnn-onnx-converter \
        --input_network models/onnx/${name}_qnn239.onnx \
        --input_dim latent $dim \
        --input_layout latent NCHW \
        --float_bitwidth 16 \
        --output_path models/qnn239/$name/model_qnn.cpp
      rm -rf models/qnn239/$name/lib
      mkdir -p models/qnn239/$name/lib
      /opt/qvenv/bin/python /qairt/bin/x86_64-linux-clang/qnn-model-lib-generator \
        -c models/qnn239/$name/model_qnn.cpp \
        -b models/qnn239/$name/model_qnn.bin \
        -t x86_64-linux-clang \
        -l motion$channels \
        -o models/qnn239/$name/lib
      rm -f models/qnn239/$name/context.bin.bin
      /qairt/bin/x86_64-linux-clang/qnn-context-binary-generator \
        --model models/qnn239/$name/lib/x86_64-linux-clang/libmotion$channels.so \
        --backend /qairt/lib/x86_64-linux-clang/libQnnHtp.so \
        --binary_file context.bin \
        --output_dir models/qnn239/$name \
        --soc_model 69 \
        --log_level warn
    "

  ls -lh "$out/context.bin.bin"
done

echo "all QNN 2.39 V79 contexts built under $PROJECT_ROOT/models/qnn239"