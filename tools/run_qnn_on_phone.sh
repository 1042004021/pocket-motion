#!/usr/bin/env bash
# Run the packaged QNN 2.39 V79 motion-module contexts on a connected phone.
#
# Usage:
#   bash run_on_phone.sh [remote_dir]
#
# Requires adb and an arm64-v8a Android device.  The contexts and binaries are
# local test artifacts; do not commit them.
set -euo pipefail

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REMOTE_DIR=${1:-/data/local/tmp/pocket-motion-qnn}
MODULES=(
  down_blocks_0_motion_modules_0_8f_32x32
  down_blocks_1_motion_modules_0_8f_16x16
  down_blocks_2_motion_modules_0_8f_8x8
)

if ! adb get-state >/dev/null 2>&1; then
  echo "no adb device; connect the phone and enable USB debugging" >&2
  exit 2
fi

echo "pushing kit to $REMOTE_DIR ..."
adb shell "rm -rf '$REMOTE_DIR'"
adb push "$LOCAL_DIR/." "$REMOTE_DIR/" >/dev/null
adb shell "chmod 755 '$REMOTE_DIR/qnn-net-run'"

mkdir -p "$LOCAL_DIR/results"

for module in "${MODULES[@]}"; do
  input="$REMOTE_DIR/models/$module/input_0.raw"
  context="$REMOTE_DIR/models/$module/context.bin"
  output="$REMOTE_DIR/models/$module/out"
  echo "=== $module ==="
  adb shell "cd '$REMOTE_DIR' && \
    echo '$input' > models/$module/input_list.txt && \
    rm -rf '$output' && \
    LD_LIBRARY_PATH='$REMOTE_DIR/lib' \
    ADSP_LIBRARY_PATH='$REMOTE_DIR/skel' \
    ./qnn-net-run \
      --retrieve_context '$context' \
      --backend '$REMOTE_DIR/lib/libQnnHtp.so' \
      --input_list '$REMOTE_DIR/models/$module/input_list.txt' \
      --output_dir '$output' \
      --log_level info"
  adb pull "$output" "$LOCAL_DIR/results/$module" >/dev/null
  echo "result: $LOCAL_DIR/results/$module"
done

echo "done; compare out.raw against models/onnx CPU output if needed"