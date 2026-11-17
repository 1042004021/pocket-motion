#!/usr/bin/env bash
# Package the QAIRT 2.39 aarch64-android QNN net-runner and V79 contexts.
#
# The produced dist/qnn239-android/ directory is a standalone on-device test
# kit.  It is intentionally not committed: binaries, contexts and raw inputs
# stay out of Git.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SDK_ROOT=${QNN_SDK_ROOT:-/home/elder/下载/qairt/qairt/2.39.0.250925}
DIST="$PROJECT_ROOT/dist/qnn239-android"
MODELS="$PROJECT_ROOT/models/qnn239"

if [[ ! -x "$SDK_ROOT/bin/aarch64-android/qnn-net-run" ]]; then
  echo "qnn-net-run not found under $SDK_ROOT; set QNN_SDK_ROOT" >&2
  exit 2
fi

rm -rf "$DIST"
mkdir -p "$DIST/lib" "$DIST/skel" "$DIST/models"

cp "$SDK_ROOT/bin/aarch64-android/qnn-net-run" "$DIST/"
chmod +x "$DIST/qnn-net-run"

for lib in \
  libQnnHtp.so \
  libQnnSystem.so \
  libQnnHtpV79Stub.so \
  libQnnHtpNetRunExtensions.so; do
  cp "$SDK_ROOT/lib/aarch64-android/$lib" "$DIST/lib/"
done

for skel in \
  libQnnHtpV79Skel.so \
  libQnnHtpV79.so \
  libQnnSystem.so; do
  cp "$SDK_ROOT/lib/hexagon-v79/unsigned/$skel" "$DIST/skel/"
done

for module in \
  down_blocks_0_motion_modules_0_8f_32x32 \
  down_blocks_1_motion_modules_0_8f_16x16 \
  down_blocks_2_motion_modules_0_8f_8x8; do
  src="$MODELS/$module"
  if [[ ! -f "$src/context.bin.bin" ]]; then
    echo "missing context: $src/context.bin.bin" >&2
    exit 2
  fi
  mkdir -p "$DIST/models/$module"
  cp "$src/context.bin.bin" "$DIST/models/$module/context.bin"
  cp "$src/io/input_0.raw" "$DIST/models/$module/input_0.raw"
done

cp "$PROJECT_ROOT/tools/run_qnn_on_phone.sh" "$DIST/run_on_phone.sh"
chmod +x "$DIST/run_on_phone.sh"

du -sh "$DIST"
echo "prepared: $DIST"
echo "next: bash $DIST/run_on_phone.sh"