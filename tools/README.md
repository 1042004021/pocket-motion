# 工具说明

模型权重、ONNX、QNN 产物和 native 二进制不进入 Git。

## device_probe.py

检查本机工具链和目标 Android 设备，不修改设备、不下载模型。

```bash
python3 tools/device_probe.py
# 多设备时
python3 tools/device_probe.py --serial 16ffd10d
```

输出 JSON，包含：

- Python 版本和 `torch/numpy/onnx/onnxruntime/qai_appbuilder` 是否可用
- QNN SDK 环境变量、常见路径和转换器命令
- adb 设备、SoC、ABI、Android 版本、内存

当前已知缺失项会在 `warnings` 中列出，不把缺失环境当成通过。

## model_inventory.py

盘点本地 AnimateDiff-Lightning 权重、safetensors 张量、SD1.5 QNN zip 和哈希，不修改模型。

```bash
.venv/bin/python tools/model_inventory.py
```

输出 `models/inventory.json`。

## export_motion_onnx.py

把 AnimateDiff-Lightning motion module 导出为固定 8 帧 ONNX：

```bash
.venv/bin/python tools/export_motion_onnx.py
```

默认导出 3 个代表性模块：

- `down_blocks.0`：`[8, 320, 32, 32]`
- `down_blocks.1`：`[8, 640, 16, 16]`
- `down_blocks.2`：`[8, 1280, 8, 8]`

加 `--all` 可导出全部 motion module。导出后会做 ONNX checker 和 CPU 数值对比。

## qnn_context_probe.py

用 `onnxruntime-qnn` 的 QNN EP `enable_htp_prepare_only` 生成 HTP context，并在 host 侧回读校验。

```bash
.venv/bin/python tools/qnn_context_probe.py
```

当前目标配置：

```text
soc_model = 69    # SM8750
htp_arch  = 79    # V79
```

注意：

- 这是 host 侧编译/仿真验证，不是手机 NPU 实测。
- host 使用 QNN 2.52，手机现有运行时是 QNN 2.39；context 兼容性还没有证明。
- 完整 SD1.5 + AnimateDiff UNet 还没有导出。

## qnn_op_probe.py

汇总 QNN 环境、ONNX 算子和 `qnn_context_probe.py` 的 context 报告：

```bash
.venv/bin/python tools/qnn_op_probe.py
```

输出 `models/qnn_op_report.json`。`status=host_qnn_context_verified` 只表示 host 验证通过，`phone_verified` 仍为 `false`。

## 后续工具

等 QAIRT 2.39 和完整 UNet 准备好后再增加：

- `run_fixed_video.py`：256×256、8 帧、固定 shape 的完整最小推理入口

## 退出码

- `0`：host 验证完成
- `1`：host 数值校验失败
- `2`：缺少关键环境或上下文，任务不能继续