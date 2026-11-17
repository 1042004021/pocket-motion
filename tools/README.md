# 工具说明

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

## 后续工具（Task01）

计划增加：

- `model_inventory.py`：盘点 SD1.5 / AnimateDiff 权重、哈希和许可证
- `qnn_op_probe.py`：转换 ONNX 并汇总 QNN 算子覆盖
- `run_fixed_video.py`：256×256、8 帧、固定 shape 的最小推理入口

模型权重、ONNX、QNN 产物和 native 二进制不进入 Git。