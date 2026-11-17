# PocketMotion

独立的 NPU 视频生成可行性项目，目标是验证 **Snapdragon 8 Elite（SM8750，V79）+ 12GB RAM** 能否把小型视频生成模型完整跑在高通 NPU 上。

当前目标组合：

```text
SD1.5 + AnimateDiff-Lightning
256×256
8 帧
4～6 步
固定 shape
QNN HTP 全链路
```

## 原则

- **NPU 优先，不做静默 CPU/GPU 回退**
- 先验证算子覆盖，再写 Android UI
- 先跑 8 帧，再考虑 16 帧
- 先做 1～2 秒循环动画，不做长视频
- 任何关键算子无法落到 NPU，就记录阻塞点，不硬上
- host 侧 QNN 验证只作为预检，不能代替真机 NPU 结果

## 当前状态

Task01 进行中，已完成：

- AnimateDiff-Lightning 4-step motion module 下载、哈希和许可证盘点
- 3 个代表性 motion module 的固定 8 帧 ONNX 导出
- host QNN 2.52 / V79 context 生成与数值校验

仍阻塞：

- 本机没有 QAIRT 2.39；host 生成的 QNN 2.52 context 不能直接当作手机可加载
- 手机当前未连接，尚未做真机 8 帧推理
- 完整 SD1.5 + AnimateDiff UNet 尚未导出和编译

详细记录见 [Task01 验证记录](doc/验证/Task01-QNN算子覆盖与真机记录.md)。

模型权重、ONNX、QNN 转换产物和 native 二进制不进入 Git。

## 目录

```text
doc/
  ├── 项目说明.md
  ├── 需求/
  ├── 蒋磊/任务/
  ├── 蒋磊/进度/
  └── 验证/
tools/
  device_probe.py
  model_inventory.py
  export_motion_onnx.py
  qnn_context_probe.py
  qnn_op_probe.py
  README.md
```

## 快速检查

```bash
# 设备/工具链
python3 tools/device_probe.py

# 模型盘点
.venv/bin/python tools/model_inventory.py

# 导出固定 8 帧 ONNX
.venv/bin/python tools/export_motion_onnx.py

# host QNN context 生成与校验
.venv/bin/python tools/qnn_context_probe.py

# 汇总 QNN/算子报告
.venv/bin/python tools/qnn_op_probe.py
```

## 相关文档

- [项目说明](doc/项目说明.md)
- [PRD v0.1](doc/需求/PRD-v0.1-已确认.md)
- [Task01 - AnimateDiff-Lightning QNN 可行性验证](doc/蒋磊/任务/Task01-AnimateDiff-Lightning-QNN可行性验证.md)
- [Task01 验证记录](doc/验证/Task01-QNN算子覆盖与真机记录.md)
- [进度](doc/蒋磊/进度/进度.md)

## 许可证边界

本项目是独立 POC，不包含现有 Android 项目代码。
AnimateDiff-Lightning 使用 `creativeml-openrail-m`；SD1.5 各模型、QNN SDK 和 local-dream 的许可证需要分别核对，不能互相套用。
QNN SDK 为 Qualcomm 专有软件，模型权重、QNN 产物和 native 二进制不进入 Git。