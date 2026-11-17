# PocketMotion

独立的 NPU 视频生成可行性项目，目标是验证 **Snapdragon 8 Elite（SM8750）+ 12GB RAM** 能否把小型视频生成模型完整跑在高通 NPU 上。

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

## 当前状态

项目初始化完成，Task01 尚未开始。
模型权重、QNN 转换产物和 native 二进制不进入 Git。

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
  README.md
```

## 快速检查

```bash
python3 tools/device_probe.py
```

## 相关文档

- [项目说明](doc/项目说明.md)
- [PRD v0.1](doc/需求/PRD-v0.1-已确认.md)
- [Task01 - AnimateDiff-Lightning QNN 可行性验证](doc/蒋磊/任务/Task01-AnimateDiff-Lightning-QNN可行性验证.md)
- [进度](doc/蒋磊/进度/进度.md)

## 许可证边界

本项目是独立 POC，不包含 现有 Android 项目代码。
后续若使用 AnimateDiff、SD1.5 或其他模型，需要分别核对模型权重和代码的许可证；local-dream 的 CC BY-NC 4.0 边界不能自动套用到新模型。
