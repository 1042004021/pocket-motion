# 任务 - Task01

## 1. 基本信息

| 字段 | 值 |
|------|-----|
| 任务ID | Task01 |
| 关联需求 | PRD-v0.1 |
| 创建时间 | 2026-11-17 |
| 负责人 | 蒋磊 |
| 预计工时 | 待评估 |
| 优先级 | P0 |

## 2. 任务信息

### 2.1 任务名称

AnimateDiff-Lightning QNN 可行性验证

### 2.2 任务描述

验证 SD1.5 + AnimateDiff-Lightning 的 8 帧、256×256、4～6 步最小链路能否完整运行在高通 NPU 上。当前只做模型转换、QNN 算子覆盖和最小推理验证，不修改 现有 Android App，不做 UI。

明确边界：

- 不接受 CPU/GPU 作为核心推理兜底
- 不实现长视频、高分辨率或动态帧数
- 不下载/分发未确认许可证的权重

### 2.3 拆分方式

按验证阶段拆成：环境与模型盘点 → 模型转换 → 算子覆盖 → 固定 shape 真机推理 → 结论。

### 2.4 依赖关系

| 前置任务 | 依赖说明 |
|----------|----------|
| Task00 | 需要项目骨架、设备探测和验证记录结构 |

### 2.5 缺失资料

| 编号 | 缺失项 | 说明 | 影响范围 | 当前状态 |
|------|--------|------|----------|----------|
| M001 | AnimateDiff-Lightning 权重 | 已锁定 `animatediff_lightning_4step_diffusers.safetensors`，许可证 `creativeml-openrail-m` | 模型转换、算子覆盖 | 已解决 |
| M002 | SD1.5 QNN 模型包 | 本地已有 3 个 `qnn2.28_8gen2` 包；尚未验证其与本次 motion module context 的混合使用 | UNet/VAE 链路 | 部分具备 |
| M003 | QAIRT/QNN SDK | 已取得 QAIRT 2.39.0.250925，并用 2.39 重新生成 V79 context | 模型转换与真机运行 | 已解决 |
| M004 | 目标模型许可证 | AnimateDiff-Lightning 为 `creativeml-openrail-m`；SD1.5 各模型许可证需逐包核对 | 后续能否交付和分发 | 部分解决 |
| M005 | 完整 SD1.5 diffusers UNet | 目前只有预编译 QNN 包，没有可插入 motion module 的 UNet 源图 | 完整视频链路 | 缺失 |
| M006 | 真机连接 | 手机已连接（`16ffd10d`），3 个 2.39 context 已完成真机加载和执行 | 真机验收 | 已解决 |

## 3. 技术方案（如有）

### 3.1 实现思路

1. 用 `device_probe.py` 固定设备与环境基线。
2. 盘点 SD1.5 QNN 模型、AnimateDiff motion module 和 tokenizer/VAE 资源。
3. 将代表性 motion module 导出为 ONNX，固定 8 帧、256×256 对应的 32×32 / 16×16 / 8×8 latent shape。
4. 使用 `onnxruntime-qnn` 的 QNN HTP prepare-only 模式，先验证 `soc_model=69`、`htp_arch=79`（SM8750/V79）下的算子覆盖和 host 数值一致性。
5. 用 QAIRT 2.39 重新生成手机可加载的 context，在真机完成单模块 8 帧推理，记录耗时、峰值内存和温度。
6. 再导出并编译完整 SD1.5 + AnimateDiff UNet；完整链路不通过，不进入 Android UI。

关键原则：**算子不支持的必须记录，不静默回退 CPU/GPU；host 验证不等于真机验证。**

### 3.2 计划涉及文件清单

| 文件路径 | 操作类型 | 说明 |
|----------|----------|------|
| `tools/model_inventory.py` | 已新增 | 盘点模型文件、哈希和许可证 |
| `tools/export_motion_onnx.py` | 已新增 | 导出固定 8 帧 motion module ONNX |
| `tools/qnn_context_probe.py` | 已新增 | 生成 QNN HTP context 并做 host 数值校验 |
| `tools/qnn_op_probe.py` | 已新增 | 汇总 QNN 环境、ONNX 算子和 context 报告 |
| `doc/验证/Task01-QNN算子覆盖与真机记录.md` | 已新增 | 记录环境、命令、结果和阻塞点 |
| `tools/run_fixed_video.py` | 未新增 | 完整 UNet 导出后再做固定 8 帧最小视频入口 |
| `tools/build_qnn239_contexts.sh` | 已新增 | QAIRT 2.39 V79 context 构建入口 |
| `tools/patch_motion_groupnorm.py` | 已新增 | V79 rank-5 GroupNorm 等价改写 |
| `tools/prepare_android_qnn_kit.sh` | 已新增 | 打包 Android adb 测试包 |
| `tools/run_qnn_on_phone.sh` | 已新增 | 真机推送、执行和日志回拉 |
| `models/` | 本地目录 | 模型权重、ONNX、QNN 产物不入 Git |

## 4. 验收标准

- [x] 输出模型与许可证盘点表
- [x] 成功导出固定 8 帧、256×256 的 motion module ONNX 图
- [x] 输出 QNN 算子覆盖报告，明确不支持项（QNN 2.52 预检 + QNN 2.39 V79 context 复验；未发现不支持项）
- [x] 真机完成 motion module 8 帧输出，核心计算走 QNN HTP，未配置 CPU/GPU 回退
- [x] 记录 3 个模块的真机耗时、峰值 RSS 和温度变化（单次稳态延迟仍待补测）
- [ ] 给出完整链路的“继续/换模型/停止”结论（motion module 已通过，完整 UNet 未验证）

## 5. 接口设计（如有）

不适用，本任务没有 App 或服务端接口。

## 6. 涉及技能（如有）

| 技能名称 | 用途说明 |
|----------|----------|
| funi-dev-process | 任务、验证记录和实现索引管理 |

## 7. 交付物

- 模型/许可证盘点结果
- QNN 算子覆盖报告
- 8 帧固定 shape 最小推理结果
- 可行性结论和后续建议

## 8. 测试要点（供测试参考）

- 8 帧和 16 帧分别测试，先只要求 8 帧
- 256×256 和 384×384 对比内存/耗时
- 断网运行，避免云端依赖
- 连续运行 3 次，记录热降频
- 任何 CPU/GPU 算子回退都要显式打印

## 9. 实现索引（任务完成时必填）

### 9.1 最终实现方案

当前为**阶段性方案**，不是最终通过方案；2.39 转换链路和 Android 测试包已补齐：

1. 从 `ByteDance/AnimateDiff-Lightning` 下载 4-step motion module，锁定 SHA-256 和许可证。
2. 用 `diffusers.MotionAdapter` 加载权重，导出 3 个代表性 `AnimateDiffTransformer3D` 为固定 shape ONNX：
   - 320 通道 / 32×32
   - 640 通道 / 16×16
   - 1280 通道 / 8×8
3. 用 `onnxruntime-qnn` 的 QNN EP `enable_htp_prepare_only` 生成 `soc_model=69`、`htp_arch=79` 的 HTP context，做初步算子覆盖。
4. 用 QAIRT 2.39 的官方转换器和 `qnn-context-binary-generator` 生成手机同版本 V79 context；通过 `patch_motion_groupnorm.py` 把 V79 不接受的 rank-5 GroupNorm 改写为等价 4D 原语。
5. 3 个模块的 2.39 context 都在 host `qnn-net-run` 上执行，并与 CPU ONNX 对齐；Android adb 测试包已生成。
6. 手机已连接，3 个 context 均成功 `createFromBinary` 并执行 8 帧；数值、耗时、峰值 RSS 和温度已记录。
7. 完整 SD1.5 UNet 仍未导出。

### 9.2 实际代码改动

| 类型 | 文件路径 | 关键位置 | 改动说明 |
|------|----------|----------|----------|
| 新增 | `tools/model_inventory.py` | `safetensors_header`、`zip_inventory` | 盘点模型、哈希、张量和 SD1.5 QNN 包 |
| 新增 | `tools/export_motion_onnx.py` | `MotionModuleWrapper`、`export_one` | 导出固定 8 帧 ONNX，并做 CPU 数值校验 |
| 新增 | `tools/qnn_context_probe.py` | `setup_qnn`、`compile_and_check` | 2.52 初步 HTP context 生成与回读 |
| 新增 | `tools/patch_motion_groupnorm.py` | `patch` | 把 V79 不支持的 rank-5 GroupNorm 改写为等价原语 |
| 新增 | `tools/build_qnn239_contexts.sh` | 主流程 | 用 QAIRT 2.39 生成 3 个 V79 context |
| 新增 | `tools/qairt/Dockerfile` | 环境 | Ubuntu 22.04 + QAIRT 2.39 转换依赖 |
| 新增 | `tools/prepare_android_qnn_kit.sh` | 打包 | 生成 adb 真机测试包 |
| 新增 | `tools/run_qnn_on_phone.sh` | 真机脚本 | 推送 context/input、执行 qnn-net-run |
| 新增 | `tools/compare_qnn_outputs.py` | `compare` | 真机输出与 CPU ONNX 的 NHWC/NCHW 对比 |
| 重写 | `tools/qnn_op_probe.py` | `host_qnn_ep_check`、`read_context_report` | 汇总 QNN 环境、ONNX 算子和 context 报告 |
| 新增 | `doc/验证/Task01-QNN算子覆盖与真机记录.md` | 全文 | 记录结果、风险和下一步 |
| 更新 | `tools/README.md`、`README.md`、`doc/项目说明.md`、`doc/蒋磊/进度/进度.md` | 状态/工具说明 | 同步当前进展 |

### 9.3 接口、数据与外部依赖影响

| 类型 | 名称 | 影响说明 |
|------|------|----------|
| 接口 | 无 | 不涉及服务端接口 |
| 数据 | 无 | 不涉及数据库 |
| 外部依赖 | `onnxruntime-qnn==2.8.0`（QNN 2.52） | 仅用于 host 侧初步算子/context 验证 |
| 外部依赖 | QAIRT 2.39.0.250925 | 已取得；用于生成手机同版本 V79 context |
| 外部依赖 | SD1.5 diffusers UNet 权重 | **仍需准备**；用于导出完整 UNet + motion module |
| 外部依赖 | 目标手机 | 已连接（`16ffd10d`），motion module 真机验证通过 |

### 9.4 验证记录与已知风险

| 项目 | 结果 | 说明 |
|------|------|------|
| ONNX checker | 通过 | 3 个固定 shape 图 |
| CPU ONNX 与 PyTorch 对比 | 通过 | 最大绝对误差 ≤ 3.81e-06 |
| QNN 2.52 context 生成 | 通过 | 3 个模块均生成 EPContext，无 CPU 子图 |
| QNN 2.39 V79 context 生成 | 通过 | 3 个模块均生成官方 context，大小约 15.1/21.7/70.8 MiB |
| QNN host 数值校验 | 通过 | 最大绝对误差 ≤ 7.43e-03，相关系数 ≥ 0.9999995 |
| Android 测试包 | 已使用 | `dist/qnn239-android/`，已在手机执行并回拉输出 |
| 真机 NPU 推理 | 通过 | 3 个 2.39 context 均在 SM8750 手机上 `createFromBinary` 并执行 8 帧 |
| 真机数值一致性 | 通过 | 最大绝对误差 ≤ 6.62e-03，相关系数 ≥ 0.9999995 |
| 真机耗时/峰值 RSS/温度 | 已记录 | 3 次连续推理 wall time 约 1021.7/585.7/516.2 ms；VmHWM 31.3/32.8/83.3 MB；温度见验证记录 |
| 完整 SD1.5 + AnimateDiff | 未执行 | 只有 motion module，没有完整 UNetMotionModel |
| 已知风险 1 | 低 | 手机端已加载成功；单次稳态延迟仍待 profiling 补测 |
| 已知风险 2 | 高 | 完整 UNet 的算子覆盖、内存和耗时未知 |
| 已知风险 3 | 中 | host 的 QNN 设备类型显示为 CPU，host 耗时不能代表手机 NPU |

## 10. 修改记录

| 时间 | 修改内容 | 修改原因 | 确认人 |
|------|----------|----------|--------|
| 2026-11-17 | 创建 AnimateDiff QNN 可行性任务 | 用户确认 NPU-only 路线 | 用户 |
| 2026-11-17 | 完成 motion module ONNX 导出、QNN 2.52 预检、QAIRT 2.39 V79 context 和 Android adb 测试包 | 用户要求持续开发，先做算子覆盖和真机测试准备 | 用户 |
| 2026-11-18 | 手机连接后完成 3 个 context 的 `createFromBinary`、8 帧执行、数值对比、耗时/峰值 RSS/温度记录 | 手机已连接，完成真机验收 | 用户 |