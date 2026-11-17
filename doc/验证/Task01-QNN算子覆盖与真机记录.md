# Task01 QNN 算子覆盖与真机记录

> 记录时间：2026-11-17
> 结论状态：**host 侧算子验证通过，真机 NPU 验证未完成**

## 1. 结论摘要

| 项目 | 结果 |
|------|------|
| AnimateDiff-Lightning motion module 固定 shape ONNX | 已导出 3 个代表性模块，覆盖 320/640/1280 通道 |
| QNN 算子覆盖 | 2.52 与 2.39 均能在 host 编译；2.39 V79 context 已生成 |
| QNN 数值一致性 | 3 个模块 QNN 输出与 CPU ONNX 输出 `allclose(atol=2e-2, rtol=2e-2)` |
| 真机 NPU 运行 | **未完成**：手机当前未连接；Android 测试包已准备好 |
| 手机 QNN 运行时兼容性 | **待真机验证**：已用 QNN 2.39 重新生成 context，与手机库版本一致 |
| 完整 SD1.5 + AnimateDiff UNet | **未导出**：目前只验证了 motion module，不是整条视频链路 |

**当前判断**：motion module 的算子层面没有发现阻塞，QNN 2.39 V79 context 也已在 host 加载验证；下一步只需手机端确认 context 可加载，然后进入完整 UNet。还不能宣称“SD1.5 + AnimateDiff-Lightning 已在 NPU 上跑通”。

## 2. 环境

| 项目 | 值 |
|------|-----|
| 主机 | Linux x86_64 |
| Python | 3.14.7 |
| PyTorch | 2.15.0+cpu |
| diffusers | 0.43.0 |
| onnx | 1.23.2 |
| onnxruntime | 1.32.0 |
| onnxruntime-qnn | 2.8.0 |
| onnxruntime-qnn 内置 QNN 版本 | 2.52.0 |
| host HTP backend 接口 | 2.41.0（加载日志） |
| QAIRT 2.39 | `2.39.0.250925`（Ubuntu 22.04 Docker 转换） |
| 目标设备 | Xiaomi 24129PN74C |
| 目标 SoC | SM8750 |
| 目标 HTP 架构 | V79 |
| 目标 Android | 16 / SDK 36 |

设备探测结果沿用 Task00，不在本次重复执行。

## 3. 模型资产

### 3.1 AnimateDiff-Lightning

| 字段 | 值 |
|------|-----|
| 仓库 | `ByteDance/AnimateDiff-Lightning` |
| 文件 | `animatediff_lightning_4step_diffusers.safetensors` |
| 大小 | 907,702,248 bytes（约 866 MiB） |
| SHA-256 | `8f3330914edd8fc2ee5659e6944323858feb0ec21cc913789202d043f69b5e17` |
| 张量数 | 567，全部 F16 |
| 参数量（按 safetensors header 统计） | 453,813,440 |
| 模块分布 | down_blocks 216 / mid_block 27 / up_blocks 324 |
| 许可证 | `creativeml-openrail-m` |
| 来源 | `https://huggingface.co/ByteDance/AnimateDiff-Lightning` |

本次使用 `hf-mirror.com` 镜像下载。模型文件位于 `models/`，已被 `.gitignore` 排除。

### 3.2 本地 SD1.5 QNN 包

本地 `/home/elder/下载/sd15/` 有 3 个 `qnn2.28_8gen2` 包，可作为后续整条 UNet/VAE 链路的候选，但本次没有加载：

| 文件 | 大小 | SHA-256 |
|------|------|---------|
| `ChilloutMix_qnn2.28_8gen2.zip` | 1,069,856,038 | `104b02923baed3d3395c9b5d5ce18912c20884bb7b404b550733c0241c6eba3e` |
| `DarkSushiV4_qnn2.28_8gen2.zip` | 1,033,065,066 | `b891f9c3f2c6350bac05696ecc10c7fa386e0e898b2af974cd210ca151edb920` |
| `MeinaMixV12_qnn2.28_8gen2.zip` | 1,031,765,133 | `90463ac0db6eb7424bb6c7a048c811bde2996b2321eaf58b64d02d626c333098` |

这些包是预编译 QNN context，不是可编辑的 UNet 源模型；不能直接把 motion module 插进去。

## 4. ONNX 导出

脚本：`tools/export_motion_onnx.py`

```bash
.venv/bin/python tools/export_motion_onnx.py
```

固定输入：

```text
[8, C, H, W]
```

256×256 图像对应 latent 32×32；下采样后 16×16、8×8。

| 模块 | 输入 shape | ONNX 大小 | 节点数 |
|------|------------|-----------|--------|
| `down_blocks.0.motion_modules.0` | `[8, 320, 32, 32]` | 8.7 MiB | 139 |
| `down_blocks.1.motion_modules.0` | `[8, 640, 16, 16]` | 34.5 MiB | 139 |
| `down_blocks.2.motion_modules.0` | `[8, 1280, 8, 8]` | 137.7 MiB | 139 |

ONNX checker 通过；3 个图在 CPU ONNX Runtime 下与 PyTorch 输出最大绝对误差分别为 `3.22e-06`、`3.81e-06`、`3.64e-06`。

## 5. ONNX 算子集合

3 个代表性图的算子集合一致：

```text
Add, Cast, Constant, Div, Erf, Gather, Identity,
InstanceNormalization, LayerNormalization, MatMul, Mul,
Reshape, Shape, Slice, Softmax, Sqrt, Transpose, Unsqueeze
```

重点关注项：

- `Erf`：GEGLU 中的精确 GELU，QNN 本次成功编译。
- `InstanceNormalization`：GroupNorm 导出后的形式，QNN 本次成功编译。
- `Softmax` / `MatMul` / `Transpose` / `Reshape`：时序 attention 的核心，本次成功编译。
- `Shape` / `Slice` / `Gather`：固定 shape 下被常量折叠，未形成动态控制流。

## 6. QNN HTP 编译与 host 校验

脚本：`tools/qnn_context_probe.py`

```bash
.venv/bin/python tools/qnn_context_probe.py
```

关键配置：

```text
soc_model = 69      # SM8750
htp_arch  = 79      # V79
enable_htp_prepare_only = 1
ep.context_enable = 1
```

结果：

| 模块 | 输入 shape | context bin | 编译耗时 | max_abs_diff | mean_abs_diff | 相关系数 | allclose |
|------|------------|-------------|----------|--------------|---------------|----------|----------|
| 320 | `[8, 320, 32, 32]` | 17.6 MiB | 110.9 s | 5.09e-03 | 4.49e-04 | 0.99999983 | 是 |
| 640 | `[8, 640, 16, 16]` | 22.5 MiB | 88.9 s | 6.44e-03 | 7.18e-04 | 0.99999966 | 是 |
| 1280 | `[8, 1280, 8, 8]` | 72.2 MiB | 10.9 s | 7.29e-03 | 9.66e-04 | 0.99999956 | 是 |

生成的 EPContext 模型只有一个 `EPContext` 节点，输入输出与原图一致；编译和回读都启用了 `session.disable_cpu_ep_fallback=1`，没有把节点静默放到 CPU。host 侧 QNN 输出与 CPU ONNX 输出一致，误差量级符合 FP16 HTP 推理。

注意：host 的 `ort.get_ep_devices()` 把 QNN 设备报成 CPU hardware type，这只是 x86 host emulation 的显示；host 运行耗时不能当作手机 NPU 性能。

### 6.1 QAIRT 2.39 复验

后续拿到了官方 QAIRT 2.39.0.250925，并用 Ubuntu 22.04 Docker 运行官方转换器，重新生成了与手机运行时同版本的 context。

这一步发现并处理了两个 QNN 2.39 的限制：

1. **Arch host 上直接运行转换器不可靠**：系统 `libc++`/`libunwind` ABI 不匹配时，Shape 推理会得到错误结果；改用 Ubuntu 22.04 容器后转换正常。
2. **V79 HTP 不接受 rank-5 GroupNorm**：原始 ONNX 的 5D `InstanceNormalization` 会被转换器映射成 rank-5 `GroupNorm`，backend 报 `incorrect Rank 5`。`tools/patch_motion_groupnorm.py` 将它改写为等价的 `ReduceMean/Sub/Mul/ReduceMean/Add/Sqrt/Div`，不改变模型语义。

QNN 2.39 V79 context 结果（host `qnn-net-run`，输入/输出按 QNN 的 NHWC 布局转换后与 CPU ONNX 对比）：

| 模块 | context 大小 | max_abs_diff | mean_abs_diff | 相关系数 | allclose |
|------|--------------|--------------|---------------|----------|----------|
| 320 / 32×32 | 15.1 MiB | 5.20e-03 | 4.48e-04 | 0.99999983 | 是 |
| 640 / 16×16 | 21.7 MiB | 7.43e-03 | 7.19e-04 | 0.99999966 | 是 |
| 1280 / 8×8 | 70.8 MiB | 6.50e-03 | 9.79e-04 | 0.99999955 | 是 |

这证明 QNN 2.39 的 V79 context 可以在同版本 host HTP 后端上加载并执行；**仍然不是手机实测**。

### 6.2 Android 测试包

已生成：

```text
dist/qnn239-android/
├── qnn-net-run                 # QAIRT 2.39 aarch64-android
├── lib/                        # libQnnHtp.so、V79 stub、QnnSystem
├── skel/                       # V79 skel / DSP 库
├── models/                     # 3 个 context + NHWC input raw
└── run_on_phone.sh             # adb 推送 + 执行
```

明天手机连上后：

```bash
bash dist/qnn239-android/run_on_phone.sh
```

脚本会推送 kit 到 `/data/local/tmp/pocket-motion-qnn`，逐个加载 context、执行 8 帧输入，并把输出拉回 `dist/qnn239-android/results/`。

## 7. 剩余风险与阻塞

### 7.1 QNN 版本：已对齐

| 位置 | 版本 |
|------|------|
| 生成 context 的 QAIRT | 2.39.0.250925 |
| host `qnn-net-run` / HTP backend | 2.39.0.250925 |
| 手机现有 QNN 库 | `AISW_VERSION 2.39.0` / `v2.39.0.250925215840_163802` |

2.39 context 已在 host HTP 后端加载并执行通过；剩余风险是**手机端 `createFromBinary` 是否接受该 context**，必须真机确认，不能仅凭 host 推断。

### 7.2 只有 motion module，没有完整 UNet

本次验证的是 UNet 内部的时间模块，不是完整 SD1.5 + AnimateDiff 图。完整链路还需要：

1. SD1.5 diffusers UNet 权重；
2. `UNetMotionModel` 或等价的 UNet + motion module 合并图；
3. 固定 8 帧、256×256、4～6 步的完整去噪循环；
4. 文本编码器、VAE、scheduler 的 NPU/宿主边界定义；
5. 与现有 QNN 后端的输入输出接口对齐。

现有 SD1.5 QNN 包是整包预编译 context，无法插入 motion module；必须重新导出并编译完整 UNet。

### 7.3 真机状态

本轮没有连接手机，也没有运行真机推理；因此没有耗时、峰值内存、温度数据。Android 测试包已准备好，等手机连接后执行。

## 8. 下一步

1. 连接手机并运行 `dist/qnn239-android/run_on_phone.sh`，确认 3 个 context 都能加载并执行。
2. 记录手机端初始化、单次推理、峰值内存和温度。
3. 下载 SD1.5 diffusers 权重，导出完整 `UNetMotionModel`，重复同样的 QNN 编译和真机验证。
4. 只有完整 UNet 的算子覆盖、真机耗时和内存都可接受，才进入 Android UI。

## 9. 证据文件

本地产物（均不入 Git）：

```text
models/onnx/*.onnx
models/onnx/*_qnn239.onnx
models/onnx/export_manifest.json
models/qnn/*/model.qnn                  # 2.52 preliminary
models/qnn/*/model_qnn.bin
models/qnn/qnn_context_report.json
models/qnn239/*/model_qnn.cpp           # 2.39 converted model
models/qnn239/*/model_qnn.bin
models/qnn239/*/context.bin.bin         # 2.39 V79 context
models/qnn239/*/io/out/Result_0/out.raw # host QNN output
models/qnn_op_report.json
models/inventory.json
dist/qnn239-android/                     # Android adb test kit
```

复现命令：

```bash
# 2.52 preliminary / op coverage
.venv/bin/python tools/model_inventory.py
.venv/bin/python tools/export_motion_onnx.py
.venv/bin/python tools/qnn_context_probe.py
.venv/bin/python tools/qnn_op_probe.py

# 2.39 V79 contexts
docker build -t pocket-motion-qairt:2.39 -f tools/qairt/Dockerfile tools/qairt
bash tools/build_qnn239_contexts.sh

# Android test kit
bash tools/prepare_android_qnn_kit.sh
bash dist/qnn239-android/run_on_phone.sh
```