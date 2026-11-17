# Task01 QNN 算子覆盖与真机记录

> 记录时间：2026-11-17
> 结论状态：**host 侧算子验证通过，真机 NPU 验证未完成**

## 1. 结论摘要

| 项目 | 结果 |
|------|------|
| AnimateDiff-Lightning motion module 固定 shape ONNX | 已导出 3 个代表性模块，覆盖 320/640/1280 通道 |
| QNN 算子覆盖 | 代表性图在 host QNN HTP 上完整编译为单个 EPContext，未出现 CPU 子图 |
| QNN 数值一致性 | 3 个模块 QNN 输出与 CPU ONNX 输出 `allclose(atol=2e-2, rtol=2e-2)` |
| 真机 NPU 运行 | **未完成**：本机无 QAIRT 2.39，手机当前未连接 |
| 手机 QNN 运行时兼容性 | **未证明**：host 生成的是 QNN 2.52 context，手机现有库是 2.39 |
| 完整 SD1.5 + AnimateDiff UNet | **未导出**：目前只验证了 motion module，不是整条视频链路 |

**当前判断**：motion module 的算子层面没有发现阻塞，可以继续；但还不能宣称“SD1.5 + AnimateDiff-Lightning 已在 NPU 上跑通”。下一步必须拿到 QAIRT 2.39，生成手机可加载的 context，并在真机完成 8 帧推理。

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

## 7. 兼容性风险（硬阻塞）

### 7.1 QNN 版本不一致

| 位置 | 版本 |
|------|------|
| host `onnxruntime-qnn` 内置 QNN | 2.52.0 |
| 生成的 context `ep_sdk_version` | `v2.52.0.261104115025` |
| 手机现有 QNN 库 | `AISW_VERSION 2.39.0` / `v2.39.0.250925215840_163802` |

QNN 有 `QNN_CONTEXT_ERROR_BINARY_VERSION` 这类错误；不能假设 2.52 生成的 context 能被 2.39 运行时加载。必须先取得 **QAIRT 2.39.0.250925**，用同一版本重新生成 context，再做真机验证。

### 7.2 只有 motion module，没有完整 UNet

本次验证的是 UNet 内部的时间模块，不是完整 SD1.5 + AnimateDiff 图。完整链路还需要：

1. SD1.5 diffusers UNet 权重；
2. `UNetMotionModel` 或等价的 UNet + motion module 合并图；
3. 固定 8 帧、256×256、4～6 步的完整去噪循环；
4. 文本编码器、VAE、scheduler 的 NPU/宿主边界定义；
5. 与现有 QNN 后端的输入输出接口对齐。

现有 SD1.5 QNN 包是整包预编译 context，无法插入 motion module；必须重新导出并编译完整 UNet。

### 7.3 真机状态

本轮没有连接手机，也没有运行真机推理；因此没有耗时、峰值内存、温度数据。

## 8. 下一步

1. 取得 QAIRT 2.39.0.250925，安装 `qnn-onnx-converter`、`qnn-context-binary-generator` 和 V79 运行时。
2. 用 2.39 重新生成 3 个 motion module context，确认手机能 `createFromBinary`。
3. 连接手机，做一个最小 native runner，只跑单模块 8 帧推理，记录耗时/内存/温度。
4. 下载 SD1.5 diffusers 权重，导出完整 `UNetMotionModel`，重复同样的 QNN 编译和真机验证。
5. 只有完整 UNet 的算子覆盖、真机耗时和内存都可接受，才进入 Android UI。

## 9. 证据文件

本地产物（均不入 Git）：

```text
models/onnx/*.onnx
models/onnx/export_manifest.json
models/qnn/*/model.qnn
models/qnn/*/model_qnn.bin
models/qnn/qnn_context_report.json
models/qnn_op_report.json
models/inventory.json
```

复现命令：

```bash
.venv/bin/python tools/model_inventory.py
.venv/bin/python tools/export_motion_onnx.py
.venv/bin/python tools/qnn_context_probe.py
.venv/bin/python tools/qnn_op_probe.py
```