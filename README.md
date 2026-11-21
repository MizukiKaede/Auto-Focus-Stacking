# Focus Stack Studio (Auto-stack)

**Focus Stack Studio（Auto-stack）** 是一款面向大批量商业与产品焦点堆栈摄影的 Windows 桌面自动景深合成工作台，基于 Python 3.11+、PySide6、ONNX Runtime、OpenCV 与 C++17 原生算子构建。

导入一整批照片后，程序能够**自动扫描排序、识别场景边界并分组、基于清晰度覆盖精选有用帧、执行亚像素对齐与全分辨率景深融合**，为每个拍摄组输出高品质 JPG 或无损 TIFF，并可选在校验通过后安全归档原图。

特别适用于大批量产品/商业摄影后期，且**不要求前期拍摄严格按照焦点远近顺序，也无需提前按组手动拆分文件夹**。

---

## 目录

- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [工作流与自动分组原理](#工作流与自动分组原理)
- [三种合成引擎](#三种合成引擎)
- [桌面工作台与内存控制](#桌面工作台与内存控制)
- [输出文件、恢复与归档安全](#输出文件恢复与归档安全)
- [原生 C++ 内核与便携打包](#原生-c-内核与便携打包)
- [项目结构与测试](#项目结构与测试)
- [第三方组件与许可](#第三方组件与许可)

---

## 核心特性

- **三大可选合成引擎**：
  - **AI 智能合成（`ai`，默认推荐）**：内置 MIT 协议 **StackMFF-V3** 多尺度焦点堆栈 ONNX 模型，支持像素级来源概率融合；自动按 `CUDA 12 → DirectML (NVIDIA / AMD / Intel GPU) → CPU` 优先级选择加速器。
  - **高质量模式（`quality`）**：全分辨率内存对齐与焦点归属，内置平坦区域稳定、材质色调协调、彩色印字边缘保护与主体外沿归属，配合 C++17 / OpenMP 原生算子加速。
  - **实验模式（`hugin_enfuse`）**：调用外部 `align_image_stack` 对齐并做受限仿射 ECC 残差微调，由程序生成全分辨率焦点硬蒙版（C++ 算子加速），再由 `enfuse` 渲染输出。
- **非顺序自动分组与精准选片**：
  - 结合 EXIF 拍摄时间、低频构图差异与白底孤立主体位移检测自动切分拍摄组，同时支持在源目录放置 `stack_groups.json` 手动固定分组。
  - 在 1280px 预览上完成仿射 ECC 配准与清晰度覆盖贪心选片，自动剔除冗余帧，只将真正贡献清晰区域的帧送入融合。
- **面向大批量任务的内存保护与并发架构**：
  - 分析与合成采用有界队列流水线并行，内置全局共享对齐缓存预算（默认 4 GiB）、单组缓存上限（默认 2 GiB）、**按可用内存自动降并发**以及一键**低内存模式**（1.0 GiB 安全红线）。
  - 基于 SQLite 持久化任务状态与计划缓存，支持异常中断后的过程恢复与可追溯的 `stack_manifest.csv` 清单。
- **现代摄影师桌面工作台**：
  - 支持**浅色工作室**与**低光暗房**双主题切换、文件夹拖拽导入、即时素材规模预检、5 阶段流水线进度条、精确秒表与可展开诊断日志。
  - 内置结果状态筛选、原图文件名模糊搜索定位、图组检查器以及支持滚轮缩放/拖拽与 1:1 切换的**全分辨率校对灯箱**。

---

## 快速开始

### 环境要求

- **操作系统**：Windows 10 / 11 64 位（推荐；亦可在 Linux / macOS 上使用 CPU 或 CUDA 版 ONNX Runtime 运行）
- **Python**：3.11 或更高版本
- **核心依赖**：`numpy`、`opencv-python`、`Pillow`、`psutil`、`PySide6`、`onnxruntime-directml`（Windows x64 默认）/ `onnxruntime`（其他平台）

### 从源码安装与启动

在 PowerShell 7（或任意终端）中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# 如需运行测试或安装 focus-stack 命令行入口：
pip install -e ".[dev]"
```

启动桌面工作台：

```powershell
python main.py
```

或使用安装后的命令入口：

```powershell
focus-stack
```

### 可选：启用 NVIDIA CUDA 12 GPU 加速

Windows 默认通过 `onnxruntime-directml`（约 25 MB）为 NVIDIA、AMD、Intel 显卡提供开箱即用的 FP16 GPU 推理加速，无需安装 CUDA 工具包。

如果本机已配置 **CUDA 12** 环境并希望获得更高的 AI 融合吞吐，可安装 CUDA 可选依赖（安装后程序会自动优先启用 `CUDAExecutionProvider`）：

```powershell
pip install -e ".[cuda]"
# 或直接安装：pip install onnxruntime-gpu>=1.20
```

> 若无可用 GPU，AI 引擎会自动回退至 FP32 ONNX 模型的 `CPUExecutionProvider`，并在大图上启用 Proxy 1600 提速策略；详见[AI 智能合成](#1-ai-智能合成ai默认推荐)。

### 使用 Windows 便携版（如从 Releases 下载）

若从 [GitHub Releases](https://github.com/MizukiKaede/Auto-stack/releases) 获取打包好的 Windows x64 便携版 ZIP：

1. 将压缩包**完整解压**到本地目录（切勿在压缩包内直接双击运行）。
2. 保留同目录下的 `_internal/` 文件夹（内含 Python 运行时、原生 C++ DLL、StackMFF-V3 权重及内置 Hugin/Enfuse 工具）。
3. 双击主程序 `.exe` 即可启动，无需额外安装 Python 或运行库。

---

## 工作流与自动分组原理

三种合成引擎共用同一套前置流水线，切换引擎仅改变融合与对应的对齐渲染阶段：

```text
扫描照片与 EXIF 排序 ─► 自动 / 手动分组 ─► 1280px 预览配准与清晰覆盖选片 ─► 所选引擎融合 ─► 编码保存 & 可选归档
```

### 1. 扫描与排序

- 默认扫描所选目录内的 `.jpg` / `.jpeg` 照片，不递归子目录、不扫描隐藏文件；可在界面勾选**包含子文件夹（递归扫描）**与**扫描隐藏文件/目录**。
- 拥有 EXIF 拍摄时间的照片优先按拍摄时间升序排列；时间相同时按文件名与路径自然排序（如 `IMG_2.jpg` 排在 `IMG_10.jpg` 之前）。
- 缺少拍摄时间的照片排在有时间照片之后，按自然顺序排列；此时时间间隔条件不生效，但仍会执行画面构图比较。

### 2. 自动场景分组

默认分组器 `StackGroupDetector` 读取长边不超过 640px 的预览图，进一步缩放到低频模糊表示以抑制正常对焦变化带来的高频细节差异，再逐张比较相邻照片。满足以下任一条件即切分新组：

| 边界条件 | 默认判定规则 |
| --- | --- |
| 预览尺寸或方向变化 | 横竖幅切换、分辨率或宽高比改变时立即切分 |
| 拍摄时间倒退 | 相邻两张均有有效时间且出现负时间差时切分 |
| 长时间停拍 | 相邻间隔 **超过 60 秒**（若界面“分组拍摄间隔”设得更大，则以较大值为准）强制切分 |
| 构图明显变化 | 低频归一化构图差异分数 `> 0.025`，即使连续拍摄也切分 |
| 停拍后画面略有变化 | 间隔达到**分组拍摄间隔**（默认 **20 秒**，可在界面 `1–3600 秒` 调节）且构图差异分数 `> 0.006` 时切分 |
| 白底孤立主体变化 | 在大面积明亮中性背景下的孤立较暗主体场景中，检测主体包围盒区域差异（`> 0.09`），识别转面或明显位移，同时容忍小幅对焦呼吸 |

> **说明**：`20 秒` 停拍本身不会无条件切分，20～60 秒区间需要伴随一定的画面变化；默认超过 `60 秒` 才会仅凭时间强制开始新组。

### 3. 组内智能选片

“识别出一组”不等于“该组所有照片都进入融合”：

1. **最小张数门槛**：界面**最少相似照片数**默认为 **4 张**（可调范围 `2–1000`）。低于该数量的组会被分类记录为无需合成，不会触发融合。
2. **参考帧与预览对齐**：在长边 **1280px** 预览上计算全画面焦点评分，选最高分为参考帧，并对其余帧做仿射 ECC 配准；配准失败的帧会被记录原因并排除候选。选片阶段同时校验画面尺寸一致性与主体是否发生转面/大位移。
3. **贪心清晰覆盖选择**：在有效细节区域中，某像素清晰度达到全组该位置最佳值的 **90%** 即视为被该帧覆盖。算法迭代选取能补充最多未覆盖区域的帧，当新增覆盖率低于 **0.05%** 或总覆盖率达到 **99.95%** 时停止，并确保参考帧始终保留。
4. **有效提交**：至少选出 2 张有效帧且有效组规模满足最小张数要求时，才进入后续融合。被跳过的冗余帧不会被删除。

### 4. 手动固定分组（`stack_groups.json`）

如需对特定照片强制划组，可在源照片目录根部放置 `stack_groups.json`：

```json
{
  "groups": [
    ["IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg", "IMG_0004.jpg"],
    ["sub/IMG_0005.jpg", "sub/IMG_0006.jpg", "sub/IMG_0007.jpg", "sub/IMG_0008.jpg"]
  ]
}
```

- 每个内层数组代表一个显式组，**优先于自动分组**；未列出的照片仍按连续片段走自动分组。
- 支持填写文件名或相对于源目录的相对路径（递归扫描存在同名文件时请使用相对路径消歧），匹配不区分大小写，不允许越出源目录或重复列出同一文件。
- 手动分组依然会经过最小张数、尺寸一致性、对齐与清晰覆盖选片校验。

实现入口：[scanner.py](focus_stack_app/core/scanner.py)、[group_detector.py](focus_stack_app/core/group_detector.py)、[manual_groups.py](focus_stack_app/core/manual_groups.py)、[whole_frame_selection.py](focus_stack_app/core/whole_frame_selection.py)。

---

## 三种合成引擎

可在界面左侧**高级与性能设置 → 合成引擎**切换引擎（或在配置中设置 `runtime.fusion_backend`）。运行中**不会**在某个引擎失败后静默切换到另一个引擎，避免产生不可预期的画质差异。

| 引擎 | 配置值 | 对齐与焦点判定 | 融合与渲染策略 | 外部/硬件依赖 | 适用场景 |
| --- | --- | --- | --- | --- | --- |
| **AI 智能合成**（默认推荐） | `ai` | 复用预览仿射变换 + 全分辨率 Lanczos4 映射；StackMFF-V3 多尺度预测每像素来源概率 | 显存自适应分块（Tile）与余弦羽化无缝拼接；自动剔除反射填充无效边界 | 内置 ONNX 权重 + `onnxruntime-directml` / `onnxruntime-gpu` / CPU | 日常绝大多数商业与产品堆栈，顺序无关、边缘自然、综合体验最佳 |
| **高质量合成引擎** | `quality` | 复用预览仿射变换 + 全分辨率 Lanczos4 内存对齐；全分辨率局部焦点标签 | 平坦纹理门控、印字边缘保护、外沿归属、材质色调协调 + 单层窄过渡混合 | 无外部工具；内置 C++17 DLL（失败自动降级 NumPy/OpenCV） | 纯传统确定性算法路线，适合需要特定传统边缘/色调规则或对比校对的场景 |
| **实验模式（Hugin / Enfuse）** | `hugin_enfuse` | `align_image_stack` 独立对齐输出 TIFF + 受限仿射 ECC 残差微调 | 程序全分辨率计算焦点硬蒙版（C++ 加速）与色调修正，交由 `enfuse --load-masks` 渲染 | 需要 `align_image_stack` 与 `enfuse`（便携版已内置） | 需比较 Hugin 几何对齐路线或大视差/特殊镜头畸变场景（中间文件较多） |

> **旧配置值兼容**（见 [fusion_modes.py](focus_stack_app/fusion_modes.py)）：
> - `fast`、`speed`、`fast_fusion`、`stackmff`、`stackmff_v3`、`ai_fusion` 自动映射到 `ai`；
> - `standard`、`opencv` 自动映射到 `quality`；
> - `hugin`、`experimental` 自动映射到 `hugin_enfuse`。

### 1. AI 智能合成（`ai`，默认推荐）

实现入口：[ai_backend.py](focus_stack_app/fusion/ai_backend.py)、[ai_runtime.py](focus_stack_app/fusion/ai_runtime.py)。

- **加速器自动探测与优先级**：
  1. `CUDAExecutionProvider`（安装了 `onnxruntime-gpu` 且本机 CUDA 12 可用，加载 FP16 权重 `stackmff_v3_final_fp16.onnx`）
  2. `DmlExecutionProvider`（Windows 默认 `onnxruntime-directml`，加载 FP16 权重，支持 NVIDIA / AMD / Intel GPU）
  3. `CPUExecutionProvider`（无 GPU 时自动回退，强制加载 FP32 权重 `stackmff_v3_final.onnx` 以避免 CPU 模拟 FP16 降速，并对大图自动启用长边 1600px 的 Proxy 推理提速）
- **多尺度推理与无缝分块**：默认在 `1.0, 0.5, 0.25` 多尺度融合概率，按加速器和帧数自动选择分块尺寸（`256` / `384` / `512` / `1024`，搭配 `64` / `128` 重叠与二维余弦窗羽化）避免显存溢出与拼缝。
- **有效足迹掩码**：通过 `_FrameFootprint` 精确追踪每帧逆变换后的原始拍摄边界，在 Softmax 前屏蔽反射填充像素，杜绝画面边缘出现伪影。
- **自适应低内存策略**：当整组全分辨率 RGB 缓存超过单组/全局预算或开启**低内存模式**时，自动切换为**双遍 Top-K 分块融合**（仅保留每像素 Top-4 来源索引与权重），大幅压低峰值内存占用。

### 2. 高质量合成引擎（`quality`）

实现入口：[backends.py](focus_stack_app/fusion/backends.py)、[quality_fusion.py](focus_stack_app/fusion/quality_fusion.py)、[focus_masks.py](focus_stack_app/fusion/focus_masks.py)。

- 将选片阶段的预览变换换算到原图坐标，以 `cv2.INTER_LANCZOS4` 在内存中逐帧对齐到参考帧画幅，不生成磁盘中间 TIFF。
- 在全分辨率计算焦点归属标签，结合平坦纹理统计、彩色印字边缘保护（`quality_printed_edge_guard`）、主体外沿归属与材质成对色调协调（`quality_surface_tone`），降低雪花噪点、色块、印字晕边与外沿混入失焦背景的问题。
- 默认采用窄过渡单层融合并按共享缓存预算管理对齐帧，缓存不足时自动淘汰并重新解码；支持 EXIF 方向矫正与 ICC 色彩配置文件保留。

### 3. 实验模式（`hugin_enfuse`）

实现入口：[backends.py](focus_stack_app/fusion/backends.py)、[enfuse.py](focus_stack_app/hugin/enfuse.py)、[hugin_locator.py](focus_stack_app/hugin/hugin_locator.py)。

- 调用 `align_image_stack` 生成对齐 TIFF，自动校验输出数量、尺寸、主体一致性和裁切比例，可重试失败会自动调整参数或顺序。
- 在 Hugin 对齐结果上执行组内串行、组间并行的受限仿射 ECC 残差微调；任一帧未通过相关性/位移/缩放门槛时，整组回退到 Hugin 原坐标，确保几何一致。
- 默认由程序计算全分辨率 0/255 硬蒙版（受进程共享的 12 CPU 预算和内存准入控制，外沿归属、内部纹理相干性与硬蒙版生成均由 C++ 算子加速），再交给 `enfuse --load-masks` 渲染。
- **Hugin 工具查找顺序**：界面“Hugin 工具目录”显式路径 → 环境变量 `HUGIN_BIN` / `HUGIN_PATH` / `HUGIN_HOME` → 项目内 `.runtime-deps/hugin/bin` → 系统 `PATH` → 常见 Windows 安装目录。

---

## 桌面工作台与内存控制

### 界面功能一览

- **顶部工作台栏**：显示应用标题、精确秒表（`00:00:00`）、实时状态药丸，以及**浅色工作室 / 暗房模式**一键切换按钮。
- **左侧控制台**：
  - 支持直接将照片文件夹**拖入窗口**，即时统计 JPG 数量、总容量，并自动提示是否检测到 `stack_groups.json`。
  - 常用参数直观呈现：源目录/结果目录一键在资源管理器打开、递归扫描开关、归档抽屉、最少相似照片数、分组拍摄间隔、输出格式（100% 品质 4:4:4 JPG / 无损 TIFF）。
  - 折叠式**高级与性能设置**：合成引擎选择及实时 AI/Hugin 运行时状态探测、一键**低内存模式**、并发合成组数、自动降并发开关、全局/单组对齐缓存上限、并发焦点分析数与一键恢复默认参数。
- **右侧校对工作区**：
  - **进度驾驶舱**：5 阶段步骤条（扫描分组 → 焦点分析 → 图像对齐 → 景深融合 → 完成）、实时总体与分阶段进度、预计剩余时间，以及可展开/一键复制的运行诊断抽屉。
  - **图组结果表与检查器**：按`全部` / `已完成` / `无需合成` / `失败`筛选，支持输入原图文件名片段模糊搜索定位图组；右侧检查器呈现原图范围、精选张数、清晰覆盖率、引擎/诊断信息与大图缩略图，点击或双击即可打开**内置 1:1 缩放/平移校对灯箱**。

### 正常模式 vs 低内存模式预设

界面勾选**低内存模式**会自动在以下两组参数间切换（定义于 [config.py](focus_stack_app/config.py)）：

| 参数项 | 正常模式（默认） | 低内存模式 |
| --- | --- | --- |
| 安全可用内存红线 | `2.0 GiB`（且保留物理内存 `10%`） | `1.0 GiB`（不叠加百分比保留） |
| 并发合成组数上限 (`merge_workers`) | `3` 组 | `1` 组 |
| 按可用内存自动降并发 (`auto_merge_workers`) | 开启 | 开启 |
| 分类与合成流水线并行 (`parallel`) | 开启 | 关闭 |
| 并发焦点分析数 (`focus_analysis_workers`) | `0`（自动推荐） | `1` 张 |
| 全局共享对齐缓存 (`fusion_cache_budget_bytes`) | `4.0 GiB` | `1.0 GiB` |
| 单组对齐缓存上限 (`opencv_aligned_cache_bytes`) | `2.0 GiB` | `0.5 GiB` |
| 选片预览帧缓存 (`selection_frame_cache_bytes`) | `400 MiB` | `128 MiB` |
| Hugin 对齐 TIFF 缓存 (`aligned_tiff_cache_bytes`) | `1.0 GiB` | `256 MiB` |

---

## 输出文件、恢复与归档安全

- **结果目录与命名**：
  - 默认输出到源文件夹下的 `合成/` 目录（可自定义），文件名以该组首张原图名称加 `_stack` 后缀生成（如 `DSC00522_stack.jpg` 或 `DSC00522_stack.tif`）。
  - **严格禁止覆盖原图**：`output.overwrite` 默认为 `False`，绝不会覆写任何输入照片或已存在的输出文件。
- **结果清单与运行缓存**：
  - 输出目录根部生成 `stack_manifest.csv`，记录每组状态、原图列表、精选帧、参考帧、覆盖率、实际使用的合成引擎、输出路径与诊断码。
  - 输出目录下的 `.stack_cache/` 保存 `database.sqlite`（任务与分析计划持久化）、`logs/application.log`（运行日志）以及任务临时工作目录。成功完成且未开启“保留焦点分析缓存”时自动清理临时目录；失败或取消的现场会保留便于排查。
- **可选原图归档**：
  - 默认关闭；开启后支持**移动**、**复制**或**硬链接**三种方式归档至独立归档目录（默认源目录下的 `归档/`）。
  - 仅在合成文件成功写出并通过可读性校验后才执行归档；若合成失败或取消，原图保持不动。

---

## 原生 C++ 内核与便携打包

### C++17 原生算子（Windows x64）

项目在 `focus_stack_app/fusion/` 随附预编译的 Windows x64 原生库（默认由 MinGW-w64 GCC 13 + OpenMP 构建，开启 `-fno-fast-math -ffp-contract=off` 保证数值一致性）：

- `statistics_native.dll`：加速清晰覆盖选片、高质量模式平坦噪点/色度统计、RGB 累加与材质色调采样。
- `fast_core.dll`：加速高质量与 Hugin 模式的外沿最近支持归属（`nearest_support`）、内部纹理结构张量判定及硬蒙版生成。

当 DLL 缺失、依赖缺失、符号不全或 ABI 不匹配时，程序会记录一次日志并自动降级到算法等价的 NumPy / OpenCV 实现继续执行，不会切换合成引擎。设置环境变量 `FOCUS_STACK_FORCE_NUMPY=1` 可强制走纯 NumPy 诊断路径。

如需自行在 Windows x64 上重新编译原生库（默认输出到 `build/native`，不覆盖随应用附带的 DLL）：

```powershell
python build_cpp.py --compiler auto --openmp on
```

### Windows 便携版打包

仓库提供基于 PyInstaller 的独立便携版打包脚本 [portable-packaging/build_portable.py](portable-packaging/build_portable.py)，会自动收拢 `focus_stack_app` 子模块、SVG 图标、原生 C++ DLL、StackMFF-V3 ONNX 权重、内置 Hugin/Enfuse 运行时与第三方许可声明。

在仓库根目录的 Windows x64 Python 环境运行以下命令。Hugin/Enfuse 二进制文件从 `.runtime-deps/hugin/bin` 读取，便携目录生成到 `build/releases/current/portable/AutoStack-main`。

```powershell
.\.venv\Scripts\python.exe -m pip install PyInstaller pefile
.\.venv\Scripts\python.exe portable-packaging/build_portable.py main
```

---

## 项目结构与测试

### 目录结构

```text
.
├── main.py                         桌面应用启动入口
├── build_cpp.py                    Windows x64 原生 C++ DLL 构建脚本
├── pyproject.toml                  包元数据、依赖与 pytest 配置
├── requirements.txt                运行依赖
├── requirements-dev.txt            开发与测试依赖
├── THIRD_PARTY_NOTICES.md          第三方模型权重与组件许可及校验值
├── focus_stack_app/
│   ├── app.py                      Qt 应用与主窗口引导
│   ├── config.py                   中央配置与正常/低内存预设
│   ├── fusion_modes.py             合成模式规范化与旧配置别名迁移
│   ├── core/                       扫描排序、场景分组、手动分组、覆盖选片与配准
│   ├── fusion/                     AI(StackMFF-V3)、高质量融合、C++17 源码/DLL 与 ONNX 权重
│   ├── hugin/                      Hugin/Enfuse 探测、对齐、残差微调、硬蒙版与输出编码
│   ├── pipeline/                   应用控制器、有界流水线协调器、工作线程与内存守卫
│   ├── storage/                    SQLite 状态持久化、磁盘缓存与 stack_manifest.csv 生成
│   ├── ui/                         PySide6 浅色/暗房工作台、进度驾驶舱、图组检查器与 1:1 灯箱
│   └── utils/                      图像 I/O、EXIF、自然排序、内存预算与性能诊断工具
├── portable-packaging/             PyInstaller Windows 便携包构建脚本
└── tests/                          51 个自动化测试模块（核心算法、融合后端、并发恢复与 UI）
```

### 运行测试

项目包含覆盖分组选片、AI / Quality / Hugin 融合后端、C++ 原生降级、内存与缓存预算、断点恢复及 UI 契约的 51 个测试模块：

```powershell
# 运行完整测试套件
pytest

# 按功能分组运行示例
pytest -q tests/test_fusion_backends.py tests/test_quality_fusion.py tests/test_native_reliability.py
pytest -q tests/test_pipeline_app.py tests/test_pipeline_recovery.py tests/test_parallel_batch_safety.py
pytest -q tests/test_studio_ui_features.py tests/test_memory_presets_and_ui.py
```

---

## 第三方组件与许可

- **StackMFF-V3（`v2.0` MIT Release）**：内置 ONNX 权重（`stackmff_v3_final.onnx`、`stackmff_v3_final_fp16.onnx`）导出自 MIT 许可的 `v2.0` 检查点，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- **ONNX Runtime / ONNX Runtime DirectML**：MIT License（Microsoft Corporation）。
- **C++ 原生运行时库**：随应用附带的 MinGW-w64 / GCC 运行时 DLL 及其对应许可证位于 `focus_stack_app/fusion/native_runtime_licenses/`。
- **Hugin（`align_image_stack`）与 Enfuse**：实验模式通过独立子进程调用外部命令行工具，它们沿用各自上游 GPL 许可与版权声明，本项目不改变其授权。
