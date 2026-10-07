# Focus Stack Assistant · dev 开发预览版

Focus Stack Assistant 是一个用于焦点堆栈摄影的桌面工具，帮助整理照片、分析焦点覆盖范围、完成图像对齐，并生成焦点合成结果。

项目使用 Python 和 PySide6 编写，重点关注大批量照片处理时的内存占用、任务并发和过程恢复能力。

## Windows 便携版下载

当前 README 对应 **dev 开发预览版**。本次打包日期：**2026-10-07**（北京时间）。

- [下载本分支便携包：AutoStack-dev-win64.zip](https://github.com/MizukiKaede/Auto-stack/releases/download/dev-win64-2026.10.07/AutoStack-dev-win64.zip)
- [查看本版发布说明、校验值与第三方源码附件](https://github.com/MizukiKaede/Auto-stack/releases/tag/dev-win64-2026.10.07)
- [查看另一分支 README](https://github.com/MizukiKaede/Auto-stack/blob/main/README.md)

适用于 **Windows 10/11 64 位**。完整解压 ZIP，然后双击 `AutoStack-dev.exe` 即可启动，无需安装 Python、Hugin 或微软运行库。
请保留旁边的 `_internal/` 文件夹，不要只复制 exe，也不要在 ZIP 内直接运行。两个版本分别解压，避免覆盖依赖。

本次只检查了主界面启动、SVG 图标、原生 DLL 和内置 Hugin/Enfuse 加载；未执行完整测试、照片融合或性能测试。
便携包内的 `BUILD-INFO.json` 记录实际打包提交与依赖版本；本 README 的后续文档提交不改变已发布的程序算法。

## 两个版本如何选择

| 版本 | 用途 | 本次便携包对应提交 | 主要区别 |
| --- | --- | --- | --- |
| [main 常规版](https://github.com/MizukiKaede/Auto-stack/releases/tag/main-win64-2026.10.07) | 日常使用的常规发布入口 | `67b606e` | 当前 main 分支的高质量、Fast C++、Hugin/Enfuse 流程 |
| [dev 开发预览版](https://github.com/MizukiKaede/Auto-stack/releases/tag/dev-win64-2026.10.07) | 体验开发分支改动、反馈问题 | `6b09168` | 增加原生内核线程预算与降级处理，以及 Hugin 蒙版/纹理算子和调度改动 |

两版都提供高质量（默认）、Fast 和实验模式。dev 的新增实现不代表每个场景都更快或效果更好；本次发布未执行照片融合回归或整批性能测试。

## 内置 Hugin / Enfuse 与第三方资料

便携版内置未经修改的 `align_image_stack`（Hugin 2025.0.1）和 `enfuse`（4.3，提交 `993fd4822e8d`），以独立命令行程序处理图像文件。
它们沿用上游的 GPL 许可证与版权声明；本项目不声称这些工具是自研组件，也不改变其授权。
包内 `LICENSES/` 保留许可证和第三方说明，`SOURCE.zip` 保存该版本 Auto-stack 源码及打包适配。
同一 Release 的第三方源码附件和来源清单提供内置开源组件的源码资料；如转发便携包，请一并保留许可证与源码获取说明。

## 主要功能

- 扫描和整理照片元数据
- 自动检测场景并分组
- 分析焦点区域与覆盖范围
- 图像对齐、质量评估和焦点融合
- 高质量模式（默认）：全分辨率内存对齐、焦点融合和边缘一致性处理
- Fast 快速模式：C++ 核心、代理图选源和一次全分辨率融合，包含小孔与边缘修复
- 实验模式：使用原有的 Hugin 对齐与 Enfuse 合成流程
- 本地缓存、任务队列和资源保护
- 提供较完整的单元测试与并发测试

## 从源码运行：环境要求

- Python 3.11 或更高版本
- Windows 环境建议使用 PowerShell 7

主要依赖：NumPy、OpenCV、Pillow、psutil、PySide6。

## 从源码运行：安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e ".[dev]"
```

## 从源码运行：启动

```powershell
python main.py
```

也可以使用安装后的命令：

```powershell
focus-stack
```

## 测试

```powershell
pytest
```

## 项目结构

```text
focus_stack_app/
├── core/       核心分析、分组、质量和选帧逻辑
├── files/      文件归档与命名冲突处理
├── fusion/     图像融合与焦点蒙版
├── hugin/      Hugin / Enfuse 集成
├── pipeline/   任务队列、工作线程和流程控制
├── storage/    数据库、清单和缓存
├── ui/         PySide6 图形界面
└── utils/      图像、EXIF、日志和资源工具
tests/          自动化测试
```

三种模式共用扫描、分组和选片。从源码运行时，界面中的“实验模式”需要可用的
`align_image_stack` 与 `enfuse`；程序会按显式路径、`HUGIN_BIN` 等环境变量、
项目自带运行时和系统路径查找。旧配置值 `opencv` 仍映射到高质量模式；`fast` 现在选择独立的 Fast 模式。

在界面高级设置的“合成引擎”中选择 **Fast 快速合成（C++）**。默认仍为高质量模式。
Fast 使用随项目附带的 Windows 64 位 `fast_core.dll`，版本为 `fast-cpp-v4-print-statistics`；
原高质量、Hugin/Enfuse 的算法及默认参数保持不变。

Quality/Fast 的原生库无法加载时，会记录原因并使用相同算法的 NumPy/OpenCV 实现继续合成，
耗时可能增加。详情见 [原生构建与降级说明](docs/native_kernels.md)。
Windows x64 可执行 `python build_cpp.py --compiler auto --openmp on` 构建两个库，
默认保存到 `build/native`，不会自动替换随应用附带的 DLL。

### 本版 Hugin / Enfuse 改动

- Hugin 外沿归属复用 Fast C++ 内核中的 `nearest_support`；内部纹理的结构张量相干性判定和焦点硬蒙版生成改用原生 C++ 算子。
- 硬蒙版仍输出 0/255、Deflate 压缩的 TIFF；蒙版任务受进程共享的 12 CPU 预算和内存准入限制，取消时会停止提交后续任务并等待在途任务退出。
- 残差 ECC 按合成组逐帧串行执行，组与组之间仍可并行；保留 OpenCV 线程设置及原有残差验收门槛，任一帧不通过时整组回退到 Hugin 原坐标。
- 引用对话中的定向验收：7 项检查通过；00522 的 50 帧复测中，首轮对齐通过，49 次 ECC 串行完成且无异常，14 帧未通过原残差门槛并按原规则回退。未执行完整融合或本版整批速度测试。
- 历史 3.4 批次曾比 3.3 慢；这些记录不是最终串行 ECC 版本的性能结论，因此本版不宣称整批提速。

完整实现与系统说明见 [技术文档](docs/技术文档.md)、[技术架构](docs/技术架构.md) 及 [重构UI说明](docs/重构ui.md)。

本地照片、生成结果、诊断日志、缓存、虚拟环境和 `.runtime-deps` 开发依赖不会提交到 Git。
原生内核所需的可分发 DLL 与许可证随应用打包。
