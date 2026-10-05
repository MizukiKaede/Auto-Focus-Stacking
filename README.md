# Focus Stack Assistant

Focus Stack Assistant 是一个用于焦点堆栈摄影的桌面工具，帮助整理照片、分析焦点覆盖范围、完成图像对齐，并生成焦点合成结果。

项目使用 Python 和 PySide6 编写，重点关注大批量照片处理时的内存占用、任务并发和过程恢复能力。

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

## 环境要求

- Python 3.11 或更高版本
- Windows 环境建议使用 PowerShell 7

主要依赖：NumPy、OpenCV、Pillow、psutil、PySide6。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e ".[dev]"
```

## 运行

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

三种模式共用扫描、分组和选片。界面中的“实验模式”需要可用的
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
