# Focus Stack Assistant

Focus Stack Assistant 是一个用于焦点堆栈摄影的桌面工具，帮助整理照片、分析焦点覆盖范围、完成图像对齐，并生成焦点合成结果。

项目使用 Python 和 PySide6 编写，重点关注大批量照片处理时的内存占用、任务并发和过程恢复能力。

## 主要功能

- 扫描和整理照片元数据
- 自动检测场景并分组
- 分析焦点区域与覆盖范围
- 图像对齐、质量评估和焦点融合
- 高质量模式（默认）：全分辨率内存对齐、焦点融合和边缘一致性处理
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

两种模式共用扫描、分组和选片。界面中的“实验模式”需要可用的
`align_image_stack` 与 `enfuse`；程序会按显式路径、`HUGIN_BIN` 等环境变量、
项目自带运行时和系统路径查找。旧配置值 `opencv` / `fast` 会映射到高质量模式。

完整实现与系统说明见 [技术文档](docs/技术文档.md)、[技术架构](docs/技术架构.md) 及 [重构UI说明](docs/重构ui.md)。

本地照片、生成结果、诊断日志、缓存、虚拟环境和运行时依赖不会提交到 Git。
