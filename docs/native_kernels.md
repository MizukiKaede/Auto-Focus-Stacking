# Windows x64 原生算子

Quality 和 Fast 使用 C++17 像素算子，OpenCV 继续提供图像处理。选片、矩阵、选源、
修色参数、帧顺序和接缝规则不因执行路径改变。加载失败只切换算子实现，不切换合成模式。

## 构建

使用 64 位 Python 和 Windows x64 编译器，在仓库根目录运行：

```powershell
python build_cpp.py --compiler auto --openmp on
python build_cpp.py --compiler mingw --compiler-path C:/Strawberry/c/bin/g++.exe --openmp off --output-dir build/native-serial
python build_cpp.py --compiler msvc --openmp on --output-dir build/native-msvc
```

`auto` 优先使用 PATH 上的 MinGW g++，否则寻找 MSVC x64 工具环境。MSVC 可在
x64 Native Tools 命令行中运行，或由脚本通过 vswhere 定位。不会安装工具链。
本轮本机 MinGW 13.1.0 的 OpenMP 开/关构建均成功；已采用开启 OpenMP 的实图验收 DLL。
本机没有 MSVC，MSVC 构建路径尚未实际编译验收。

MinGW 使用 `-std=c++17 -O2 -fno-fast-math -ffp-contract=off -shared`，静态链接
libgcc/libstdc++；OpenMP 使用 `-fopenmp`。MSVC 使用 `/std:c++17 /O2 /EHsc /fp:strict /MT /LD`，
OpenMP 使用 `/openmp`。默认不要求 AVX2。构建输出包括两 DLL、实际需要的运行时 DLL、
`build_manifest.json` 和运行时许可证。整批编译、架构与加载检查成功后才发布文件。
OpenMP 开关依据 [GCC 文档](https://gcc.gnu.org/onlinedocs/gcc/OpenMP.html) 和
[MSVC 文档](https://learn.microsoft.com/en-us/cpp/build/reference/openmp-enable-openmp-2-0-support?view=msvc-170)。

MinGW 的 OpenMP 可能依赖 libgomp、libwinpthread、libgcc 或 libdl；它们由同一工具链
收集，不能只复制两个业务 DLL。MSVC OpenMP 使用工具链提供的 x64 vcomp140.dll。
正式发布时把已验收构建的 DLL 和许可证复制到 `focus_stack_app/fusion`，然后重新打包。
Python 的加载器保留该目录的 DLL 搜索句柄，避免依赖用户 PATH。

## 执行和线程

`RuntimeConfig.native_threads` 的 `0` 为按模式验收后的自动预算，`1` 强制单线程，
更大的整数为显式并行上限。自动值最多 3，受 CPU 数、合成组数、并行分析占用及
OpenCV 线程上限约束。没有通过实图验收的模式自动值保持 1。

本轮 00522（50 帧）和 03003（21 帧）的原生并行与强制 NumPy 路径，均保持标签、
编码前 RGB 和成品解码像素精确一致。Fast 单次整组耗时分别从 82.677 秒降至
70.461 秒、39.397 秒降至 29.709 秒，因此启用 Fast 自动预算。Quality 默认串行：
00522 测量期间用户确认有后台 CPU 大进程，且 profile 显示缓存动态降额，耗时
不可直接比较；不能据此判断并行性能。本轮未扩大或重复实图矩阵。
完整环境、耗时、峰值内存及比较证据见
`diagnostics/native_reliability_20261004_v1/validation_report.md`。

每次 ctypes 调用在调用线程设置 C++ thread_local 预算，未修改全局 OpenMP 环境。
少于 262144 个像素的循环串行；共享容器构建、噪声分组和 tile 操作仍串行。
同一个 RenderAccumulator 不允许多个调用者同时操作，帧按原顺序累加。
旧 DLL 缺少线程能力导出时可继续运行，但按单线程记录。

`statistics_native_runtime`、`fast_cpp_execution` 和 `native_thread_budget` profile 事件
记录实现、加载原因、进程累计调用数和线程上限；上限不表示每个小循环都启动了该数量的线程。

## 降级和安全

DLL 缺失、依赖缺失、符号不全或 ABI 不兼容会记录一次日志并使用 NumPy/OpenCV。
统计库兼容没有 ABI 导出的旧库；Fast 的 ABI 必须为 3。
`FOCUS_STACK_FORCE_NUMPY=1` 可用于诊断，必须在新进程导入模块前设置；这是故障模拟
与回归接口，不影响保存的合成模式。大图 NumPy 计算以块执行，避免整组 RGB 常驻。

原地输出要求匹配类型、形状、连续且可写，不会通过复制输出掩盖错误。非法 probe 类别
在 Python 写入前报错，C++ 同时跳过非法类别。累加器使用上下文协议，在取消或异常时
确定释放；关闭后调用报错，重复关闭安全。

接缝无有效贡献时保留对齐后的参考图，包括真正的黑色像素，不增加插值补洞规则。
采用条件为指定样本标签和编码前 RGB 精确一致；浮点中间量容差不能代替最终结果检查。
