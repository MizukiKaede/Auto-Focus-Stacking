# C++ 统计与色调校正（2026-10-03）

已采用项目外 `D:\TEMP\cpp_statistics_20261003\candidate_v2` 的验证版本。

`focus_stack_app/fusion/statistics_native.cpp` 是原生实现，旁边的
`statistics_native.py` 通过 ctypes 调用 `statistics_native.dll`。
随源码提供的 DLL 适用于 Windows x64，使用 Zig 0.16.0 的 C++ 编译器构建。
运行时不需要安装编译器。DLL 与 C++ 源码已加入 setuptools 包数据。

迁移范围：

- `SurfaceToneHarmonizer`：材质探针累积、材质分类、掩膜乘法、密度归一化、补偿权重及像素回写。
- `FlatTextureStatistics`：亮度分箱、分位数、median/MAD 噪声统计和色差计算。
- `SurfaceBoundaryOwnership` / `PrintedEdgeOwnership`：复用原生色差计算。
- `ValidSourceOwnership` 未修改；当前 Quality 路径不调用 `_CoherentNeutralEdges`。

OpenCV 滤波、形态学及融合规则保持原有实现。没有静默 Python 回退；DLL 缺失或平台不匹配时导入会报错。

## 重建

在 `focus_stack_app/fusion` 目录执行（把编译器路径改成实际路径）：

```powershell
& 'D:\TEMP\cpp_statistics_20261003\toolchain\ziglang\zig.exe' c++ -O3 -ffp-contract=off -shared -o statistics_native.dll statistics_native.cpp
```

不要加入 `-static`：此工具链会因此生成静态归档而非可加载 DLL。
禁止 fast-math 和浮点收缩，以保留 NumPy float32 运算与舍入行为。
更换平台需重新构建并验证加载方式；当前交付仅验证 Windows x64。

## 已完成的验收

GPT-6 Luna（xhigh）执行了 42 项定向检查，并检查 float32 阈值、半值取偶舍入和截断。
0522 使用既有 50 帧选片和矩阵，参考帧 DSC00557，Quality cached/gate/localized，
OpenCV 线程数 3，JPEG100/4:4:4。

| 项目 | 原版 | C++ 版 |
|---|---:|---:|
| 完整融合（扣除捕获写盘） | 190.582 秒 | 103.266 秒 |
| SurfaceToneHarmonizer.correct | 106.036 秒 | 32.300 秒 |

单对完整融合耗时减少 45.82%；这是一次配对实测，不是多轮中位数。
焦点标签、编码前 RGB 与最终 JPEG 解码像素全部完全一致。

完整报告与原始证据保留于 `D:\TEMP\cpp_statistics_20261003\performance`。
