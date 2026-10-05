# Hugin 原生算子与准备阶段并发

Hugin 外沿归属复用 `fast_cpp.nearest_support`，内部纹理的相干性判定和硬蒙版生成使用新增 C++ 算子。Sobel、高斯滤波、距离变换、ECC 和 TIFF 编码继续调用已有 OpenCV / Pillow 接口。阈值、选源顺序及 Enfuse 参数保持原值。

`RuntimeConfig.hugin_parallel_cpu_budget` 默认 12，接受 1–12；1 关闭蒙版帧间并发。应用将该值传给真实 Enfuser，并在已有 OpenCV 线程预算入口限制内部线程数。直接调用 Enfuser 时，同名关键字默认 12。

硬蒙版任务在同一 Python 进程内共享加权 CPU 预算，最多并发十二个任务，内存预留可能进一步降低实际并发。蒙版任务每个计费 1，C++ 蒙版循环没有内部 OpenMP 并发。该预算不限制残差 ECC、原 Hugin 对齐、外部 Enfuse、分析或其他原有计算，也不是 CPU 亲和性设置。

残差 ECC 恢复为每组在调用线程逐帧串行计算：按输入顺序读取缓存、缩小为灰度图并估计矩阵，不排队灰度图，也不申请共享蒙版预算。不同合成组仍可独立同时运行；每次 ECC 保留应用级 OpenCV 内部线程设置。`HuginAlignmentRefiner.cpu_budget` 参数保留兼容，但不调度 ECC。矩阵和诊断完成后统一提交，任何原门槛拒绝仍使全栈回退到 Hugin 原几何。蒙版的共享内存准入保留至少 2 GiB 或物理内存的 10%，内存不足时减少在途缓冲。

硬蒙版保持 `hardmask-N.tif` 编号、0/255 uint8 像素及 Deflate 压缩。取消或任务失败会停止提交、取消尚未启动的任务，等待执行中的任务释放资源；只有全部蒙版完成才运行 Enfuse。每个任务继承独立的性能记录上下文。

新增 DLL 符号分别可选绑定。旧 DLL 缺少某个新符号时，仅该算子使用分块 NumPy 实现；已有 ABI=3 的接口继续可用。`FOCUS_STACK_FORCE_NUMPY=1` 保留全 NumPy 诊断路径。构建沿用 `build_cpp.py` 和原来的浮点编译参数，不启用 fast-math。

Profile 的 `hugin_preparation_budget` / `hugin_preparation_execution` 记录蒙版预算、任务峰值和共享 CPU 准入峰值；共享峰值明确是进程生命周期统计。`hugin_residual_execution` 记录每组 ECC 串行执行及 OpenCV 内部线程数。`hugin_native_execution` 记录原生库及新增接口的加载状态。具体样本验收和当次速度记录另存诊断报告。
