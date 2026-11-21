# Hugin 材质与边缘修复

本次只修改 `hugin_enfuse` 后端。产品 Quality 后端的合成类、渲染方式、选片、配准矩阵及现有颜色/边缘组件保持原样。

## 问题与处理

2026-10-01 的 `测试集/合成/3/DSC00522_stack.jpg` 使用 50 张原片。原 Hugin 路径虽然蒙版模式名为 `quality`，却只执行旧焦点标签和中性区中值；它没有调用现有的外沿、印字和材质颜色修复。检测到颜色漂移后自动使用五层 Enfuse 混合，可能将失焦轮廓带入清晰主体。原成品的中间 TIFF 已清理，不能据此完全排除配准因素。

三个合成 worker 同时临时修改同一个 aligner 的配置。日志出现一级命令包含后续重试参数。现在每次 Hugin 重试复制适配器并配置局部实例，原适配器始终保留基础配置，重试等级不再跨组串扰。

## Hugin 专用执行顺序

1. 使用原有 Hugin 对齐和验证，保留原有重试等级及阈值。
2. 从未经校色的对齐 RGB 计算焦点和来源统计，保留原有 Hugin 标签规则。
3. 在 Hugin 的 `quality` 蒙版中追加背景纹理来源稳定规则，减少纸面和阴影的块状切换；`gate` 蒙版已有这一步，不重复运行。随后修正外轮廓及白字相邻漆面的连续来源，使用边缘实际清晰度约束，保留真实底纹。
4. 在同材质内部校正各来源的低频色差，另写无损 `tone_inputs/*.tif`。不修改原照片或原始对齐 TIFF。
5. 原有 Enfuse 外部进程加载最终蒙版，渲染校正输入，再由原有编码器输出 JPEG。

默认自动模式在材质校色启用时使用一层 Enfuse，色差由材质模型处理。明确配置的多层融合仍保留；`EnfuseConfig.focus_blend_levels > 1` 时扩大焦点支持范围。没有将 Hugin 渲染器替换为产品 Quality 渲染器。

## 独立配置

- `RuntimeConfig.hugin_edge_ownership=True`：开启 Hugin 外沿与印字来源修复。
- `RuntimeConfig.hugin_surface_tone=True`：开启 Hugin 材质颜色修复。
- 两个开关独立于 `quality_printed_edge_guard` 和 `quality_surface_tone`。
- `hugin_focus_mask_mode=legacy` 保留旧路径，不启用上述修复。
- 外部自定义蒙版和曝光融合保留原有执行方式。

接入代码位于 `hugin/focus_repair.py`、`hugin/enfuse.py` 及 `HuginEnfuseBackend`；共用修复组件的实现没有修改。

## 验证范围

只验证本次实际 group12 的 0522，使用原来的 50 张选片、实际顺序和参考 DSC00557；新结果及中间证据写入独立目录。五组带手大动作的对齐失败按用户说明接受，不重跑或放宽阈值。

针对性测试和实图结果记录在 `diagnostics/hugin_quality_20261001/修复结果.md`，以该报告的最终状态为准。
