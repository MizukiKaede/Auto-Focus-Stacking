---
name: Focus Stack Studio
description: Daylight-first creative studio workbench and low-glare Darkroom UI for batch focus stacking.
colors:
  primary: "#2563EB"
  primary-hover: "#1D4ED8"
  primary-soft: "#D9E4F7"
  canvas-light: "#DDE1E7"
  surface-light: "#E8EBEF"
  rail-light: "#D4D9E2"
  mat-light: "#CFD4DD"
  ink-primary: "#181D26"
  ink-secondary: "#363E4E"
  ink-muted: "#555F71"
  border-light: "#C6CCD8"
  border-strong-light: "#B0B8C7"
  canvas-dark: "#101216"
  surface-dark: "#171A21"
  rail-dark: "#14171D"
  mat-dark: "#0C0E12"
  ink-dark-primary: "#F1F5F9"
  border-dark: "#252B36"
  success: "#059669"
  warning: "#D97706"
  danger: "#DC2626"
typography:
  brand:
    fontFamily: "Microsoft YaHei UI, Microsoft YaHei, Segoe UI, PingFang SC, Noto Sans CJK SC, sans-serif"
    fontSize: "15px"
    fontWeight: 700
  section:
    fontFamily: "Microsoft YaHei UI, Microsoft YaHei, Segoe UI, PingFang SC, Noto Sans CJK SC, sans-serif"
    fontSize: "12px"
    fontWeight: 700
    letterSpacing: "0.3px"
  body:
    fontFamily: "Microsoft YaHei UI, Microsoft YaHei, Segoe UI, PingFang SC, Noto Sans CJK SC, sans-serif"
    fontSize: "13px"
    fontWeight: 400
  mono:
    fontFamily: "Consolas, Cascadia Mono, Segoe UI Mono, monospace"
    fontSize: "12px"
rounded:
  sm: "4px"
  md: "6px"
  lg: "8px"
  xl: "10px"
  dropzone: "12px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "#FFFFFF"
    rounded: "7px"
    padding: "0 11px"
    height: "36px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.ink-secondary}"
    rounded: "{rounded.md}"
    padding: "0 9px"
    height: "28px"
  chip-active:
    backgroundColor: "{colors.primary}"
    textColor: "#FFFFFF"
    rounded: "{rounded.md}"
    padding: "0 10px"
    height: "27px"
  input-default:
    backgroundColor: "{colors.surface-light}"
    textColor: "{colors.ink-primary}"
    rounded: "{rounded.md}"
    padding: "0 9px"
    height: "30px"
---

## Overview

Focus Stack Studio (`景深合成工作台`) is a PySide6 desktop workbench designed for daytime office work (`Daylight Matte Studio`, default) with a one-click low-glare `Darkroom` (`暗房模式`) theme toggle. The visual system avoids glaring `#FFFFFF` panels altogether, pairing soft matte neutral grays (`#DDE1E7` / `#E8EBEF` / `#D4D9E2` in Light; `#101216` / `#171A21` / `#14171D` in Darkroom) with crisp `1px` hairlines and a restrained Cobalt accent (`#2563EB` / `#3B82F6`) so fused photographs remain the visual focal point.

## Colors

### Primary
- **Cobalt Accent** (`#2563EB` light / `#3B82F6` dark): Reserved for primary execution CTA (`开始处理`), active filter chips, active pipeline stage indicators, progress bar fill, focus rings, and drag-active borders.
- **Cobalt Soft** (`#D9E4F7` light / `#17253D` dark): Soft tint behind active pipeline steps and neutral status pills.

### Neutral
- **Daylight Matte Canvas / Darkroom Canvas** (`#DDE1E7` / `#101216`): Workspace background canvas framing cards and proofing surfaces without high-luminance glare.
- **Elevated Card Surface** (`#E8EBEF` / `#171A21`): Header bar, progress strip, results surface, and data table background.
- **Architectural Control Rail** (`#D4D9E2` / `#14171D`): Left sidebar housing directory inputs, grouping settings, and pinned action footer.
- **Photographic Preview Mat** (`#CFD4DD` / `#0C0E12`): Neutral matte frame behind stacked thumbnails and the 1:1 lightbox canvas so image highlights and shadows read without color cast.
- **Hairline Borders** (`#C6CCD8` subtle, `#B0B8C7` strong in Light; `#252B36` subtle, `#363F50` strong in Dark): Structural separation without drop shadows.
- **Semantic Status** (`#047857` / `#34D399` success, `#B45309` / `#FBBF24` warning/fallback, `#B91C1C` / `#F87171` error): Strictly tied to completion, Hugin fallback warnings, and merge/alignment failures.

**The Flat Neutrality Rule.** Keep containers neutral (`0` colored glows, `0` gradients); reserve saturated Cobalt strictly for the primary action and active state cues, and never overlay text directly on top of preview thumbnails.

## Typography

**Body & UI Font:** `"Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif`
**Label / Monospace Font:** `"Consolas", "Cascadia Mono", "Segoe UI Mono", monospace`

**Character:** Compact desktop tool typography prioritized for fast CJK/Latin scannability and aligned tabular readouts.

### Hierarchy
- **Brand Title** (`700`, `15px`): Top bar application identity (`Focus Stack Studio`).
- **Section Heading** (`700`, `12px`, `letter-spacing: 0.3px`): Sidebar section headers and panel headers (`目录与输入`, `合成与分组设置`, `合成结果`).
- **Body & Controls** (`400`–`500`, `13px`): Form labels, inputs, table cells, checkboxes, and action buttons.
- **Muted Metadata & Hints** (`400`, `11px`–`12px`): Folder photo count summaries, coverage captions, timestamps, and status bar hints.
- **Monospace Telemetry** (`600`, `12px` Consolas): Header stopwatch (`00:00:00`), KPI readouts, slice counts, and diagnostic logs.

## Layout

- **Minimum & Default Footprint:** Default `1180x780`, enforced minimum `960x640`.
- **Top Studio Header (`52px` fixed height):** Brand mark/title on the left; `28px` stopwatch badge, semantic status pill, and Light/Darkroom toggle on the right.
- **Architectural Two-Zone Split:**
  - **Left Control Rail (`328px` fixed width):** Vertically scrollable parameter stack (`目录与输入`, `合成与分组设置`, collapsible `高级与性能设置`) with horizontal scrolling disabled and `AdjustToMinimumContentsLengthWithIcon` on combo boxes; pinned `railFooter` holds full-width `开始处理` (`startButton`) and `停止` (`stopButton`).
  - **Right Proofing Stage:** Compact `ProgressPanel` at top (`ProgressStrip`, 5-step inline pipeline stepper, master + dual analysis/merge progress bars, collapsible `诊断日志` drawer) above `GroupListWidget` (`1fr` vertical stretch).
- **Results & Specimen Inspector Split:** Two-row non-overlapping toolbar (`合成结果` count + batch folder/CSV buttons; search box + segmented filter chips) over a horizontal `QSplitter` (`[450, 320]` initial ratio) pairing the 6-column results table with a scrollable `inspectorPane` (`250px`–`460px` wide) whose `previewStage` height is bounded to `168px` (and auto-hidden on failed groups with diagnostics) so action buttons, error banners, and coverage remain above the fold.

## Elevation & Depth

Surfaces are **flat by default**. Depth and hierarchy come from tonal stepping (`#D4D9E2` rail vs `#DDE1E7` canvas vs `#E8EBEF` surface in Light; `#101216` vs `#14171D` vs `#171A21` in Darkroom) paired with `1px` structural borders (`#C6CCD8` / `#252B36`). Interactive dialogs (`ImageLightboxDialog`) open in dedicated modal framing with neutral photographic mats.

## Shapes

- **Cards & Surfaces:** `8px`–`10px` radius (`10px` on `progressStrip` and `resultsSurface`; `8px` on `inspectorPane`, `QTableWidget`, and `previewStage`).
- **Inputs, Buttons & Checkboxes:** `6px` radius (`7px` on `startButton` / `stopButton`; `4px` on checkbox indicators).
- **Pills & Stepper Badges:** `10px` radius on `countPill`; `6px` on status badge and timer.

## Components

### Buttons
- **Primary (`#startButton`):** `36px` min-height, `7px` radius, Cobalt `#2563EB` (`#1D4ED8` hover, `#1E40AF` pressed), `#FFFFFF` bold label.
- **Stop (`#stopButton`):** `36px` min-height, `7px` radius, neutral button fill with semantic `#DC2626` label and subtle danger hover tint.
- **Ghost (`#ghostBtn` / `#themeToggleBtn`):** `28px` min-height, `6px` radius, transparent fill with `1px` border.
- **Disclosure (`#disclosureBtn`):** Left-aligned toggle with subtle border and transparent resting fill (`32px` min-height).

### Chips
- **Segmented Filter Chips (`.segBtn`):** `27px` min-height, `6px` radius, `0 10px` padding; inactive uses subsurface `#F9FAFC` / `#13161C` with `1px` border; active switches to solid Cobalt `#2563EB` (`#3B82F6` dark) with `#FFFFFF` label.

### Cards / Containers
- **Primary Panels (`#progressStrip`, `#resultsSurface`):** `10px` radius, `1px` subtle border, `bg_surface` fill.
- **Dropzone (`#emptyDropCard`):** `12px` radius, `1.5px dashed` border (`accent_soft` background and Cobalt dashed stroke when `dragActive="true"`).
- **Diagnostic Banners (`#diagBanner`, `#warnBanner`):** `6px` radius, `10px 8px` internal padding, soft semantic surface (`danger_soft` / `warning_soft`) with matching `1px` border and selectable wrapped text.

### Inputs / Fields
- **Text, SpinBox & ComboBox:** `30px` min-height, `6px` radius, `0 9px` padding, `1px` strong border shifting to `1.5px` Cobalt on focus, SVG arrow indicators cleanly positioned with zero browser/native overlap.

### Lightbox Proofing Viewer (`ImageLightboxDialog`)
- Dedicated inspection window with `28px` toolbar (`filename · dimensions · file size`, live zoom `%` badge, `适应窗口`, `1:1 像素`, `-`/`+`, `系统打开`, `定位文件`), `QGraphicsView` wheel zoom + drag pan + double-click Fit/1:1 toggle (`SmoothTransformation` when fitted, `FastTransformation` pixel grid at `>= 150%`).

## Do's and Don'ts

### Do:
- **Do** keep `ThemeTokens` in `focus_stack_app/ui/theme.py` as the single palette/stylesheet source of truth for both Light and Darkroom themes.
- **Do** keep all SVG icons monochrome (`viewBox="0 0 24 24"`) and tint them at runtime via `themed_icon` / `themed_pixmap` rather than mixing Unicode emoji with SVGs.
- **Do** call `app.setStyle("Fusion")` at most once on `QApplication` rather than recreating per-window `QStyle` instances when switching themes.
- **Do** keep `MainWindow(..., restore_settings=False)` as the default constructor behavior for isolated headless tests while `create_application` passes `restore_settings=True`.

### Don't:
- **Don't** place multi-element single-row toolbars inside the right workspace without checking the `960x640` minimum footprint.
- **Don't** draw text overlays inside the preview thumbnail frame (`previewStage`).
- **Don't** remove `table_selected_fg` from `QTableWidget::item:selected` without overriding `QPalette.HighlightedText`, or Light-mode selected rows will render white text on a `#CFDEF5` tint.
