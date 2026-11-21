# Product

<!-- impeccable:product-schema 1 -->

## Platform

desktop (Windows 10/11 PySide6 / Qt6; recorded as desktop application outside mobile/web)

## Users

Designers and commercial / studio photographers working at bright office workstations who shoot large batches of focus-bracketed product photos (often out of focus order) and need reliable, batch-automated focus stacking and visual inspection.

## Product Purpose

Focus Stack Studio (Auto-stack) scans a photo folder, automatically groups shots belonging to the same subject/scene, evaluates sharpness coverage across frames to select only the needed slices, aligns and fuses each group into a crisp composite (JPG or TIFF), and optionally archives source originals safely.

Success means the designer can drop a folder in, immediately see what was found, start a batch without tuning cryptic knobs, monitor real progress and diagnostics without wondering if the app froze, and inspect crisp composite results and slice details cleanly.

## Positioning

Unlike single-stack interactive GUI tools or CLI scripts, Auto-stack combines non-sequential scene grouping, bounded-memory multi-worker execution, crash-safe SQLite/disk recovery, and three selectable fusion engines (`quality`, `fast` C++ core, and experimental `hugin_enfuse`) in one self-contained desktop workflow.

## Operating Context

- Primary usage environment: office desk lighting during regular work hours (light-first ergonomic surface with an optional low-glare darkroom mode for inspecting dark-subject photos).
- Layout preference: narrow left control rail + wide right workspace focused on results & preview, with compact inline run progress that expands during active processing without crowding the resting workspace.
- Priority functional scope: enhanced live run status & diagnostics (clear phase indicators, progress, expandable/copyable logs & diagnostics, completion alert, quick output/manifest opening) plus completing essential desktop controls and inspection details without clutter.

## Capabilities and Constraints

- Tech stack: Python >= 3.11, PySide6 >= 6.6 (`focus_stack_app/ui/`), packaged for Windows via PyInstaller (`portable-packaging/`).
- Supported inputs/outputs: scans `.jpg` / `.jpeg` (with optional `recursive` and `include_hidden` scanner flags and optional `stack_groups.json` manual overrides); outputs 100% quality JPG or lossless TIFF plus `stack_manifest.csv` and `.stack_cache/logs/application.log`.
- Three fusion backends: `quality` (default full-resolution memory-managed blend), `fast` (C++ core with proxy-map selection), and `hugin_enfuse` (external `align_image_stack` + `enfuse`, optional custom path overrides `hugin_bin`).
- Core parameters: `source_dir`, `output_dir`, `archive_dir`, `archive_enabled`, `archive_mode` (`move`/`copy`/`hardlink`), `minimum_stack_group_size` (2..1000, default 4), `grouping_pause_seconds` (1..3600, default 20), `output_format`, `fusion_backend`, `merge_workers` (1..6, default 3), `focus_analysis_workers` (0..10, default 0=auto), `parallel` (default True), `preserve_cache` (default False), `recursive`, `include_hidden`, `hugin_bin`.
- Existing automated test contracts on `MainWindow`, `ProgressPanel`, `ProgressSnapshot`, and `GroupListWidget` must remain compatible (`tests/test_dark_workspace_ui.py`, `tests/test_grouping_pause_ui.py`, `tests/test_performance_diagnostics.py`, `tests/test_pipeline_app.py`, `tests/test_pipeline_recovery.py`).

## Brand Commitments

- Name: Focus Stack Studio / 景深合成工作台
- Language: Simplified Chinese primary UI copy with clean technical clarity.
- SVG icon set located in `focus_stack_app/ui/icons/` (and packaged via `pyproject.toml`).

## Evidence on Hand

- `README.md`, `focus_stack_app/pipeline/application_controller.py`, `focus_stack_app/pipeline/merge_worker.py`, `focus_stack_app/pipeline/events.py`, `focus_stack_app/ui/*`.
- Earlier UI screenshots document neon dark cards, overlapping inspector text, a cramped preview, and truncated table columns. [Historical screenshots](https://github.com/MizukiKaede/Auto-Focus-Stacking/tree/bab997bcc05922abd86d0ac06efc9dd5b3ec843c/docs/images)

## Product Principles

1. Clarity in office daylight: crisp neutral surfaces, restrained typography scale, and zero decorative chrome so product photography stands out accurately.
2. Quiet at rest, informative in motion: keep the workspace calm and spacious before a batch starts; expand live progress, elapsed timing, and diagnostics cleanly when running.
3. Truthful inspection: show real selected frames, coverage, backend fallbacks, errors/skip reasons, and high-resolution preview without truncation or overlap.
4. Effortless defaults with accessible depth: common batch settings stay scannable on top; recursive scan, Hugin paths, cache, and concurrency stay neatly tucked into progressive disclosures.
