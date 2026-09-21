# Focus Stack Assistant

Memory-safe metadata, grouping, and focus-stack workflow helpers built with Python and PySide6.

## Requirements

- Python 3.11+
- OpenCV, NumPy, Pillow, psutil, and PySide6

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .[dev]
```

## Run

```powershell
python main.py
```

## Test

```powershell
pytest
```

Local photo sets, generated output, diagnostics, caches, and runtime dependencies are intentionally excluded from version control.
