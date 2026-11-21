"""Only adapt DLL search paths for the frozen portable distribution."""
import os
import sys
from pathlib import Path

_handles = []
if getattr(sys, 'frozen', False) and hasattr(os, 'add_dll_directory'):
    root = Path(sys._MEIPASS)
    for directory in (root, root / 'focus_stack_app' / 'fusion'):
        if directory.is_dir():
            _handles.append(os.add_dll_directory(str(directory)))
    # Keep handles alive for DLLs imported after the window has opened.
    sys._autostack_dll_handles = _handles
