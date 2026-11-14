# -*- mode: python ; coding: utf-8 -*-
"""Windows portable build with the shared multi-resolution layers icon."""
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

project = Path(SPECPATH).parent
runtime = Path(os.environ.get('AUTOSTACK_HUGIN_BIN', project / '.runtime-deps/hugin/bin'))
package = project / 'focus_stack_app'

a = Analysis(
    [str(project / 'packaging/portable_entry.py')],
    pathex=[str(project)],
    binaries=[
        (str(package / 'fusion/*.dll'), 'focus_stack_app/fusion'),
        (str(runtime / '*.dll'), '.runtime-deps/hugin/bin'),
        (str(runtime / '*.exe'), '.runtime-deps/hugin/bin'),
    ],
    datas=[
        (str(package / 'ui/icons'), 'focus_stack_app/ui/icons'),
        (str(package / 'fusion/native_runtime_licenses'), 'focus_stack_app/fusion/native_runtime_licenses'),
        (str(package / 'fusion/build_manifest.json'), 'focus_stack_app/fusion'),
    ],
    hiddenimports=['PySide6.QtSvg'] + collect_submodules('focus_stack_app'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(project / 'packaging/runtime_hook.py')],
    excludes=['pytest', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AutoStack-main',
    icon=str(package / 'ui/icons/layers.ico'),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AutoStack-main',
)
