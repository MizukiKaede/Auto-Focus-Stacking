"""Build a Windows portable distribution from the current checkout."""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SUPPORT = PROJECT / 'build' / 'packaging'
DEFAULT_OUTPUT = PROJECT / 'build' / 'releases' / 'current'


def prepare_runtime(output):
    runtime = output / 'hugin-bin'
    runtime.mkdir(parents=True, exist_ok=True)
    hugin = PROJECT / '.runtime-deps' / 'hugin' / 'bin'
    for name in ('align_image_stack.exe', 'enfuse.exe'):
        if not (hugin / name).is_file():
            raise FileNotFoundError(hugin / name)
    for pattern in ('*.dll', '*.exe'):
        for source in hugin.glob(pattern):
            shutil.copy2(source, runtime / source.name)
    for source in (SUPPORT / 'assets' / 'msvc-raw').glob('*.dll_amd64'):
        shutil.copy2(source, runtime / source.name.removesuffix('_amd64'))
    return runtime


def build(branch, output):
    runtime = prepare_runtime(output)
    package = PROJECT / 'focus_stack_app'
    python = Path(sys.executable)
    commit = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip()
    env = dict(os.environ)
    env['PYTHONPATH'] = str(SUPPORT / 'tools')
    env['PYINSTALLER_CONFIG_DIR'] = str(output / 'cache')
    # Do not let unrelated desktop tools supply same-named runtime DLLs.
    env['PATH'] = os.pathsep.join([str(python.parent),
                                  str(Path(os.environ['SystemRoot']) / 'System32'),
                                  os.environ['SystemRoot']])
    args = [str(python), '-m', 'PyInstaller', '--noconfirm', '--onedir', '--windowed',
            '--name', f'AutoStack-{branch}', '--paths', str(PROJECT),
            '--distpath', str(output / 'portable'),
            '--workpath', str(output / 'work'),
            '--specpath', str(output / 'specs'),
            '--icon', str(package / 'ui' / 'icons' / 'layers.ico'),
            '--runtime-hook', str(ROOT / 'runtime_hook.py'),
            '--hidden-import', 'PySide6.QtSvg',
            '--collect-binaries', 'onnxruntime',
            '--collect-submodules', 'focus_stack_app',
            '--add-data', f'{package / "ui" / "icons"}:focus_stack_app/ui/icons',
            '--add-binary', f'{package / "fusion" / "*.dll"}:focus_stack_app/fusion',
            '--add-binary', f'{runtime / "*.dll"}:.runtime-deps/hugin/bin',
            '--add-binary', f'{runtime / "*.exe"}:.runtime-deps/hugin/bin',
            '--exclude-module', 'pytest', '--exclude-module', 'tkinter',
            str(ROOT / 'portable_entry.py')]
    for name in ('native_runtime_licenses', 'build_manifest.json', 'weights'):
        path = package / 'fusion' / name
        if path.exists():
            args[-1:-1] = ['--add-data', f'{path}:focus_stack_app/fusion/{name}' if path.is_dir()
                          else f'{path}:focus_stack_app/fusion']
    notices = PROJECT / 'THIRD_PARTY_NOTICES.md'
    if notices.is_file():
        args[-1:-1] = ['--add-data', f'{notices}:.']
    log = output / f'build-{branch}.log'
    print(f'Building {branch} @ {commit}; log: {log}', flush=True)
    with log.open('w', encoding='utf-8') as handle:
        subprocess.run(args, env=env, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    portable = output / 'portable' / f'AutoStack-{branch}' / '_internal'
    # This Qt build imports Windows' unversioned ICU API, not Poppler's ICU 78.
    # Leave the OS DLL to Windows rather than shipping a same-named substitute.
    for name in ('icuuc.dll', 'icudt78.dll'):
        unwanted = portable / name
        if unwanted.is_file():
            unwanted.unlink()
    print(f'Built {branch}', flush=True)
    return commit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('branch', nargs='?', default='main', choices=('main', 'dev'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    options = parser.parse_args()
    build(options.branch, options.output.resolve())
