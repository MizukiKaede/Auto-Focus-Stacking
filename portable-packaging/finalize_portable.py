"""Attach documentation, source records, licenses and ZIP the checked builds."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

from build_portable import DEFAULT_OUTPUT, PROJECT, SUPPORT

ASSETS = SUPPORT / 'assets'
sys.path.insert(0, str(SUPPORT / 'tools'))
import pefile

PACKAGES = ('numpy', 'opencv-python', 'Pillow', 'psutil', 'PySide6', 'onnxruntime-directml',
            'PySide6_Essentials', 'PySide6_Addons', 'shiboken6', 'pyinstaller')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def native_inventory(source, portable):
    result = {}
    for original in sorted((source / 'focus_stack_app' / 'fusion').glob('*.dll')):
        bundled = portable / '_internal' / 'focus_stack_app' / 'fusion' / original.name
        assert bundled.is_file() and digest(original) == digest(bundled), original
        pe = pefile.PE(str(bundled), fast_load=True)
        assert pe.FILE_HEADER.Machine == 0x8664, bundled
        pe.close()
        result[original.name] = digest(bundled)
    # Confirm non-system dependencies of the distributed native kernels and
    # Hugin tools exist in the package; Microsoft runtime DLLs must be app-local.
    internal = portable / '_internal'
    files = {p.name.lower(): p for p in internal.rglob('*') if p.suffix.lower() in ('.dll', '.exe', '.pyd')}
    starts = list((internal / 'focus_stack_app' / 'fusion').glob('*.dll'))
    starts += list((internal / '.runtime-deps' / 'hugin' / 'bin').glob('*.exe'))
    pending = list(starts)
    seen = set()
    system = Path('C:/Windows/System32')
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        pe = pefile.PE(str(path))
        assert pe.FILE_HEADER.Machine == 0x8664, path
        for entry in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', []):
            name = entry.dll.decode().lower()
            if name.startswith(('api-ms-', 'ext-ms-')):
                continue
            if name in files:
                pending.append(files[name])
            else:
                assert not name.startswith(('msvcp', 'vcruntime', 'vcomp', 'libgomp', 'libwinpthread', 'libgcc')), (path, name)
                assert (system / name).is_file(), (path, name)
        pe.close()
    return result


def licenses(portable):
    target = portable / 'LICENSES'
    target.mkdir(exist_ok=True)
    shutil.copy2(Path(sys.base_prefix) / 'LICENSE.txt', target / 'Python-LICENSE.txt')
    for package in PACKAGES:
        distribution = metadata.distribution(package)
        folder = target / package
        folder.mkdir(exist_ok=True)
        (folder / 'METADATA.txt').write_text(distribution.read_text('METADATA') or '', encoding='utf-8')
        for item in distribution.files or ():
            path = Path(item)
            if 'licenses' in path.parts or path.name.lower().startswith(('license', 'copying', 'notice')):
                original = Path(distribution.locate_file(item))
                if original.is_file():
                    output = folder / path
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, output)
    for name in ('LGPL-3.0.txt', 'GPL-3.0.txt', 'MSVC-Redistributable-License.docx'):
        shutil.copy2(ASSETS / name, target / name)
    with zipfile.ZipFile(ASSETS / 'MSVC-Redistributable-License.docx') as archive:
        tree = ET.fromstring(archive.read('word/document.xml'))
        namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        text = '\n'.join(''.join(p.itertext()) for p in tree.findall('.//w:p', namespace))
        (target / 'MSVC-Redistributable-License.txt').write_text(text, encoding='utf-8')
    hugin = target / 'Hugin-Enfuse'
    hugin.mkdir(exist_ok=True)
    for original in (ASSETS / 'hugin-licenses').iterdir():
        if original.is_file():
            shutil.copy2(original, hugin / original.name.removeprefix('CM_FP_doc.'))
    (hugin / 'SOURCES.txt').write_text(
        'Hugin 2025.0.1 (9609d5726ad8), copyright Pablo d\'Angelo and contributors.\n'
        'Enfuse 4.3 (993fd4822e8d), copyright Andrew Mihal, Christoph Spiel and contributors.\n'
        'Both tools: GNU GPL version 2 or later. The bundled COPYING.txt contains GPL v2.\n'
        'Original Windows distribution: https://sourceforge.net/projects/hugin/files/hugin/hugin-2025.0/\n'
        'Hugin source: https://sourceforge.net/p/hugin/hugin/ci/9609d5726ad8/tree/\n'
        'Enblend/Enfuse source: https://sourceforge.net/p/enblend/code/ci/993fd4822e8d/tree/\n', encoding='utf-8')
    (target / 'THIRD-PARTY.txt').write_text(
        'The original package license and copyright notices are preserved in the subdirectories.\n'
        'Qt / PySide6 / Shiboken are distributed using the open-source license alternatives in their METADATA.\n'
        'Qt and PySide6 are dynamically linked; shared libraries are in _internal/PySide6 and _internal/shiboken6.\n'
        'Qt sources: https://download.qt.io/official_releases/qt/\n'
        'PySide6 / Shiboken sources: https://download.qt.io/official_releases/QtForPython/pyside6/\n'
        'PyInstaller bootloader: GPL with the distribution exception in its preserved license.\n'
        'Microsoft runtime version: 14.51.36247.0. Original, unmodified x64 redistributable DLLs.\n'
        'Microsoft runtime license: https://aka.ms/VCRedistLicense\n'
        'Native runtime licenses from each branch are preserved beside its fusion DLLs when supplied.\n', encoding='utf-8')


def finalize(branch, output):
    portable = output / 'portable' / f'AutoStack-{branch}'
    check = json.loads((output / f'{branch}-check.json').read_text(encoding='utf-8'))
    assert check['ok'], check
    commit = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip()
    native = native_inventory(PROJECT, portable)
    licenses(portable)
    upstream_licenses = SUPPORT / 'publication' / 'upstream-licenses'
    for path in upstream_licenses.iterdir():
        shutil.copy2(path, portable / 'LICENSES' / 'Hugin-Enfuse' / path.name)
    for name in ('README.md', 'THIRD_PARTY_NOTICES.md'):
        shutil.copy2(PROJECT / name, portable / name)
    subprocess.run(['git', '-C', str(PROJECT), 'archive', '--format=zip',
                    f'--output={portable / "SOURCE.zip"}', 'HEAD'], check=True)
    info = {
        'repository': 'https://github.com/MizukiKaede/Auto-Focus-Stacking',
        'branch': branch, 'commit': commit,
        'built_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'platform': 'Windows 10/11 x64', 'python': sys.version.split()[0],
        'packaging': 'PyInstaller onedir, windowed',
        'dependencies': {p: metadata.version(p) for p in PACKAGES},
        'native_sha256': native, 'source_algorithms': 'unmodified',
        'hugin': '2025.0.1.9609d5726ad8', 'enfuse': '4.3-993fd4822e8d',
        'msvc_runtime': '14.51.36247.0',
        'verification': {'window': True, 'svg_icon': True, 'native_libraries': True,
                         'bundled_hugin_tools': True, 'onnx_runtime': check['ai']},
    }
    (portable / 'BUILD-INFO.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    (portable / '使用说明.txt').write_text(
        f'AutoStack {branch} 分支便携版\n提交：{commit}\n\n'
        f'1. 将 ZIP 完整解压到一个文件夹。\n2. 双击 AutoStack-{branch}.exe 启动。\n'
        '3. 选择或拖入照片文件夹，再点击“开始处理”。默认输出到原照片文件夹内的“合成”目录。\n\n'
        '无需安装 Python、Hugin 或微软运行库；AI、高质量和实验模式所需依赖已附带。\n'
        '适用 Windows 10/11 64 位。请保留 _internal 文件夹，不要只复制 exe，也不要在 ZIP 内直接运行。\n'
        '两个分支应分别解压，避免相互覆盖。\n'
        '许可证位于 LICENSES；版本与依赖记录位于 BUILD-INFO.json；原始源码及打包适配位于 SOURCE.zip。\n'
        '已检查主界面、图标、ONNX 模型、原生库与内置工具加载。\n', encoding='utf-8-sig')
    archive_path = output / f'AutoStack-{branch}-win64.zip'
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(portable.rglob('*')):
            if path.is_file():
                archive.write(path, Path(portable.name) / path.relative_to(portable))
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    print(json.dumps({'branch': branch, 'zip': str(archive_path), 'bytes': archive_path.stat().st_size,
                      'sha256': digest(archive_path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('branch', nargs='?', default='main', choices=('main', 'dev'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    options = parser.parse_args()
    finalize(options.branch, options.output.resolve())
