"""Build the two Windows x64 libraries without overwriting shipped binaries."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'focus_stack_app' / 'fusion'
RUNTIMES = {'libgomp-1.dll', 'libwinpthread-1.dll', 'libstdc++-6.dll', 'libdl.dll',
            'libgcc_s_seh-1.dll', 'libgcc_s_dw2-1.dll', 'libgcc_s_sjlj-1.dll'}


def run(args, **kwargs):
    result = subprocess.run(args, check=True, text=True, capture_output=True, **kwargs)
    return result.stdout


def msvc_environment():
    if shutil.which('cl.exe'):
        return dict(os.environ)
    installer = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / 'Microsoft Visual Studio' / 'Installer' / 'vswhere.exe'
    if not installer.is_file():
        raise RuntimeError('MSVC not found. Run in an x64 Native Tools prompt or install the C++ build tools.')
    location = run([str(installer), '-latest', '-products', '*', '-requires',
                    'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath']).strip()
    vcvars = Path(location) / 'VC' / 'Auxiliary' / 'Build' / 'vcvars64.bat'
    if not location or not vcvars.is_file():
        raise RuntimeError('MSVC x64 toolchain not found')
    # Only the discovered local setup script enters cmd; compiler argv remains structured.
    output = run(['cmd.exe','/d','/s','/c', f'call "{vcvars}" >nul && set'])
    return dict(line.split('=',1) for line in output.splitlines() if '=' in line and not line.startswith('='))


def gcc_runtime_files(compiler, dlls):
    objdump = compiler.with_name('objdump.exe')
    if not objdump.is_file():
        raise RuntimeError('MinGW objdump.exe is required to collect DLL runtime dependencies')
    pending = list(dlls)
    found = {}
    while pending:
        binary = pending.pop()
        names = re.findall(r'DLL Name:\s*(\S+)', run([str(objdump), '-p', str(binary)]))
        for name in names:
            key = name.lower()
            if key in found or key not in RUNTIMES:
                continue
            value = run([str(compiler), '-print-file-name='+name]).strip()
            path = Path(value)
            if not path.is_file():
                path = compiler.parent / name
            if not path.is_file():
                raise RuntimeError(f'Compiler runtime dependency missing: {name}')
            found[key] = path.resolve()
            pending.append(path)
    return found


def machine(path):
    with path.open('rb') as file:
        file.seek(0x3c)
        offset = struct.unpack('<I', file.read(4))[0]
        file.seek(offset)
        if file.read(4) != b'PE\0\0':
            raise RuntimeError(f'Invalid PE library: {path}')
        return struct.unpack('<H', file.read(2))[0]


def publish(build, destination):
    """Roll back publication errors, including a destination DLL held open by Windows."""
    artifacts = [p for p in build.iterdir() if p.suffix == '.dll' or p.name == 'build_manifest.json']
    artifacts += list((build/'native_runtime_licenses').glob('*'))
    previous = build/'previous'
    previous.mkdir()
    saved, copied = {}, []
    for source in artifacts:
        target = destination/source.relative_to(build)
        if target.is_file():
            backup = previous/source.relative_to(build)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            saved[target] = backup
    destination.mkdir(parents=True, exist_ok=True)
    # Finish every copy before changing shipped files. Each replacement is atomic,
    # so a failed copy or Windows file lock cannot leave a truncated DLL behind.
    with tempfile.TemporaryDirectory(prefix='.native-publish-', dir=destination) as temporary:
        staging = Path(temporary)
        for source in artifacts:
            staged = staging/source.relative_to(build)
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
        try:
            for source in artifacts:
                relative = source.relative_to(build)
                target = destination/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging/relative, target)
                copied.append(target)
        except OSError:
            for target in reversed(copied):
                if target in saved:
                    restored = staging/target.relative_to(destination)
                    shutil.copy2(saved[target], restored)
                    os.replace(restored, target)
                else:
                    target.unlink()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', choices=('auto','mingw','msvc'), default='auto')
    parser.add_argument('--compiler-path', type=Path, help='Explicit g++.exe; MSVC uses its x64 developer environment')
    parser.add_argument('--openmp', choices=('on','off'), default='on')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'build'/'native')
    args = parser.parse_args()
    if os.name != 'nt' or struct.calcsize('P') != 8:
        parser.error('This build supports Windows x64 with a 64-bit Python')
    gcc = args.compiler_path or shutil.which('g++.exe')
    choice = ('mingw' if gcc else 'msvc') if args.compiler == 'auto' else args.compiler
    if choice == 'msvc' and args.compiler_path:
        parser.error('--compiler-path is for MinGW; use the MSVC developer environment')
    compiler = Path(gcc).resolve() if choice == 'mingw' and gcc else None
    if choice == 'mingw' and (compiler is None or not compiler.is_file()):
        parser.error('MinGW g++.exe not found; supply --compiler-path')
    destination = args.output_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    env = msvc_environment() if choice == 'msvc' else dict(os.environ)
    commands = []
    with tempfile.TemporaryDirectory(prefix='native-build-', dir=destination.parent) as temporary:
        build = Path(temporary)
        libraries = []
        for stem in ('fast_core','statistics_native'):
            target = build / (stem+'.dll')
            source = SOURCE / (stem+'.cpp')
            if choice == 'mingw':
                command = [str(compiler), '-std=c++17','-O2','-fno-fast-math','-ffp-contract=off',
                           '-shared','-static-libgcc','-static-libstdc++']
                if args.openmp == 'on': command.append('-fopenmp')
                command += [str(source), '-o', str(target)]
            else:
                command = ['cl.exe','/nologo','/std:c++17','/O2','/EHsc','/fp:strict','/MT','/LD']
                if args.openmp == 'on': command.append('/openmp')
                command += [str(source), '/Fe:'+str(target), '/Fo:'+str(build/(stem+'.obj'))]
            commands.append(command)
            run(command, cwd=build, env=env)
            if machine(target) != 0x8664:
                raise RuntimeError('Compiler produced a non-x64 library')
            libraries.append(target)
        dependencies = gcc_runtime_files(compiler, libraries) if choice == 'mingw' else {}
        if choice == 'msvc' and args.openmp == 'on':
            # Locate the redistributable from this toolchain, never from arbitrary PATH DLLs.
            redist = env.get('VCToolsRedistDir') or env.get('VCTOOLSREDISTDIR')
            candidates = list(Path(redist).glob('x64/Microsoft.VC*.OpenMP/vcomp140.dll')) if redist else []
            if not candidates:
                raise RuntimeError('MSVC x64 OpenMP redistributable vcomp140.dll not found')
            dependencies['vcomp140.dll'] = candidates[-1]
        for name,path in dependencies.items(): shutil.copy2(path, build/name)
        # Smoke-load in a short-lived process so no library is held open on publication.
        probe = "import ctypes,os,sys; d=os.add_dll_directory(sys.argv[1]); " \
                "a=ctypes.CDLL(os.path.join(sys.argv[1],'fast_core.dll')); " \
                "b=ctypes.CDLL(os.path.join(sys.argv[1],'statistics_native.dll')); " \
                "assert a.fast_core_abi()==3 and b.statistics_core_abi()==1"
        import sys
        run([sys.executable,'-c',probe,str(build)])
        version = (run([str(compiler),'--version']).splitlines()[0] if compiler else env.get('VCToolsVersion', 'MSVC'))
        metadata = {'compiler': choice, 'compiler_version': version, 'commands': commands, 'openmp': args.openmp,
                    'architecture': 'x64', 'dependencies': {k:str(v) for k,v in dependencies.items()},
                    'output': str(destination)}
        (build/'build_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        licenses = build / 'native_runtime_licenses'
        shutil.copytree(SOURCE/'native_runtime_licenses', licenses)
        (licenses/'NOTICE.txt').write_text(
            'GCC runtimes: GNU GPL with GCC Runtime Library Exception; '
            'https://gcc.gnu.org/onlinedocs/libgomp/Copying.html\n'
            'MinGW-w64 runtime: https://github.com/mingw-w64/mingw-w64/blob/master/COPYING\n'
            'dlfcn-win32 (libdl): https://github.com/dlfcn-win32/dlfcn-win32/blob/master/COPYING\n'
            'MSVC runtime: redistribute only under the Visual Studio license terms.\n', encoding='utf-8')
        publish(build, destination)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.stdout or '', error.stderr or '')
        raise SystemExit(f'Native build failed ({error.returncode}); existing DLLs were not replaced')
