"""Portable launch wrapper; the application source remains unmodified."""
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    check = next((arg.split('=', 1)[1] for arg in sys.argv[1:]
                  if arg.startswith('--portable-check=')), None)
    from focus_stack_app.app import create_application
    app, window = create_application([sys.argv[0]])
    window.show()
    if check:
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QPixmap

        def inspect():
            report = {'ok': False, 'python': sys.version,
                      'bundle': str(Path(sys._MEIPASS)),
                      'window_title': window.windowTitle(),
                      'window_visible': window.isVisible(),
                      'qt_platform': app.platformName()}
            try:
                import ctypes
                from focus_stack_app.fusion import fast_cpp, statistics_native
                directory = Path(sys._MEIPASS) / 'focus_stack_app' / 'fusion'
                fast = ctypes.CDLL(str(directory / 'fast_core.dll'))
                stats = ctypes.CDLL(str(directory / 'statistics_native.dll'))
                assert fast.fast_core_abi() == 3
                assert getattr(stats, 'probe_update')
                if hasattr(statistics_native, '_library'):
                    assert statistics_native._library.dll is not None
                report['native'] = {'fast_abi': 3, 'statistics': 'loaded',
                                    'fast_runtime': fast_cpp.runtime_info()}
                icon = Path(sys._MEIPASS) / 'focus_stack_app' / 'ui' / 'icons' / 'clock.svg'
                assert icon.is_file() and not QPixmap(str(icon)).isNull()
                app_icon = app.windowIcon()
                assert not app_icon.isNull() and not window.windowIcon().isNull()
                report['app_icon_sizes'] = [(size.width(), size.height())
                                            for size in app_icon.availableSizes()]
                assert (256, 256) in report['app_icon_sizes']
                from focus_stack_app.hugin.hugin_locator import HuginLocator
                tools = HuginLocator().find_installation(require=True)
                import subprocess
                report['tools'] = {}
                for name, path, option in (
                    ('align_image_stack', tools.align_image_stack, '--help'),
                    ('enfuse', tools.enfuse, '--version'),
                ):
                    assert Path(sys._MEIPASS) in path.parents
                    result = subprocess.run([str(path), option], capture_output=True,
                                            text=True, errors='replace', timeout=20,
                                            creationflags=subprocess.CREATE_NO_WINDOW)
                    output = result.stdout + result.stderr
                    assert name in output.lower(), (name, result.returncode, output)
                    assert result.returncode in (0, 1), (name, result.returncode, output)
                    report['tools'][name] = {'path': str(path),
                                            'exit_code': result.returncode,
                                            'output': output.splitlines()[:3]}
                assert window.isVisible() and app.platformName() == 'windows'
                screenshot = str(Path(check).with_suffix('.png'))
                assert window.grab().save(screenshot)
                report['ok'] = True
            except Exception:
                report['error'] = traceback.format_exc()
            Path(check).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            window.close()
            app.exit(0 if report['ok'] else 1)

        QTimer.singleShot(500, inspect)
    return app.exec()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        details = traceback.format_exc()
        check = next((arg.split('=', 1)[1] for arg in sys.argv[1:]
                      if arg.startswith('--portable-check=')), None)
        if check:
            Path(check).write_text(json.dumps({'ok': False, 'error': details}, indent=2), encoding='utf-8')
        else:
            directory = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'AutoStack' / Path(sys.executable).stem
            directory.mkdir(parents=True, exist_ok=True)
            log = directory / 'startup-error.log'
            log.write_text(details, encoding='utf-8')
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, f'程序启动失败，详细信息已保存到：\n{log}', 'AutoStack', 0x10)
        raise SystemExit(1)
