"""Regressions for real Hugin option semantics and lossless JPEG export."""
from pathlib import Path
import tempfile
import unittest

from focus_stack_app.hugin.align import AlignConfig, AlignImageStack, AlignmentError
from focus_stack_app.hugin.enfuse import Enfuser
from focus_stack_app.hugin.process import CommandResult


class ImageQualityTests(unittest.TestCase):
    def test_focus_breathing_and_centre_use_correct_options(self):
        command = AlignImageStack('align.exe', config=AlignConfig(optimize_centre=True)).build_command(['a.jpg', 'b.jpg'], 'aligned_')
        self.assertEqual(command.count('-m'), 1)
        self.assertIn('-i', command)
        self.assertIn('--use-given-order', command)
        self.assertNotIn('--align-to-first', command)
        self.assertNotIn('-t', command)
        self.assertNotIn('-x', command)
        command = AlignImageStack('align.exe', config=AlignConfig(optimize_field_of_view=False, optimize_scale=False)).build_command(['a.jpg', 'b.jpg'], 'aligned_')
        self.assertNotIn('-m', command)
        self.assertIn('--use-given-order', command)

    def test_partial_alignment_cannot_be_published_as_stack(self):
        class Runner:
            def run(self, command, **kwargs):
                prefix = command[command.index('-a') + 1]
                Path(prefix + '0000.tif').write_bytes(b'partial')
                return CommandResult(tuple(command), 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / 'a.jpg', root / 'b.jpg']
            for path in paths:
                path.touch()
            with self.assertRaisesRegex(AlignmentError, 'incomplete fusion'):
                AlignImageStack('align.exe', runner=Runner()).align(paths, work_dir=root)

    def test_fusion_uses_lossless_intermediate_and_jpeg_444(self):
        try:
            from PIL import Image, JpegImagePlugin
        except ImportError:
            self.skipTest('Pillow unavailable')
        class Runner:
            command = None
            def run(self, command, **kwargs):
                self.command = command
                target = Path(command[command.index('-o') + 1])
                Image.new('RGB', (32, 24), (12, 140, 190)).save(target, icc_profile=b'test-profile')
                return CommandResult(tuple(command), 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.tif'
            source.touch()
            runner = Runner()
            output = root / 'result.jpg'
            Enfuser('enfuse.exe', runner=runner).fuse([source], output, work_dir=root / 'work')
            self.assertIn('--hard-mask', runner.command)
            self.assertEqual(Path(runner.command[runner.command.index('-o') + 1]).suffix, '.tif')
            with Image.open(output) as result:
                self.assertEqual(result.size, (32, 24))
                self.assertEqual(JpegImagePlugin.get_sampling(result), 0)
                self.assertEqual(result.info['icc_profile'], b'test-profile')


if __name__ == '__main__':
    unittest.main()
