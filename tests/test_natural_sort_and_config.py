from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from focus_stack_app.config import AppConfig, OutputConfig
from focus_stack_app.core.group_analyzer import GroupAnalyzer
from focus_stack_app.utils.natural_sort import natural_sorted


class NaturalSortAndConfigTests(unittest.TestCase):
    def test_natural_sort_numeric_chunks(self) -> None:
        self.assertEqual(natural_sorted(["DSC10.JPG", "DSC2.JPG", "DSC1.JPG"]), ["DSC1.JPG", "DSC2.JPG", "DSC10.JPG"])

    def test_config_roundtrip_and_safe_output_defaults(self) -> None:
        config = AppConfig()
        self.assertEqual(config.output.jpeg_quality, 100)
        self.assertFalse(config.output.overwrite)
        self.assertEqual(config.analysis.scene_similarity_threshold, 0.82)
        self.assertEqual(config.analysis.duplicate_focus_threshold, 0.995)
        self.assertEqual(config.analysis.minimum_stack_images, 2)
        self.assertEqual(config.analysis.minimum_stack_group_size, 4)
        self.assertEqual(config.runtime.max_hugin_workers, 3)
        self.assertFalse(config.runtime.preserve_cache)
        self.assertEqual(config.runtime.focus_analysis_workers, 0)
        self.assertEqual(config.analysis.minimum_stack_stability, 0.98)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config.save(path)
            loaded = AppConfig.load(path)
            self.assertEqual(loaded.to_mapping(), config.to_mapping())

    def test_output_validation(self) -> None:
        with self.assertRaises(ValueError):
            OutputConfig(jpeg_quality=101)

    def test_unstable_four_image_scene_skips_fusion_before_image_analysis(self) -> None:
        result = GroupAnalyzer(AppConfig()).analyze_group(
            {
                "id": 1,
                "images": ["one.JPG", "two.JPG", "three.JPG", "four.JPG"],
                "comparisons": [
                    {"low_frequency_similarity": 0.99},
                    {"low_frequency_similarity": 0.91},
                    {"low_frequency_similarity": 0.99},
                ],
            }
        )
        self.assertEqual(result["merge_status"], "NO_MERGE_UNSTABLE")
        self.assertFalse(result["needs_merge"])
        self.assertEqual(result["selected_count"], 0)


if __name__ == "__main__":
    unittest.main()

