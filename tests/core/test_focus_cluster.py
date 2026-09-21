import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from focus_stack_app.core.focus_cluster import (
    FocusClusterConfig,
    deduplicate_focus_maps,
)


def _maps():
    base = np.zeros((40, 60), np.float32)
    base[:, :20] = 0.95
    near = base.copy()
    near[:, :20] = 0.88
    different = np.zeros_like(base)
    different[:, 40:] = 0.95
    return [base.astype(np.float16), near.astype(np.float16), different.astype(np.float16)]


def test_duplicate_focus_keeps_highest_quality():
    result = deduplicate_focus_maps(
        _maps(), [0.60, 0.95, 0.70],
        FocusClusterConfig(similarity_threshold=0.85, normalized_difference_threshold=0.20),
    )
    assert 1 in result.representative_indices
    assert 0 in result.rejected_indices
    assert 2 in result.representative_indices
    assert result.reasons[0].startswith("redundant focus")

