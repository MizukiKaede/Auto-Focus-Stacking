import pytest

np = pytest.importorskip("numpy")

from focus_stack_app.core.coverage_selector import (
    CoverageConfig,
    order_selected_by_transition,
    select_focus_images,
)


def test_greedy_selection_covers_disjoint_regions():
    maps = []
    for start in (0, 20, 40):
        fmap = np.zeros((30, 60), np.float16)
        fmap[:, start:start + 20] = 1.0
        maps.append(fmap)
    result = select_focus_images(
        maps, [0.8, 0.9, 0.7],
        CoverageConfig(mode="balanced", min_coverage_gain=0.05, transition_reorder=False),
    )
    assert set(result.selected_indices) == {0, 1, 2}
    assert result.coverage >= 0.98
    assert all(result.gains[index] > 0.0 for index in result.selected_indices)


def test_transition_order_reduces_adjacent_map_distance():
    maps = []
    for start in (0, 40, 10):
        fmap = np.zeros((20, 60), np.float16)
        fmap[:, start:start + 20] = 1.0
        maps.append(fmap)
    ordered = order_selected_by_transition([0, 1, 2], maps)
    assert ordered[0] == 0
    assert ordered[1:] == [2, 1]


def test_broad_half_focused_frame_cannot_claim_full_coverage():
    # This models the real regression where a defocused frame retained about
    # half of the best local edge response almost everywhere.  At the former
    # 0.45 default it won the seed and falsely reported 100% coverage.
    broad_soft = np.full((24, 60), 0.50, np.float16)
    maps = [broad_soft]
    for start in (0, 20, 40):
        focused = np.zeros((24, 60), np.float16)
        focused[:, start:start + 20] = 1.0
        maps.append(focused)

    result = select_focus_images(
        maps,
        [1.0, 0.8, 0.8, 0.8],
        CoverageConfig(transition_reorder=False),
    )

    assert result.selected_indices == [1, 2, 3]
    assert result.coverage == pytest.approx(1.0)
    assert 0 not in result.selected_indices
