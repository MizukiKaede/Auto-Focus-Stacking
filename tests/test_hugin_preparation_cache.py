from focus_stack_app.hugin import alignment_refinement
from focus_stack_app.hugin.enfuse import _prepare_alignment_refiner


def test_refiner_preparation_uses_peek_loader_then_restores_consumer(monkeypatch):
    calls = {"prepare": [], "consume": []}

    class TrackingRefiner:
        def __init__(self, loader, reference_index):
            self.loader = loader
            self.reference_index = reference_index
            self.reference = loader(reference_index)

        def prepare(self, frame_count):
            for index in range(frame_count):
                if index != self.reference_index:
                    self.loader(index)

        def load(self, index):
            return self.loader(index)

    monkeypatch.setattr(alignment_refinement, "HuginAlignmentRefiner", TrackingRefiner)

    def preparation_loader(index):
        calls["prepare"].append(index)
        return index

    def consuming_loader(index):
        calls["consume"].append(index)
        return index

    refiner = _prepare_alignment_refiner(
        consuming_loader, preparation_loader, reference_index=2, frame_count=4,
    )

    assert calls["prepare"] == [2, 0, 1, 3]
    assert calls["consume"] == []
    assert refiner.loader is consuming_loader
    assert refiner.load(0) == 0
    assert calls["consume"] == [0]


def test_refiner_keeps_existing_loader_behavior_without_peek_loader(monkeypatch):
    calls = []

    class TrackingRefiner:
        def __init__(self, loader, reference_index):
            self.loader = loader
            self.reference_index = reference_index
            loader(reference_index)

        def prepare(self, frame_count):
            for index in range(frame_count):
                if index != self.reference_index:
                    self.loader(index)

    monkeypatch.setattr(alignment_refinement, "HuginAlignmentRefiner", TrackingRefiner)

    def loader(index):
        calls.append(index)
        return index

    refiner = _prepare_alignment_refiner(loader, None, reference_index=0, frame_count=3)

    assert calls == [0, 1, 2]
    assert refiner.loader is loader
