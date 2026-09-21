from focus_stack_app.core.alignment_order import PairwiseRegistration, build_alignment_order, choose_preview_reference


def edge(a, b, confidence, *, valid=True, residual=1.0):
    return PairwiseRegistration(a, b, valid, confidence, confidence, confidence, residual, 0.0, 1.0, 0.0, "test")


def test_preview_reference_is_registration_graph_medoid():
    edges = [edge(0, 1, .9), edge(1, 2, .9), edge(0, 2, .1, valid=False)]
    reference, diagnostics = choose_preview_reference(3, edges)
    assert reference == 1
    assert diagnostics["edge_count"] == 3


def test_uncertain_alignment_order_falls_back_to_capture_order(monkeypatch):
    monkeypatch.setattr(
        "focus_stack_app.core.alignment_order.analyze_pairwise_registration",
        lambda images, config=None: [edge(0, 1, 0.0, valid=False)],
    )
    result = build_alignment_order([object(), object()], capture_order=[7, 3])
    assert result["alignment_order"] == [7, 3]
    assert result["alignment_order_fallback_used"] is True
    assert result["alignment_order_diagnostics"]["code"] == "ALIGNMENT_ORDER_UNCERTAIN"
