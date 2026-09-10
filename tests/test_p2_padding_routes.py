"""Expert-coordinate route comparison tests."""

from scripts.a1.audit_p2_padding_routes import route_delta


def test_order_is_not_expert_switch():
    a = {"indices": [0, 2], "weights": [0.7, 0.3], "logits": [2.0, 0.0, 1.0]}
    b = {"indices": [2, 0], "weights": [0.3, 0.7], "logits": [2.0, 0.0, 1.0]}
    assert route_delta(a, b) == {"set_changed": False, "weight_l1": 0.0, "logit_l1_mean": 0.0}


def test_expert_switch_and_weight_change():
    a = {"indices": [0, 2], "weights": [0.5, 0.5], "logits": [1.0, 0.0, 1.0]}
    b = {"indices": [0, 1], "weights": [0.5, 0.5], "logits": [1.0, 1.0, 0.0]}
    delta = route_delta(a, b)
    assert delta["set_changed"]
    assert delta["weight_l1"] == 1.0
