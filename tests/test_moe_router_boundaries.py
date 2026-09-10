"""Boundary and error-handling tests for MoE routers.

Tests:
  - 3-D / 5-D input rejection (must be 4-D NCHW)
  - Channel mismatch detection
  - NaN/Inf input detection
  - NaN/Inf logits detection in BaseRouter._process_logits
  - top_k clamping (k > num_experts, k=0)
  - Exception hierarchy: MoERouterError/ShapeMismatchError inherit YOLOMasterError
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.moe import base as moe_base
from ultralytics.nn.modules.moe.loss import MoELoss
from ultralytics.nn.modules.moe.modules import OptimizedMOEImproved
from ultralytics.nn.modules.moe.routers import (
    BaseRouter,
    UltraEfficientRouter,
    EfficientSpatialRouter,
    AdaptiveRoutingLayer,
    DynamicRoutingLayer,
    LocalRoutingLayer,
    _validate_router_input,
)
from ultralytics.utils.errors import MoERouterError, ShapeMismatchError, YOLOMasterError


# =============================================================================
# Fixtures
# =============================================================================

IN_CHANNELS = 64
NUM_EXPERTS = 4
TOP_K = 2


@pytest.fixture
def ultra_router():
    return UltraEfficientRouter(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K)


@pytest.fixture
def spatial_router():
    return EfficientSpatialRouter(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K)


@pytest.fixture
def adaptive_router():
    return AdaptiveRoutingLayer(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K)


@pytest.fixture
def local_router():
    return LocalRoutingLayer(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K)


def _valid_input(batch=2, channels=IN_CHANNELS, h=16, w=16):
    return torch.randn(batch, channels, h, w)


def test_p1_private_router_noise_is_repeatable_and_does_not_advance_global_rng():
    router = AdaptiveRoutingLayer(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K, noise_std=0.05)
    logits = torch.zeros(8, NUM_EXPERTS)
    global_state = torch.random.get_rng_state().clone()

    router.configure_p1_private_noise(260829)
    first = router._sample_p1_private_noise(logits)
    assert torch.equal(global_state, torch.random.get_rng_state())

    router.configure_p1_private_noise(260829)
    second = router._sample_p1_private_noise(logits)
    router.configure_p1_private_noise(260830)
    third = router._sample_p1_private_noise(logits)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_p1_private_router_noise_keeps_hard_top2_and_is_disabled_in_eval():
    router = AdaptiveRoutingLayer(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K, noise_std=0.05)
    logits = torch.zeros(8, NUM_EXPERTS)
    router.configure_p1_private_noise(260829)
    weights, indices, _ = router._process_logits(logits, router.noise_std, True)
    assert weights.shape == indices.shape == (8, TOP_K)
    assert router.p1_noise_step == 1

    eval_weights, eval_indices, _ = router._process_logits(logits, router.noise_std, False)
    assert eval_weights.shape == eval_indices.shape == (8, TOP_K)
    assert router.p1_noise_step == 1


def test_non_p1_router_keeps_historical_noisy_loss_view():
    router = BaseRouter(num_experts=NUM_EXPERTS, top_k=TOP_K)
    assert router.p1_balance_on_clean_routes is False
    logits = torch.tensor([[0.04, 0.03, 0.00, -0.01], [0.04, 0.03, 0.00, -0.01]])
    router.configure_p1_private_noise(260831)

    _, dispatch_indices, loss_info = router._process_logits(logits, noise_std=0.05, training=True)

    assert "dispatch_router_logits" not in loss_info
    assert "dispatch_router_probs" not in loss_info
    assert "dispatch_topk_indices" not in loss_info
    assert torch.equal(loss_info["topk_indices"], dispatch_indices)
    assert not torch.equal(loss_info["router_logits"], logits)
    assert torch.allclose(loss_info["router_probs"], F.softmax(loss_info["router_logits"], dim=1))


def test_p1_clean_aux_view_does_not_change_noisy_hard_top2_dispatch_or_global_rng():
    router = BaseRouter(num_experts=NUM_EXPERTS, top_k=TOP_K)
    router.p1_balance_on_clean_routes = True
    router.configure_p1_private_noise(260831)
    logits = torch.tensor(
        [[0.04, 0.03, 0.00, -0.01], [0.04, 0.03, 0.00, -0.01]],
        requires_grad=True,
    )
    global_state = torch.random.get_rng_state().clone()

    dispatch_weights, dispatch_indices, loss_info = router._process_logits(
        logits, noise_std=0.05, training=True
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(260831)
    expected_noise = torch.randn(logits.shape, generator=generator, dtype=torch.float32)
    expected_dispatch_logits = logits + expected_noise * 0.05
    expected_dispatch_probs = F.softmax(expected_dispatch_logits, dim=1)
    expected_dispatch_values, expected_dispatch_indices = torch.topk(expected_dispatch_probs, TOP_K, dim=1)
    expected_dispatch_weights = expected_dispatch_values / expected_dispatch_values.sum(dim=1, keepdim=True)
    expected_clean_probs = F.softmax(logits, dim=1)
    expected_clean_indices = torch.topk(expected_clean_probs, TOP_K, dim=1).indices

    assert torch.equal(global_state, torch.random.get_rng_state())
    assert router.p1_noise_step == 1
    assert torch.allclose(dispatch_weights, expected_dispatch_weights)
    assert torch.equal(dispatch_indices, expected_dispatch_indices)
    assert not torch.equal(dispatch_indices, expected_clean_indices)
    assert loss_info["router_logits"] is logits
    assert torch.allclose(loss_info["router_probs"], expected_clean_probs)
    assert torch.equal(loss_info["topk_indices"], expected_clean_indices)
    assert torch.allclose(loss_info["dispatch_router_logits"], expected_dispatch_logits)
    assert torch.allclose(loss_info["dispatch_router_probs"], expected_dispatch_probs)
    assert torch.equal(loss_info["dispatch_topk_indices"], expected_dispatch_indices)


def test_p1_clean_aux_view_is_seed_invariant_while_private_dispatch_can_differ():
    logits = torch.tensor([[0.04, 0.03, 0.00, -0.01], [0.04, 0.03, 0.00, -0.01]])
    loss_views = []
    dispatch_indices = []
    for seed in (260829, 260831):
        router = BaseRouter(num_experts=NUM_EXPERTS, top_k=TOP_K)
        router.p1_balance_on_clean_routes = True
        router.configure_p1_private_noise(seed)
        _, indices, loss_info = router._process_logits(logits, noise_std=0.05, training=True)
        loss_views.append(loss_info)
        dispatch_indices.append(indices)

    assert torch.equal(loss_views[0]["router_logits"], loss_views[1]["router_logits"])
    assert torch.equal(loss_views[0]["router_probs"], loss_views[1]["router_probs"])
    assert torch.equal(loss_views[0]["topk_indices"], loss_views[1]["topk_indices"])
    assert not torch.equal(dispatch_indices[0], dispatch_indices[1])


def test_p1_clean_and_dispatch_views_are_exact_when_training_noise_is_zero():
    router = BaseRouter(num_experts=NUM_EXPERTS, top_k=TOP_K)
    router.p1_balance_on_clean_routes = True
    logits = torch.tensor([[0.04, 0.03, 0.00, -0.01], [0.02, 0.01, -0.01, -0.02]])

    dispatch_weights, dispatch_indices, loss_info = router._process_logits(
        logits, noise_std=0.0, training=True
    )

    assert dispatch_weights.shape == dispatch_indices.shape == (2, TOP_K)
    assert torch.equal(loss_info["router_logits"], loss_info["dispatch_router_logits"])
    assert torch.equal(loss_info["router_probs"], loss_info["dispatch_router_probs"])
    assert torch.equal(loss_info["topk_indices"], loss_info["dispatch_topk_indices"])
    assert torch.equal(dispatch_indices, loss_info["dispatch_topk_indices"])


def test_p1_clean_aux_view_rejects_capacity_routing_semantics():
    router = BaseRouter(num_experts=NUM_EXPERTS, top_k=TOP_K, capacity_factor=1.0)
    router.p1_balance_on_clean_routes = True

    with pytest.raises(MoERouterError, match="incompatible with capacity_factor"):
        router._process_logits(torch.zeros(4, NUM_EXPERTS), noise_std=0.0, training=True)


def test_p1_clean_aux_view_drives_moe_loss_and_router_gradient():
    router = BaseRouter(num_experts=NUM_EXPERTS, top_k=TOP_K)
    router.p1_balance_on_clean_routes = True
    router.configure_p1_private_noise(260831)
    logits = torch.tensor(
        [[0.04, 0.03, 0.00, -0.01], [0.02, 0.01, -0.01, -0.02]],
        requires_grad=True,
    )
    _, _, loss_info = router._process_logits(logits, noise_std=0.05, training=True)
    loss_fn = MoELoss(
        balance_loss_coeff=1.0,
        z_loss_coeff=0.1,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
    )

    actual = loss_fn(
        loss_info["router_probs"],
        loss_info["router_logits"],
        loss_info["topk_indices"],
    )
    clean_probs = F.softmax(logits, dim=1)
    clean_indices = torch.topk(clean_probs, TOP_K, dim=1).indices
    expected = loss_fn(clean_probs, logits, clean_indices)

    assert torch.equal(actual, expected)
    assert torch.isfinite(actual)
    assert actual.detach().item() > 0
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad).item() > 0


def test_p1_clean_aux_opt_in_does_not_change_eval_routing():
    logits = torch.tensor([[0.04, 0.03, 0.00, -0.01], [0.02, 0.01, -0.01, -0.02]])
    historical = BaseRouter(num_experts=NUM_EXPERTS, top_k=TOP_K)
    clean_aux = BaseRouter(num_experts=NUM_EXPERTS, top_k=TOP_K)
    clean_aux.p1_balance_on_clean_routes = True

    expected_weights, expected_indices, expected_info = historical._process_logits(
        logits, noise_std=0.05, training=False
    )
    actual_weights, actual_indices, actual_info = clean_aux._process_logits(
        logits, noise_std=0.05, training=False
    )

    assert expected_info == actual_info == {}
    assert torch.equal(actual_weights, expected_weights)
    assert torch.equal(actual_indices, expected_indices)


def test_optimized_moe_snapshot_uses_dispatch_view_while_aux_uses_clean_view(monkeypatch):
    module = OptimizedMOEImproved(
        in_channels=8,
        out_channels=8,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        noise_std=0.0,
        progressive_sparsity=False,
        add_residual=False,
    ).train()
    module._moe_force_snapshot = True
    clean_logits = torch.tensor(
        [[0.04, 0.03, 0.00, -0.01], [0.02, 0.01, -0.01, -0.02]],
        requires_grad=True,
    )
    clean_probs = F.softmax(clean_logits, dim=1)
    clean_indices = torch.topk(clean_probs, TOP_K, dim=1).indices
    dispatch_logits = clean_logits + torch.tensor(
        [[-0.10, -0.10, 0.20, 0.10], [-0.10, -0.10, 0.10, 0.20]]
    )
    dispatch_probs = F.softmax(dispatch_logits, dim=1)
    dispatch_values, dispatch_indices = torch.topk(dispatch_probs, TOP_K, dim=1)
    dispatch_weights = dispatch_values / dispatch_values.sum(dim=1, keepdim=True)
    loss_info = {
        "router_logits": clean_logits,
        "router_probs": clean_probs,
        "topk_indices": clean_indices,
        "dispatch_router_logits": dispatch_logits,
        "dispatch_router_probs": dispatch_probs,
        "dispatch_topk_indices": dispatch_indices,
    }
    monkeypatch.setattr(
        module.routing,
        "forward",
        lambda _inputs, top_k=None: (dispatch_weights, dispatch_indices, loss_info),
    )

    output = module(torch.randn(2, 8, 4, 4))
    expected_aux = module.moe_loss_fn(clean_probs, clean_logits, clean_indices)
    snapshot = module.last_routing_snapshot

    assert torch.isfinite(output).all()
    assert torch.equal(module.aux_loss, expected_aux)
    assert module.aux_loss.detach().item() > 0
    assert torch.allclose(snapshot["mean_router_probs"], dispatch_probs.detach().mean(dim=0))
    expected_dispatch_counts = torch.bincount(dispatch_indices.reshape(-1), minlength=NUM_EXPERTS).float()
    expected_dispatch_usage = expected_dispatch_counts / expected_dispatch_counts.sum()
    assert torch.equal(snapshot["topk_counts"], expected_dispatch_counts)
    assert torch.equal(snapshot["expert_usage"], expected_dispatch_usage)
    assert not torch.allclose(snapshot["mean_router_probs"], clean_probs.detach().mean(dim=0))
    task_loss = output.square().mean()
    composite_loss = task_loss + module.aux_loss
    assert torch.allclose(composite_loss - task_loss, module.aux_loss)


# =============================================================================
# _validate_router_input unit tests
# =============================================================================

class TestValidateRouterInput:
    def test_valid_4d_input_passes(self):
        x = _valid_input()
        _validate_router_input(x, IN_CHANNELS)  # should not raise

    def test_3d_input_raises(self):
        x = torch.randn(2, IN_CHANNELS, 16)
        with pytest.raises(MoERouterError, match="4-D"):
            _validate_router_input(x, IN_CHANNELS)

    def test_5d_input_raises(self):
        x = torch.randn(2, IN_CHANNELS, 16, 16, 1)
        with pytest.raises(MoERouterError, match="4-D"):
            _validate_router_input(x, IN_CHANNELS)

    def test_channel_mismatch_raises_shape_error(self):
        x = torch.randn(2, 32, 16, 16)  # 32 != 64
        with pytest.raises(ShapeMismatchError):
            _validate_router_input(x, IN_CHANNELS)

    def test_nan_input_raises(self):
        x = _valid_input()
        x[0, 0, 0, 0] = float("nan")
        with pytest.raises(MoERouterError, match="NaN"):
            _validate_router_input(x, IN_CHANNELS)

    def test_inf_input_raises(self):
        x = _valid_input()
        x[0, 0, 0, 0] = float("inf")
        with pytest.raises(MoERouterError, match="NaN"):
            _validate_router_input(x, IN_CHANNELS)

    def test_nonfinite_debug_path_never_performs_network_post(self, monkeypatch):
        import ultralytics.nn.modules.moe.routers as routers

        monkeypatch.setenv("ULTRA_DEBUG_NONFINITE", "1")
        monkeypatch.setenv("ULTRA_DEBUG_POST_URL", "http://127.0.0.1:9/collect")
        monkeypatch.setattr(routers, "urlopen", lambda *args, **kwargs: pytest.fail("network post attempted"), raising=False)
        with pytest.raises(MoERouterError):
            _validate_router_input(torch.full((1, IN_CHANNELS, 2, 2), float("nan")), IN_CHANNELS)


# =============================================================================
# UltraEfficientRouter boundary tests
# =============================================================================

class TestUltraEfficientRouterBoundaries:
    def test_valid_forward(self, ultra_router):
        x = _valid_input()
        ultra_router.eval()
        topk_vals, topk_idx, usage, imp, z = ultra_router(x)
        assert topk_vals.shape[0] == 2
        assert topk_idx.shape[0] == 2

    def test_3d_input_raises(self, ultra_router):
        x = torch.randn(2, IN_CHANNELS, 16)
        with pytest.raises(MoERouterError, match="4-D"):
            ultra_router(x)

    def test_channel_mismatch_raises(self, ultra_router):
        x = torch.randn(2, 32, 16, 16)
        with pytest.raises(ShapeMismatchError):
            ultra_router(x)

    def test_nan_input_raises(self, ultra_router):
        x = _valid_input()
        x[0, 0, 0, 0] = float("nan")
        with pytest.raises(MoERouterError, match="NaN"):
            ultra_router(x)

    def test_top_k_exceeds_num_experts_clamped(self, ultra_router):
        ultra_router.eval()
        x = _valid_input()
        topk_vals, topk_idx, _, _, _ = ultra_router(x, top_k=100)
        assert topk_vals.shape[1] <= NUM_EXPERTS

    def test_top_k_zero_clamped_to_one(self, ultra_router):
        ultra_router.eval()
        x = _valid_input()
        topk_vals, topk_idx, _, _, _ = ultra_router(x, top_k=0)
        assert topk_vals.shape[1] >= 1


# =============================================================================
# EfficientSpatialRouter boundary tests
# =============================================================================

class TestEfficientSpatialRouterBoundaries:
    def test_valid_forward(self, spatial_router):
        x = _valid_input()
        spatial_router.eval()
        result = spatial_router(x)
        assert result[0].shape[0] == 2

    def test_3d_input_raises(self, spatial_router):
        x = torch.randn(2, IN_CHANNELS, 16)
        with pytest.raises(MoERouterError, match="4-D"):
            spatial_router(x)

    def test_channel_mismatch_raises(self, spatial_router):
        x = torch.randn(2, 128, 16, 16)
        with pytest.raises(ShapeMismatchError):
            spatial_router(x)

    @pytest.mark.parametrize("noise_std", [float("nan"), float("inf")])
    def test_nonfinite_noise_std_raises(self, spatial_router, noise_std):
        spatial_router.noise_std = noise_std
        with pytest.raises(MoERouterError, match="noise_std"):
            spatial_router(_valid_input())

    def test_nonfinite_internal_router_output_raises(self, spatial_router, monkeypatch):
        monkeypatch.setattr(spatial_router.router, "forward", lambda _: torch.full((2, NUM_EXPERTS, 1, 1), float("nan")))
        with pytest.raises(MoERouterError, match="internal output"):
            spatial_router(_valid_input())


# =============================================================================
# AdaptiveRoutingLayer boundary tests
# =============================================================================

class TestAdaptiveRoutingLayerBoundaries:
    def test_valid_forward(self, adaptive_router):
        x = _valid_input()
        adaptive_router.eval()
        result = adaptive_router(x)
        assert result[0].shape[0] == 2

    def test_3d_input_raises(self, adaptive_router):
        x = torch.randn(2, IN_CHANNELS, 16)
        with pytest.raises(MoERouterError, match="4-D"):
            adaptive_router(x)


# =============================================================================
# LocalRoutingLayer boundary tests
# =============================================================================

class TestLocalRoutingLayerBoundaries:
    def test_valid_forward(self, local_router):
        x = _valid_input()
        local_router.eval()
        result = local_router(x)
        assert result[0].shape[0] == 2

    def test_3d_input_raises(self, local_router):
        x = torch.randn(2, IN_CHANNELS, 16)
        with pytest.raises(MoERouterError, match="4-D"):
            local_router(x)


class TestDynamicRoutingLayerBoundaries:
    def test_eval_hard_top_k_matches_training_mask_numerics(self):
        router = DynamicRoutingLayer(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K)
        x = _valid_input()
        logits = router.routing_network(router.global_pool(x))

        assert torch.allclose(router._hard_top_k(logits), router._soft_top_k(logits), atol=1e-6)

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            DynamicRoutingLayer(IN_CHANNELS, NUM_EXPERTS, top_k=0)

    def test_legacy_checkpoint_without_in_channels_preserves_output(self):
        torch.manual_seed(0)
        router = DynamicRoutingLayer(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K).eval()
        x = _valid_input()

        with torch.no_grad():
            expected = router(x)
        state_before = {name: tensor.clone() for name, tensor in router.state_dict().items()}

        del router.in_channels
        with torch.no_grad():
            actual = router(x)

        assert torch.equal(actual, expected)
        assert state_before.keys() == router.state_dict().keys()
        for name, tensor in router.state_dict().items():
            assert torch.equal(tensor, state_before[name])

    def test_legacy_checkpoint_without_in_channels_still_checks_channels(self):
        router = DynamicRoutingLayer(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K)
        del router.in_channels

        with pytest.raises(ShapeMismatchError):
            router(torch.randn(2, IN_CHANNELS // 2, 16, 16))


def test_capacity_is_per_expert_and_handles_biased_routing():
    from ultralytics.nn.modules.moe.routers import BaseRouter

    router = BaseRouter(num_experts=4, top_k=2, capacity_factor=0.5)
    logits = torch.tensor([[9.0, 8.0, 0.0, -1.0]]).repeat(8, 1)
    weights, indices, info = router._process_logits(logits, noise_std=0.0, training=True)

    assert info["capacity_limit"] == 2  # ceil(0.5 * 8 * 2 / 4)
    assert info["overflow_count"] == 12
    assert info["overflow_fraction"] == pytest.approx(12 / 16)
    assert info["overflow_mask"].shape == indices.shape
    for expert in range(4):
        accepted = (indices == expert) & ~info["overflow_mask"]
        assert accepted.sum().item() <= info["capacity_limit"]
    assert torch.equal(indices[2:, 0], torch.zeros(6, dtype=torch.long))
    assert torch.equal(weights[2:], torch.tensor([[1.0, 0.0]]).repeat(6, 1))
    assert torch.allclose(weights.sum(dim=1), torch.ones(8))


def test_capacity_indices_and_weights_drop_the_same_assignments():
    from ultralytics.nn.modules.moe.routers import BaseRouter

    router = BaseRouter(num_experts=3, top_k=2, capacity_factor=0.75)
    logits = torch.tensor([[7.0, 6.0, 0.0], [7.0, 6.0, 0.0], [7.0, 6.0, 0.0]])
    weights, indices, info = router._process_logits(logits, noise_std=0.0, training=True)

    assert info["capacity_limit"] == 2
    assert torch.equal(info["overflow_mask"][2], torch.tensor([True, True]))
    assert indices[2, 0].item() == 0
    assert torch.equal(weights[2], torch.tensor([1.0, 0.0]))


def test_capacity_overflow_is_deterministic_across_repeated_calls():
    from ultralytics.nn.modules.moe.routers import BaseRouter

    router = BaseRouter(num_experts=4, top_k=2, capacity_factor=0.5)
    logits = torch.zeros(12, 4)
    first = router._process_logits(logits, noise_std=0.0, training=True)
    second = router._process_logits(logits, noise_std=0.0, training=True)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2]["overflow_mask"], second[2]["overflow_mask"])


def test_capacity_overflow_hard_forward_retains_router_gradient():
    from ultralytics.nn.modules.moe.routers import BaseRouter

    router = BaseRouter(num_experts=4, top_k=2, capacity_factor=0.5)
    logits = torch.tensor([[9.0, 8.0, 0.0, -1.0]]).repeat(12, 1).requires_grad_()
    weights, _, info = router._process_logits(logits, noise_std=0.0, training=True)
    overflow = info["token_overflow_mask"]

    assert torch.equal(weights[overflow].detach(), torch.tensor([[1.0, 0.0]]).repeat(overflow.sum(), 1))
    weights[overflow, 0].sum().backward()

    overflow_gradient = logits.grad[overflow]
    assert torch.isfinite(overflow_gradient).all()
    assert torch.count_nonzero(overflow_gradient).item() > 0


@pytest.mark.parametrize("capacity_factor", [0.0, -1.0, float("nan"), float("inf")])
def test_capacity_factor_must_be_positive_and_finite(capacity_factor):
    from ultralytics.nn.modules.moe.routers import BaseRouter

    router = BaseRouter(num_experts=4, top_k=1, capacity_factor=capacity_factor)
    with pytest.raises(MoERouterError, match="capacity_factor"):
        router._process_logits(torch.zeros(2, 4), noise_std=0.0, training=True)

    def test_nonfinite_internal_output_raises(self, monkeypatch):
        router = DynamicRoutingLayer(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K)
        monkeypatch.setattr(
            router.routing_network,
            "forward",
            lambda _: torch.full((2, NUM_EXPERTS, 1, 1), float("nan")),
        )
        with pytest.raises(MoERouterError, match="internal output"):
            router(_valid_input())


# =============================================================================
# Exception hierarchy tests
# =============================================================================

class TestExceptionHierarchy:
    def test_moerouter_error_inherits_yolomaster(self):
        assert issubclass(MoERouterError, YOLOMasterError)

    def test_shapemismatch_error_inherits_yolomaster(self):
        assert issubclass(ShapeMismatchError, YOLOMasterError)

    def test_catch_all_with_yolomaster_error(self):
        """Caller can catch all YOLO-Master errors with one except clause."""
        x = torch.randn(2, 32, 16, 16)
        router = UltraEfficientRouter(IN_CHANNELS, NUM_EXPERTS)
        with pytest.raises(YOLOMasterError):
            router(x)


# =============================================================================
# FP16 routing precision regressions
# =============================================================================

class TestABlockMoEDiagnostics:
    def test_diagnostics_fail_at_first_nonfinite_boundary(self, monkeypatch):
        block = object.__new__(moe_base.ABlockMoE)
        nn.Module.__init__(block)
        block.attn = nn.Identity()
        block.mlp = nn.Identity()
        monkeypatch.setattr(moe_base, "_MOE_FINITE_DIAGNOSTICS", True)
        monkeypatch.setattr(moe_base, "_MOE_FINITE_DIAGNOSTIC_MAX_EVENTS", 1)

        x = torch.ones(1, 1, 1, 1)
        block.attn = nn.Sequential(nn.Identity())
        block.attn.forward = lambda _: torch.full_like(x, float("nan"))
        with pytest.raises(RuntimeError, match="attention output"):
            block(x)

    def test_diagnostics_are_disabled_by_default(self, monkeypatch):
        block = object.__new__(moe_base.ABlockMoE)
        nn.Module.__init__(block)
        block.attn = nn.Identity()
        block.mlp = nn.Identity()
        monkeypatch.setattr(moe_base, "_MOE_FINITE_DIAGNOSTICS", False)

        output = block(torch.ones(1, 1, 1, 1))
        assert torch.isfinite(output).all()


class TestEfficientSpatialRouterPrecision:
    def test_half_spatial_reduction_and_weights_stay_fp32(self):
        """Routing statistics and normalized Top-K weights avoid fp16 reductions."""
        torch.manual_seed(0)
        router = EfficientSpatialRouter(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K).eval().half()
        x = torch.randn(2, IN_CHANNELS, 32, 32, dtype=torch.float16)
        try:
            weights, indices, _ = router(x)
        except RuntimeError as exc:
            if "not implemented for 'Half'" in str(exc):
                pytest.skip(f"CPU fp16 operator unavailable: {exc}")
            raise

        assert weights.dtype == torch.float32
        assert indices.dtype == torch.long
        assert torch.isfinite(weights).all()
        assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)

    def test_process_logits_normalizes_extreme_half_logits_in_fp32(self):
        """Small selected probabilities retain precision after fp16-router output."""
        router = EfficientSpatialRouter(IN_CHANNELS, NUM_EXPERTS, top_k=TOP_K).eval()
        logits = torch.tensor([[12.0, 0.0, -12.0, -24.0]], dtype=torch.float16)
        weights, indices, _ = router._process_logits(logits, noise_std=0.0, training=False)

        assert weights.dtype == torch.float32
        assert torch.isfinite(weights).all()
        assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1e-6)
        assert indices.dtype == torch.long
