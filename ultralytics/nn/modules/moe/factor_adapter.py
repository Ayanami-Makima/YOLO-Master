"""Function-preserving residual adapters for the A1 P1 factorial experiment."""

from __future__ import annotations

import copy

import torch
from torch import nn

from ..block import A2C2f, C3k2
from .modules import A2C2fMoE


class ResidualFactorAdapter(nn.Module):
    """Keep a pretrained base block intact and learn a zero-gated residual factor.

    The adapter is exactly equivalent to ``base`` at initialization because the
    channel-wise residual gain is initialized to zero.  The base can therefore
    remain frozen while a dense or MoE factor branch is trained fairly.
    """

    def __init__(self, base: nn.Module, factor: nn.Module, channels: int, freeze_base: bool = True) -> None:
        """Wrap ``base`` with ``base(x) + gain * factor(base(x))``."""
        super().__init__()
        self.base = base
        self.factor = factor
        self.gain = nn.Parameter(torch.zeros(channels))
        self.freeze_base = freeze_base
        # Keep this policy on the module so the generic optimizer can discover
        # the P1 residual gate without introducing experiment-only config keys.
        self.p1_gain_lr_scale = 100.0
        self.p1_gain_no_warmup = True

        # Ultralytics' graph runner reads these attributes from each top-level
        # layer.  Preserve them when replacing a parsed model layer in-place.
        for attribute in ("i", "f", "type"):
            if hasattr(base, attribute):
                setattr(self, attribute, copy.deepcopy(getattr(base, attribute)))
        self.np = sum(parameter.numel() for parameter in self.parameters())
        if freeze_base:
            self.freeze_base_parameters()

    def freeze_base_parameters(self) -> int:
        """Freeze the pretrained path and return its parameter count."""
        self.base.eval()
        count = 0
        for parameter in self.base.parameters():
            parameter.requires_grad = False
            count += parameter.numel()
        return count

    def train(self, mode: bool = True):
        """Keep the frozen base in evaluation mode when the adapter trains."""
        super().train(mode)
        if self.freeze_base:
            self.base.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the immutable pretrained block and its learnable residual."""
        base_output = self.base(x)
        residual = self.factor(base_output)
        return base_output + self.gain.view(1, -1, 1, 1) * residual


class C3k2ResidualFactor(ResidualFactorAdapter):
    """YAML-reconstructable native C3k2 plus a zero-gated P1 factor branch."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        moe: bool = False,
        num_experts: int = 4,
        top_k: int = 2,
        expert_type: str = "dense_mlp",
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        """Construct a native base and a matched dense or sparse residual."""
        base = C3k2(c1, c2, n=n, c3k=c3k, e=e, attn=attn, g=g, shortcut=shortcut)
        factor_args = {
            "c1": c2,
            "c2": c2,
            "n": n,
            "a2": True,
            "area": 1,
            "residual": False,
            "mlp_ratio": 2.0,
            "e": 0.5,
            "g": 1,
            "shortcut": True,
        }
        factor = (
            A2C2fMoE(
                **factor_args,
                num_experts=num_experts,
                top_k=top_k,
                expert_type=expert_type,
            )
            if moe
            else A2C2f(**factor_args)
        )
        if moe:
            # Train and audit with the same hard Top-K cardinality.  A versioned
            # P1 runner may still apply calibrated Gaussian exploration to the
            # logits during the first epoch; this never changes K and is off in
            # validation, inference, and export.
            for module in factor.modules():
                if hasattr(module, "progressive_sparsity") and hasattr(module, "top_k"):
                    module.progressive_sparsity = False
                    module._current_top_k = module.top_k
                    module.warmup_steps = 0
                    module.expert_dropout_rate = 0.0
                    routing = getattr(module, "routing", None)
                    if routing is not None and hasattr(routing, "noise_std"):
                        routing.noise_std = 0.0
        self.moe = moe
        self.num_experts = num_experts if moe else 0
        self.top_k = top_k if moe else 0
        self.routing_schedule = "hard_top_k_from_step_zero" if moe else "not_applicable"
        self.router_noise_std = 0.0 if moe else None
        self.router_noise_schedule = "runner_controlled_or_disabled" if moe else "not_applicable"
        self.expert_dropout_rate = 0.0 if moe else None
        super().__init__(base, factor, c2, freeze_base=True)
