"""Two-rank CPU/Gloo continuous-training gate for a real routed module."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ultralytics.nn.modules.moe.modules import OptimizedMOE
from ultralytics.utils.torch_utils import TORCH_1_9


def _init_gloo(rank: int, world: int, timeout: timedelta) -> None:
    """Initialize Gloo without libuv on modern Windows torchrun workers."""
    if os.name == "nt" and TORCH_1_9:
        store = dist.TCPStore(
            host_name=os.environ["MASTER_ADDR"],
            port=int(os.environ["MASTER_PORT"]),
            world_size=world,
            is_master=False,
            timeout=timeout,
            use_libuv=False,
        )
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world, timeout=timeout)
        return
    dist.init_process_group("gloo", timeout=timeout)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 2, f"P0 gate requires exactly two ranks, got {world}"
    torch.set_num_threads(1)
    _init_gloo(rank, world, timedelta(seconds=60))
    try:
        torch.manual_seed(1234)
        model = OptimizedMOE(8, 8, num_experts=2, top_k=2)
        ddp = DDP(model, find_unused_parameters=True, broadcast_buffers=False)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)
        # Spatially varying, rank-consistent inputs avoid GroupNorm's exact
        # cancellation for constant inputs while keeping DDP gradients equal.
        inputs = torch.linspace(0.5, 1.5, steps=4 * 8 * 2 * 2).reshape(4, 8, 2, 2)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = ddp(inputs + step * 0.25).square().mean()
            loss.backward()
            grads = [p.grad for p in ddp.module.parameters() if p.requires_grad and p.grad is not None]
            assert grads, "routed module produced no gradients"
            assert all(torch.isfinite(grad).all() for grad in grads), "non-finite routed gradient"
            assert sum(float(grad.abs().sum()) for grad in grads) > 0.0, "all routed gradients are zero"
            optimizer.step()
            flat = torch.cat([p.detach().reshape(-1) for p in ddp.module.parameters()])
            gathered = [torch.empty_like(flat) for _ in range(world)]
            dist.all_gather(gathered, flat)
            assert torch.allclose(gathered[0], gathered[1]), f"parameters diverged after step {step}"
        if rank == 0:
            print("P0 routed DDP gate passed: backend=gloo, world_size=2, steps=2")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
