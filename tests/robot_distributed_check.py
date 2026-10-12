"""Run with torchrun --standalone --nproc_per_node=2 tests/robot_distributed_check.py."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from test_robot_model import model as model_fixture, sample


def main():
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    model = model_fixture.__wrapped__()
    wrapped = DistributedDataParallel(model, broadcast_buffers=False)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        loss,_ = wrapped(sample(('manip',) if rank == 0 else ('nav',)))
        loss.backward()
        assert torch.isfinite(loss)
        for parameter in model.mot.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    # DDP averages gradients: local scaling must recover the global valid mean.
    value=torch.tensor(float(rank+1),requires_grad=True)
    count=torch.tensor(1 if rank == 0 else 3)
    normalized=model._global_means((value,), (count,))[0]
    normalized.backward()
    assert value.grad.item() == .5
    if rank == 0: print('PASS: two DDP updates with different active streams; global count scaling')
    dist.destroy_process_group()


if __name__ == '__main__': main()
