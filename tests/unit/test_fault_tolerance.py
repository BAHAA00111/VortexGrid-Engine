import pytest
import torch
import torch.nn as nn
from vortexgrid.checkpointing.async_sharded_saver import AsyncShardedSaver
from vortexgrid.checkpointing.fault_tolerance import (
    ElasticFaultHandler,
    FaultToleranceConfig,
    HeartbeatMonitor,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 8)

    def forward(self, x):
        return self.layer(x)


@pytest.mark.unit
def test_heartbeat_monitor_single_node():
    monitor = HeartbeatMonitor(interval_seconds=0.1)
    assert monitor.ping() is True


@pytest.mark.unit
def test_fault_handler_oom_recovery():
    model = DummyModel()
    config = FaultToleranceConfig(max_retries=2, retry_delay_seconds=0.1)
    handler = ElasticFaultHandler(model=model, config=config)

    attempts = 0

    def flaky_step():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory in test simulation")
        return "success"

    success, result = handler.execute_with_fault_tolerance(flaky_step)
    assert success is True
    assert result == "success"
    assert attempts == 2


@pytest.mark.unit
def test_fault_handler_auto_recover_latest_checkpoint(tmp_path):
    # Prepare saver and write dummy step 10 checkpoint
    saver = AsyncShardedSaver(base_dir=tmp_path)
    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_dir = saver.save_checkpoint(
        step=10,
        model=model,
        optimizer=optimizer,
        async_save=False,
    )
    saver.shutdown()

    # Instantiate fault handler with mock checkpoint location
    config = FaultToleranceConfig(recovery_checkpoint_dir=str(tmp_path))
    handler = ElasticFaultHandler(model=model, optimizer=optimizer, config=config)

    extra = handler.recover_from_latest_checkpoint()
    assert extra is not None