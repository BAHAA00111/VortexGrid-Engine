import pytest
from vortexgrid.engine.distributed_context import (
    DistributedConfig,
    DistributedContext,
    get_context,
    get_rank,
    get_world_size,
    is_master,
)


@pytest.fixture(autouse=True)
def reset_context_singleton():
    """Pytest fixture to clean up singleton state before and after each test execution."""
    DistributedContext._initialized = False
    DistributedContext._instance = None
    yield
    DistributedContext._initialized = False
    DistributedContext._instance = None


@pytest.mark.unit
def test_single_process_initialization():
    """Validates default single-process fallback attributes."""
    ctx = get_context()
    assert ctx.rank == 0
    assert ctx.world_size == 1
    assert ctx.is_master is True
    assert ctx.local_rank == 0
    assert is_master() is True
    assert get_rank() == 0
    assert get_world_size() == 1


@pytest.mark.unit
def test_custom_distributed_config():
    """Validates custom DistributedConfig injection into the context."""
    config = DistributedConfig(timeout_seconds=600, nccl_buffsize=8388608)
    ctx = DistributedContext(config=config).initialize()
    assert ctx.config.timeout_seconds == 600
    assert ctx.config.nccl_buffsize == 8388608


@pytest.mark.unit
def test_telemetry_summary_structure():
    """Validates that get_summary returns all required monitoring metrics."""
    ctx = get_context()
    summary = ctx.get_summary()

    required_keys = {
        "rank",
        "local_rank",
        "world_size",
        "local_world_size",
        "node_id",
        "is_master",
        "device",
        "backend",
        "cuda_available",
        "allocated_vram_mb",
    }
    assert required_keys.issubset(summary.keys())
