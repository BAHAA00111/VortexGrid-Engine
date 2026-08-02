from pathlib import Path
import pytest
import torch
from vortexgrid.telemetry.memory_profiler import MemoryProfiler, MemoryStats


@pytest.mark.unit
def test_memory_stats_cpu_fallback():
    # Test CPU / non-CUDA fallback return behavior
    profiler = MemoryProfiler(device=-1, enable_trace_history=False)
    stats = profiler.get_memory_stats()

    assert isinstance(stats, MemoryStats)
    assert stats.allocated_mb == 0.0
    assert stats.fragmentation_ratio == 0.0
    assert isinstance(stats.to_dict(), dict)


@pytest.mark.unit
def test_memory_profiler_reset_and_purge():
    profiler = MemoryProfiler(enable_trace_history=False)
    profiler.purge_cache()
    profiler.reset_peak_stats()
    profiler.shutdown()


@pytest.mark.unit
def test_memory_snapshot_export_fallback(tmp_path: Path):
    profiler = MemoryProfiler(device=-1, enable_trace_history=False)
    snapshot_path = profiler.dump_memory_snapshot(export_path=tmp_path, tag="test")

    # Should safely return None on non-CUDA environments
    assert snapshot_path is None
