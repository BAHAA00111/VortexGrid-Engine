import pytest
from vortexgrid.telemetry.hardware_profiler import HardwarePerformanceStats, HardwareProfiler


@pytest.mark.unit
def test_transformer_flops_calculation():
    # Test FLOP calculation for 1B parameter model with batch_size=2, seq_len=2048
    flops = HardwareProfiler.calculate_transformer_flops(
        num_params=1_000_000_000,
        seq_len=2048,
        batch_size=2,
        num_layers=24,
        hidden_dim=4096,
        is_training=True,
    )

    assert flops > 0
    # ~6 * 1e9 * 2 * 2048 = 2.4576e13 + attention FLOPS
    assert flops > 2.4e13


@pytest.mark.unit
def test_hardware_profiler_cpu_fallback():
    profiler = HardwareProfiler(device=-1)
    profiler.start_step_timer()
    stats = profiler.stop_step_timer_and_compute_stats(total_flops=1e12)

    assert isinstance(stats, HardwarePerformanceStats)
    assert stats.tflops_achieved == 0.0
    assert stats.comm_latency_ms == 0.0


@pytest.mark.unit
def test_comm_latency_single_node():
    profiler = HardwareProfiler(device=-1)
    # Single-node / uninitialized distributed setup returns 0.0 ms
    latency = profiler.measure_comm_latency()
    assert latency == 0.0