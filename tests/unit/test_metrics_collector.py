import time
import pytest
from vortexgrid.telemetry.metrics_collector import MetricsCollector, StepTelemetryPayload


@pytest.mark.unit
def test_metrics_collector_step_lifecycle():
    collector = MetricsCollector(
        num_params=100_000_000,
        hidden_dim=768,
        num_layers=12,
        enable_memory_profiling=False,
        device=-1,
    )

    collector.start_step()
    time.sleep(0.01)  # Simulate small step workload

    payload = collector.collect_step_metrics(
        step=1,
        batch_size=4,
        seq_len=512,
        is_training=True,
    )

    assert isinstance(payload, StepTelemetryPayload)
    assert payload.step == 1
    assert payload.tokens_per_second > 0
    assert "mfu_percent" in payload.hardware_stats
    assert "allocated_mb" in payload.memory_stats

    collector.shutdown()


@pytest.mark.unit
def test_metrics_collector_json_serialization():
    collector = MetricsCollector(
        num_params=50_000_000,
        hidden_dim=512,
        num_layers=6,
        enable_memory_profiling=False,
        device=-1,
    )

    collector.start_step()
    payload = collector.collect_step_metrics(step=10, batch_size=2, seq_len=128)
    json_str = payload.to_json()

    assert '"step": 10' in json_str
    assert '"tokens_per_second"' in json_str

    collector.shutdown()