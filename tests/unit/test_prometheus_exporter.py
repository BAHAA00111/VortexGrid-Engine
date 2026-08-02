import pytest
from prometheus_client import CollectorRegistry
from vortexgrid.telemetry.metrics_collector import StepTelemetryPayload
from vortexgrid.telemetry.prometheus_exporter import PrometheusExporter


@pytest.mark.unit
def test_prometheus_exporter_initialization():
    registry = CollectorRegistry()
    exporter = PrometheusExporter(port=9099, registry=registry)
    assert exporter.port == 9099
    assert exporter._server_started is False


@pytest.mark.unit
def test_prometheus_exporter_metric_update():
    registry = CollectorRegistry()
    exporter = PrometheusExporter(port=9098, registry=registry)

    payload = StepTelemetryPayload(
        step=10,
        global_rank=0,
        tokens_per_second=1500.0,
        seq_len=2048,
        batch_size=4,
        memory_stats={"allocated_mb": 12000.0, "reserved_mb": 16000.0, "fragmentation_ratio": 0.25},
        hardware_stats={"mfu_percent": 48.5, "tflops_achieved": 150.0},
        nccl_barrier_ms=1.2,
    )

    # Should update gauges cleanly without error
    exporter.export_step_metrics(payload)