"""
Prometheus & Grafana Exporter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Exposes real-time distributed hardware, memory, and throughput metrics over an HTTP endpoint
for Prometheus scraping and Grafana dashboard visualization.
"""

from __future__ import annotations

import time
from typing import Optional, Union

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    REGISTRY,
    start_http_server,
)

from vortexgrid import logger
from vortexgrid.telemetry.metrics_collector import StepTelemetryPayload


class PrometheusExporter:
    """
    HTTP Prometheus metrics exporter for VortexGrid distributed training workloads.
    """

    def __init__(
        self,
        port: int = 9090,
        host: str = "0.0.0.0",
        registry: Optional[CollectorRegistry] = None,
    ) -> None:
        self.port = port
        self.host = host
        self.registry = registry or CollectorRegistry()
        self._server_started = False

        # Define Prometheus Metrics Gauges & Counters with isolated registry
        self.gauges = {
            "mfu": Gauge(
                "vortexgrid_mfu_percent",
                "Model FLOPs Utilization (MFU) percentage",
                ["rank"],
                registry=self.registry,
            ),
            "tflops": Gauge(
                "vortexgrid_tflops_achieved",
                "Achieved hardware TFLOPS per GPU",
                ["rank"],
                registry=self.registry,
            ),
            "throughput": Gauge(
                "vortexgrid_tokens_per_second",
                "Token processing throughput (tokens/sec)",
                ["rank"],
                registry=self.registry,
            ),
            "vram_allocated": Gauge(
                "vortexgrid_vram_allocated_mb",
                "Allocated VRAM memory in MB",
                ["rank"],
                registry=self.registry,
            ),
            "vram_reserved": Gauge(
                "vortexgrid_vram_reserved_mb",
                "Reserved VRAM memory in MB",
                ["rank"],
                registry=self.registry,
            ),
            "vram_fragmentation": Gauge(
                "vortexgrid_vram_fragmentation_ratio",
                "VRAM allocator memory fragmentation ratio",
                ["rank"],
                registry=self.registry,
            ),
            "nccl_barrier_ms": Gauge(
                "vortexgrid_nccl_barrier_latency_ms",
                "NCCL rank synchronization barrier overhead in milliseconds",
                ["rank"],
                registry=self.registry,
            ),
            "current_step": Gauge(
                "vortexgrid_current_step",
                "Current training step counter",
                ["rank"],
                registry=self.registry,
            ),
        }

        self.step_counter = Counter(
            "vortexgrid_total_steps_completed",
            "Total number of training steps completed",
            ["rank"],
            registry=self.registry,
        )

    def start_server(self) -> None:
        """Starts the background Prometheus HTTP metric server."""
        if self._server_started:
            return

        try:
            start_http_server(port=self.port, addr=self.host, registry=self.registry)
            self._server_started = True
            logger.info(
                f"Prometheus HTTP exporter server started on http://{self.host}:{self.port}/metrics"
            )
        except Exception as e:
            logger.error(
                f"Failed to start Prometheus exporter HTTP server on port {self.port}: {str(e)}"
            )

    def export_step_metrics(self, payload: StepTelemetryPayload) -> None:
        """Publishes a StepTelemetryPayload snapshot to Prometheus Gauges."""
        rank_str = str(payload.global_rank)

        # Hardware & throughput metrics
        self.gauges["mfu"].labels(rank=rank_str).set(
            payload.hardware_stats.get("mfu_percent", 0.0)
        )
        self.gauges["tflops"].labels(rank=rank_str).set(
            payload.hardware_stats.get("tflops_achieved", 0.0)
        )
        self.gauges["throughput"].labels(rank=rank_str).set(payload.tokens_per_second)

        # Memory state metrics
        self.gauges["vram_allocated"].labels(rank=rank_str).set(
            payload.memory_stats.get("allocated_mb", 0.0)
        )
        self.gauges["vram_reserved"].labels(rank=rank_str).set(
            payload.memory_stats.get("reserved_mb", 0.0)
        )
        self.gauges["vram_fragmentation"].labels(rank=rank_str).set(
            payload.memory_stats.get("fragmentation_ratio", 0.0)
        )

        # Communications & step metadata
        self.gauges["nccl_barrier_ms"].labels(rank=rank_str).set(
            payload.nccl_barrier_ms
        )
        self.gauges["current_step"].labels(rank=rank_str).set(payload.step)

        self.step_counter.labels(rank=rank_str).inc()
