"""
Centralized Metrics Collector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Aggregates PyTorch C++/CUDA runtime metrics, tracking Model FLOPs Utilization (MFU),
token processing throughput (tokens/sec), NCCL collective overhead, and VRAM fragmentation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist

from vortexgrid import logger
from vortexgrid.telemetry.hardware_profiler import HardwarePerformanceStats, HardwareProfiler
from vortexgrid.telemetry.memory_profiler import MemoryProfiler, MemoryStats


@dataclass
class StepTelemetryPayload:
    """Unified telemetry payload for a single training step."""

    step: int
    global_rank: int
    tokens_per_second: float
    seq_len: int
    batch_size: int
    memory_stats: Dict[str, Any]
    hardware_stats: Dict[str, Any]
    nccl_barrier_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class MetricsCollector:
    """
    High-performance telemetry collector that captures hardware metrics,
    throughput statistics, and distributed barrier overheads during model training.
    """

    def __init__(
        self,
        num_params: int,
        hidden_dim: int,
        num_layers: int,
        peak_gpu_tflops: float = 312.0,
        enable_memory_profiling: bool = True,
        device: Optional[Union[torch.device, int]] = None,
    ) -> None:
        self.num_params = num_params
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.memory_profiler = MemoryProfiler(
            device=device,
            enable_trace_history=enable_memory_profiling,
        )
        self.hardware_profiler = HardwareProfiler(
            peak_gpu_tflops=peak_gpu_tflops,
            device=device,
        )

        self._step_start_time: float = 0.0
        self.history: List[StepTelemetryPayload] = []

    def start_step(self) -> None:
        """Marks the start of a training step and initializes CUDA timers."""
        self._step_start_time = time.perf_counter()
        self.hardware_profiler.start_step_timer()

    def measure_nccl_barrier_overhead(
        self,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> float:
        """Measures the current rank synchronization delay across distributed workers."""
        if not dist.is_initialized():
            return 0.0

        start = time.perf_counter()
        dist.barrier(group=process_group)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return round(elapsed_ms, 3)

    def collect_step_metrics(
        self,
        step: int,
        batch_size: int,
        seq_len: int,
        is_training: bool = True,
        measure_barrier: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> StepTelemetryPayload:
        """
        Calculates step throughput, FLOPs, MFU, VRAM stats, and NCCL overheads.
        """
        step_duration = time.perf_counter() - self._step_start_time
        total_tokens = batch_size * seq_len
        tokens_per_sec = (total_tokens / step_duration) if step_duration > 0 else 0.0

        # Calculate theoretical Transformer FLOPs for MFU computation
        total_flops = HardwareProfiler.calculate_transformer_flops(
            num_params=self.num_params,
            seq_len=seq_len,
            batch_size=batch_size,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            is_training=is_training,
        )

        # Measure optional NCCL barrier overhead
        barrier_ms = (
            self.measure_nccl_barrier_overhead(process_group)
            if measure_barrier
            else 0.0
        )

        # Stop timers and compile hardware stats
        hw_stats: HardwarePerformanceStats = (
            self.hardware_profiler.stop_step_timer_and_compute_stats(
                total_flops=total_flops,
                comm_latency_ms=barrier_ms,
            )
        )

        # Fetch current memory state
        mem_stats: MemoryStats = self.memory_profiler.get_memory_stats()

        rank = dist.get_rank() if dist.is_initialized() else 0

        payload = StepTelemetryPayload(
            step=step,
            global_rank=rank,
            tokens_per_second=round(tokens_per_sec, 2),
            seq_len=seq_len,
            batch_size=batch_size,
            memory_stats=mem_stats.to_dict(),
            hardware_stats=hw_stats.to_dict(),
            nccl_barrier_ms=barrier_ms,
        )

        self.history.append(payload)

        # Log metrics at DEBUG level
        logger.debug(
            f"[Step {step} | Rank {rank}] MFU: {hw_stats.mfu_percent:.2f}% | "
            f"Throughput: {payload.tokens_per_second:.1f} tok/s | "
            f"Peak VRAM: {mem_stats.max_allocated_mb:.1f}MB | "
            f"VRAM Frag: {mem_stats.fragmentation_ratio:.2f}"
        )

        return payload

    def export_metrics_log(self, file_path: Union[str, Path]) -> None:
        """Exports gathered step metrics history to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [p.to_dict() for p in self.history]
        with open(path, "w", encoding="utf-8") as f:
            json.dumps(data, indent=2)

        logger.info(f"Exported {len(self.history)} telemetry entries to '{path}'")

    def shutdown(self) -> None:
        """Shuts down underlying profiler threads and memory recording."""
        self.memory_profiler.shutdown()