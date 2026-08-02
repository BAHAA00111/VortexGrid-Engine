"""
Communication Latency & TFLOPS Profiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Measures real-time NCCL distributed collective communication latencies and tracks
achieved hardware TFLOPS and Model FLOPs Utilization (MFU) using high-precision CUDA events.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist


@dataclass
class HardwarePerformanceStats:
    """Performance telemetry snapshot containing communication latency and compute throughput metrics."""

    comm_latency_ms: float
    tflops_achieved: float
    mfu_percent: float
    step_time_ms: float
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HardwareProfiler:
    """
    Measures GPU compute performance (TFLOPS / MFU) and NCCL collective latency.

    Utilizes CUDA events for microsecond-accurate kernel timing without CPU synchronization overhead.
    """

    def __init__(
        self,
        peak_gpu_tflops: float = 312.0,  # Peak Tensor Core FP16/BF16 TFLOPS for NVIDIA A100 (SXM4)
        device: Optional[Union[torch.device, int]] = None,
    ) -> None:
        self.peak_gpu_tflops = peak_gpu_tflops

        if device is None:
            self.device_index = (
                torch.cuda.current_device() if torch.cuda.is_available() else -1
            )
        elif isinstance(device, torch.device):
            self.device_index = device.index if device.index is not None else 0
        else:
            self.device_index = device

        self._is_cuda = torch.cuda.is_available() and self.device_index >= 0

        # CUDA Event timers for step and communication profiling
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None

        if self._is_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)

    def measure_comm_latency(
        self,
        tensor_size_mb: float = 16.0,
        num_warmup: int = 2,
        num_iters: int = 5,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> float:
        """
        Executes benchmark NCCL All-Reduce ops to record rank-to-rank latency in milliseconds.
        """
        if not dist.is_initialized() or not self._is_cuda:
            return 0.0

        world_size = dist.get_world_size(process_group)
        if world_size <= 1:
            return 0.0

        num_elements = int((tensor_size_mb * 1024 * 1024) / 4)  # float32 elements
        dummy_tensor = torch.ones(
            num_elements, device=f"cuda:{self.device_index}", dtype=torch.float32
        )

        # Warmup iterations
        for _ in range(num_warmup):
            dist.all_reduce(dummy_tensor, group=process_group)

        torch.cuda.synchronize(self.device_index)
        start_time = time.perf_counter()

        for _ in range(num_iters):
            dist.all_reduce(dummy_tensor, group=process_group)

        torch.cuda.synchronize(self.device_index)
        elapsed_sec = (time.perf_counter() - start_time) / num_iters
        latency_ms = elapsed_sec * 1000.0

        return round(latency_ms, 3)

    @staticmethod
    def calculate_transformer_flops(
        num_params: int,
        seq_len: int,
        batch_size: int,
        num_layers: int,
        hidden_dim: int,
        is_training: bool = True,
    ) -> float:
        """
        Calculates theoretical FLOPs for standard Transformer forward/backward passes.

        Formula: 6 * N * B * S + 12 * L * H * B * S^2 (6 FLOPs per param + Self-Attention matrix ops)
        """
        # 6 FLOPs per parameter (2 for forward pass, 4 for backward pass)
        factor = 6.0 if is_training else 2.0
        param_flops = factor * num_params * batch_size * seq_len

        # Self-Attention QK^T and Attention-Value multiplications
        attn_flops = 12.0 * num_layers * hidden_dim * batch_size * (seq_len**2)

        return param_flops + attn_flops

    def start_step_timer(self) -> None:
        """Records start CUDA event for step latency timing."""
        if self._is_cuda and self._start_event:
            self._start_event.record()

    def stop_step_timer_and_compute_stats(
        self,
        total_flops: float,
        comm_latency_ms: float = 0.0,
    ) -> HardwarePerformanceStats:
        """
        Stops step CUDA event, synchronizes, and returns TFLOPS and MFU performance metrics.
        """
        if not self._is_cuda or self._start_event is None or self._end_event is None:
            return HardwarePerformanceStats(
                comm_latency_ms=0.0,
                tflops_achieved=0.0,
                mfu_percent=0.0,
                step_time_ms=0.0,
                timestamp=time.time(),
            )

        self._end_event.record()
        torch.cuda.synchronize(self.device_index)

        step_time_ms = self._start_event.elapsed_time(self._end_event)
        step_time_sec = step_time_ms / 1000.0

        if step_time_sec > 0:
            achieved_tflops = (total_flops / step_time_sec) / 1e12
            mfu_percent = (achieved_tflops / self.peak_gpu_tflops) * 100.0
        else:
            achieved_tflops = 0.0
            mfu_percent = 0.0

        return HardwarePerformanceStats(
            comm_latency_ms=round(comm_latency_ms, 3),
            tflops_achieved=round(achieved_tflops, 2),
            mfu_percent=round(mfu_percent, 2),
            step_time_ms=round(step_time_ms, 2),
            timestamp=time.time(),
        )
