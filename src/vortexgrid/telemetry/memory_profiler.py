"""
CUDA Memory Profiler & Snapshot Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provides real-time visibility into GPU VRAM allocations, peak usage, memory reservation,
and fragmentation ratios. Captures PyTorch CUDA memory snapshots for visual debugging.
"""
from __future__ import annotations

import gc
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist

from vortexgrid import logger


@dataclass
class MemoryStats:
    """Container for active GPU VRAM state metrics in Megabytes (MB)."""
    
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float
    free_mb: float
    total_mb: float
    fragmentation_ratio: float
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryProfiler:
    """
    Monitors CUDA VRAM consumption, tracks peak allocation history,
    computes fragmentation ratios, and exports memory history snapshots.
    """
    def __init__(
        self,
        device: Optional[Union[torch.device, int]] = None,
        enable_trace_history: bool = True,
        max_trace_entries: int = 100000,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                self.device_index = torch.cuda.current_device()
            else:
                self.device_index = -1
        elif isinstance(device, torch.device):
            self.device_index = device.index if device.index is not None else 0
        else:
            self.device_index = device

        self.enable_trace_history = enable_trace_history
        self.max_trace_entries = max_trace_entries

        if self._is_cuda_active():
            if self.enable_trace_history:
                # Enable memory history recording for detailed trace snapshots
                try:
                    torch.cuda.memory._record_memory_history(
                        enabled="all",
                        max_entries=self.max_trace_entries,
                        device=self.device_index,
                    )
                    logger.info(
                        f"CUDA Memory History recording initialized on device cuda:{self.device_index}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to enable CUDA memory history tracing: {str(e)}"
                    )

    def _is_cuda_active(self) -> bool:
        return torch.cuda.is_available() and self.device_index >= 0

    def get_memory_stats(self) -> MemoryStats:
        """Captures real-time CUDA memory metrics and calculates fragmentation ratio."""
        if not self._is_cuda_active():
            return MemoryStats(
                allocated_mb=0.0,
                reserved_mb=0.0,
                max_allocated_mb=0.0,
                max_reserved_mb=0.0,
                free_mb=0.0,
                total_mb=0.0,
                fragmentation_ratio=0.0,
                timestamp=time.time(),
            )

        bytes_per_mb = 1024 * 1024
        allocated = torch.cuda.memory_allocated(self.device_index) / bytes_per_mb
        reserved = torch.cuda.memory_reserved(self.device_index) / bytes_per_mb
        max_allocated = torch.cuda.max_memory_allocated(self.device_index) / bytes_per_mb
        max_reserved = torch.cuda.max_memory_reserved(self.device_index) / bytes_per_mb

        total_bytes, free_bytes = torch.cuda.mem_get_info(self.device_index)
        total_mb = total_bytes / bytes_per_mb
        free_mb = free_bytes / bytes_per_mb

        # Fragmentation Ratio: Unallocated reserved memory vs total reserved memory
        # High fragmentation (>0.35) indicates CUDA allocator memory bloating
        if reserved > 0:
            fragmentation = (reserved - allocated) / reserved
        else:
            fragmentation = 0.0

        return MemoryStats(
            allocated_mb=round(allocated, 2),
            reserved_mb=round(reserved, 2),
            max_allocated_mb=round(max_allocated, 2),
            max_reserved_mb=round(max_reserved, 2),
            free_mb=round(free_mb, 2),
            total_mb=round(total_mb, 2),
            fragmentation_ratio=round(fragmentation, 4),
            timestamp=time.time(),
        )

    def dump_memory_snapshot(
        self,
        export_path: Union[str, Path],
        tag: str = "snapshot",
    ) -> Optional[Path]:
        if not self._is_cuda_active():
            logger.warning("Skipping CUDA memory snapshot dump: CUDA is not available.")
            return None

        rank = dist.get_rank() if dist.is_initialized() else 0
        path = Path(export_path)
        path.mkdir(parents=True, exist_ok=True)

        snapshot_file = path / f"cuda_memory_{tag}_rank{rank}.pickle"

        try:
            snapshot = torch.cuda.memory._snapshot(device=self.device_index)
            with open(snapshot_file, "wb") as f:
                pickle.dump(snapshot, f)

            logger.info(
                f"Exported CUDA memory snapshot to '{snapshot_file}' (Rank {rank})"
            )
            return snapshot_file
        except Exception as e:
            logger.error(
                f"Failed to export CUDA memory snapshot to '{snapshot_file}': {str(e)}"
            )
            return None

    def reset_peak_stats(self) -> None:
        """Resets peak memory allocation trackers."""
        if self._is_cuda_active():
            torch.cuda.reset_peak_memory_stats(self.device_index)

    def purge_cache(self) -> None:
        """Cleans unused cached allocator memory and triggers garbage collection."""
        gc.collect()
        if self._is_cuda_active():
            torch.cuda.empty_cache()

    def shutdown(self) -> None:
        """Disables memory history tracking upon profiler destruction."""
        if self._is_cuda_active() and self.enable_trace_history:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass