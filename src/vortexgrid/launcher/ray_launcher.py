"""
Ray Cluster Orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provides programmatic Ray job submission, GPU placement group placement,
and multi-node cluster autoscaling management matching 'manifests/ray/cluster-config.yaml'.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from vortexgrid import logger

try:
    import ray
    from ray.job_submission import JobStatus, JobSubmissionClient
    from ray.util.placement_group import PlacementGroup, placement_group

    HAS_RAY = True
except ImportError:
    HAS_RAY = False


@dataclass
class RayClusterSpec:
    """Dataclass holding cluster and job deployment specifications."""

    cluster_address: str = "auto"
    dashboard_url: str = "http://127.0.0.1:8265"
    num_workers: int = 4
    gpus_per_worker: int = 1
    cpus_per_worker: int = 8
    placement_strategy: str = "SPREAD"  # Options: SPREAD, STRICT_SPREAD, PACK
    pip_dependencies: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, manifest_path: Union[str, Path]) -> RayClusterSpec:
        """Parses manifest YAML configuration file to generate RayClusterSpec."""
        path = Path(manifest_path)
        if not path.exists():
            logger.warning(
                f"Manifest file '{manifest_path}' not found. Falling back to default spec."
            )
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        config: Dict[str, Any] = raw_config if isinstance(raw_config, dict) else {}

        cluster_cfg: Dict[str, Any] = (
            config.get("cluster", {}) if isinstance(config.get("cluster"), dict) else {}
        )
        worker_cfg: Dict[str, Any] = (
            config.get("worker_group", {})
            if isinstance(config.get("worker_group"), dict)
            else {}
        )
        runtime_cfg: Dict[str, Any] = (
            config.get("runtime_env", {})
            if isinstance(config.get("runtime_env"), dict)
            else {}
        )

        return cls(
            cluster_address=str(cluster_cfg.get("address", "auto")),
            dashboard_url=str(
                cluster_cfg.get("dashboard_url", "http://127.0.0.1:8265")
            ),
            num_workers=int(worker_cfg.get("num_workers", 4)),
            gpus_per_worker=int(worker_cfg.get("gpus_per_worker", 1)),
            cpus_per_worker=int(worker_cfg.get("cpus_per_worker", 8)),
            placement_strategy=str(worker_cfg.get("placement_strategy", "SPREAD")),
            pip_dependencies=list(runtime_cfg.get("pip", [])),
            env_vars=dict(runtime_cfg.get("env_vars", {})),
        )


class RayLauncher:
    """
    a cluster launcher for orchestrating distributed Ray tasks and jobs.
    """

    def __init__(
        self,
        spec: Optional[RayClusterSpec] = None,
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if not HAS_RAY:
            raise RuntimeError(
                "Ray SDK is not installed. Install via 'pip install ray[default]' to use RayLauncher."
            )

        if manifest_path is not None:
            self.spec = RayClusterSpec.from_manifest(manifest_path)
        else:
            self.spec = spec or RayClusterSpec()

        self._job_client: Optional[JobSubmissionClient] = None
        self._current_placement_group: Optional[PlacementGroup] = None

    def initialize_cluster_connection(self) -> None:
        """Establishes connection to the running Ray cluster head node."""
        if not ray.is_initialized():
            logger.info(
                f"Connecting to Ray cluster at '{self.spec.cluster_address}'..."
            )
            ray.init(
                address=self.spec.cluster_address,
                ignore_reinit_error=True,
                logging_level=logging.INFO,
            )
            logger.info("Successfully connected to Ray cluster driver node.")

    def create_gpu_placement_group(self, timeout_seconds: int = 120) -> PlacementGroup:
        """
        Creates worker bundle placement group matching GPU/CPU hardware requirements.
        """
        self.initialize_cluster_connection()

        bundles: List[Dict[str, Union[int, float]]] = [
            {
                "CPU": float(self.spec.cpus_per_worker),
                "GPU": float(self.spec.gpus_per_worker),
            }
            for _ in range(self.spec.num_workers)
        ]

        logger.info(
            f"Creating Ray PlacementGroup ({self.spec.num_workers} bundles, "
            f"strategy={self.spec.placement_strategy})..."
        )

        pg = placement_group(bundles, strategy=self.spec.placement_strategy)

        # Wait for placement group resource allocation
        ready = ray.get(pg.ready(), timeout=timeout_seconds)
        if not ready:
            raise TimeoutError(
                f"Failed to reserve placement group resources within {timeout_seconds}s"
            )

        self._current_placement_group = pg
        logger.info(f"PlacementGroup '{pg.id.hex()}' successfully created and READY.")
        return pg

    def submit_job(
        self,
        entrypoint: str,
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Submits an asynchronous distributed training job to the Ray cluster using JobSubmissionClient.
        """
        if self._job_client is None:
            self._job_client = JobSubmissionClient(self.spec.dashboard_url)

        runtime_env = {
            "pip": self.spec.pip_dependencies,
            "env_vars": {**dict(os.environ), **self.spec.env_vars},
        }

        logger.info(f"Submitting Ray Job with entrypoint command: '{entrypoint}'")
        submitted_id = self._job_client.submit_job(
            entrypoint=entrypoint,
            submission_id=job_id,
            runtime_env=runtime_env,
            metadata=metadata or {"framework": "VortexGrid"},
        )

        logger.info(
            f"Ray Job successfully submitted with Submission ID: '{submitted_id}'"
        )
        return submitted_id

    def wait_for_job_completion(
        self,
        job_id: str,
        poll_interval_seconds: int = 5,
        timeout_seconds: Optional[int] = None,
    ) -> JobStatus:
        """Polls submitted job status until terminal state (SUCCEEDED, FAILED, STOPPED)."""
        if self._job_client is None:
            self._job_client = JobSubmissionClient(self.spec.dashboard_url)

        start_time = time.time()
        logger.info(f"Polling status for Ray Job ID: '{job_id}'...")

        while True:
            status = self._job_client.get_job_status(job_id)
            logger.debug(f"Job '{job_id}' current status: {status}")

            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED):
                logger.info(f"Job '{job_id}' reached terminal state: {status}")
                return status

            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                raise TimeoutError(
                    f"Job '{job_id}' exceeded timeout of {timeout_seconds}s"
                )

            time.sleep(poll_interval_seconds)

    def shutdown(self) -> None:
        """Disconnects Ray cluster session."""
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray cluster driver session closed.")
