"""
Kubernetes Elastic Job & KubeRay Launcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provides cloud-native deployment, dynamic CRD templating, zero-downtime elastic
autoscaling, and lifecycle management for PyTorchJob and KubeRay operators.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from vortexgrid import logger

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    HAS_K8S = True
except ImportError:
    HAS_K8S = False


@dataclass
class K8sElasticJobSpec:
    """Dataclass holding Kubernetes PyTorchJob / KubeRay CRD configurations."""

    job_name: str = "vortexgrid-elastic-job"
    namespace: str = "default"
    image: str = "vortexgrid/cuda-engine:latest"
    min_replicas: int = 2
    max_replicas: int = 8
    target_replicas: int = 4
    gpus_per_replica: int = 1
    cpus_per_replica: int = 8
    memory_per_replica: str = "32Gi"
    crd_group: str = "kubeflow.org"
    crd_version: str = "v1"
    crd_plural: str = "pytorchjobs"
    env_vars: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, manifest_path: Union[str, Path]) -> K8sElasticJobSpec:
        """Parses Kubernetes YAML manifest to construct K8sElasticJobSpec."""
        path = Path(manifest_path)
        if not path.exists():
            logger.warning(
                f"K8s manifest '{manifest_path}' not found. Falling back to default spec."
            )
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        manifest: Dict[str, Any] = raw if isinstance(raw, dict) else {}
        metadata: Dict[str, Any] = (
            manifest.get("metadata", {})
            if isinstance(manifest.get("metadata"), dict)
            else {}
        )
        spec: Dict[str, Any] = (
            manifest.get("spec", {})
            if isinstance(manifest.get("spec"), dict)
            else {}
        )

        return cls(
            job_name=str(metadata.get("name", "vortexgrid-elastic-job")),
            namespace=str(metadata.get("namespace", "default")),
            gpus_per_replica=int(spec.get("gpusPerReplica", 1)),
            min_replicas=int(spec.get("minReplicas", 2)),
            max_replicas=int(spec.get("maxReplicas", 8)),
            target_replicas=int(spec.get("replicas", 4)),
        )


class K8sElasticLauncher:
    """
    Kubernetes Orchestrator managing PyTorchJob and KubeRay CRDs with elastic scaling.
    """

    def __init__(
        self,
        spec: Optional[K8sElasticJobSpec] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        in_cluster: bool = False,
    ) -> None:
        if not HAS_K8S:
            raise RuntimeError(
                "Kubernetes SDK is not installed. Install via 'pip install kubernetes' to use K8sElasticLauncher."
            )

        if manifest_path is not None:
            self.spec = K8sElasticJobSpec.from_manifest(manifest_path)
        else:
            self.spec = spec or K8sElasticJobSpec()

        # Initialize Kubernetes API clients
        try:
            if in_cluster:
                config.load_incluster_config()
            else:
                config.load_kube_config()
            logger.info("Kubernetes API client authenticated successfully.")
        except Exception as e:
            logger.warning(f"Could not load Kubernetes configuration: {str(e)}")

        self.custom_api = client.CustomObjectsApi()
        self.core_api = client.CoreV1Api()

    def generate_pytorchjob_manifest(self) -> Dict[str, Any]:
        """Generates dynamic Kubeflow PyTorchJob custom object manifest."""
        return {
            "apiVersion": f"{self.spec.crd_group}/{self.spec.crd_version}",
            "kind": "PyTorchJob",
            "metadata": {
                "name": self.spec.job_name,
                "namespace": self.spec.namespace,
                "labels": {
                    "app.kubernetes.io/name": "vortexgrid",
                    "vortexgrid/elastic": "true",
                },
            },
            "spec": {
                "elasticPolicy": {
                    "minReplicas": self.spec.min_replicas,
                    "maxReplicas": self.spec.max_replicas,
                    "rdzvBackend": "c10d",
                },
                "pytorchReplicaSpecs": {
                    "Worker": {
                        "replicas": self.spec.target_replicas,
                        "restartPolicy": "OnFailure",
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "pytorch",
                                        "image": self.spec.image,
                                        "command": [
                                            "python3",
                                            "-m",
                                            "vortexgrid.scripts.run_distributed_train",
                                        ],
                                        "env": [
                                            {"name": k, "value": v}
                                            for k, v in self.spec.env_vars.items()
                                        ],
                                        "resources": {
                                            "limits": {
                                                "nvidia.com/gpu": str(
                                                    self.spec.gpus_per_replica
                                                ),
                                                "cpu": str(self.spec.cpus_per_replica),
                                                "memory": self.spec.memory_per_replica,
                                            },
                                            "requests": {
                                                "cpu": str(
                                                    math.ceil(
                                                        self.spec.cpus_per_replica / 2
                                                    )
                                                ),
                                                "memory": "16Gi",
                                            },
                                        },
                                    }
                                ]
                            }
                        },
                    }
                },
            },
        }

    def deploy_elastic_job(self) -> Dict[str, Any]:
        """Deploys or updates elastic PyTorchJob CRD manifest on Kubernetes."""
        body = self.generate_pytorchjob_manifest()

        try:
            # Check if job already exists
            existing = self.custom_api.get_namespaced_custom_object(
                group=self.spec.crd_group,
                version=self.spec.crd_version,
                namespace=self.spec.namespace,
                plural=self.spec.crd_plural,
                name=self.spec.job_name,
            )
            logger.info(
                f"Job '{self.spec.job_name}' exists. Performing zero-downtime elastic patch update..."
            )
            response = self.custom_api.patch_namespaced_custom_object(
                group=self.spec.crd_group,
                version=self.spec.crd_version,
                namespace=self.spec.namespace,
                plural=self.spec.crd_plural,
                name=self.spec.job_name,
                body=body,
            )
        except ApiException as e:
            if e.status == 404:
                logger.info(
                    f"Creating new PyTorchJob '{self.spec.job_name}' in namespace '{self.spec.namespace}'..."
                )
                response = self.custom_api.create_namespaced_custom_object(
                    group=self.spec.crd_group,
                    version=self.spec.crd_version,
                    namespace=self.spec.namespace,
                    plural=self.spec.crd_plural,
                    body=body,
                )
            else:
                logger.error(f"Kubernetes API Deployment error: {str(e)}")
                raise e

        return response if isinstance(response, dict) else {}

    def scale_worker_replicas(self, target_replicas: int) -> Dict[str, Any]:
        """
        Dynamically resizes worker replica count within set min/max boundaries.
        """
        if not (self.spec.min_replicas <= target_replicas <= self.spec.max_replicas):
            raise ValueError(
                f"Target replicas {target_replicas} out of bounds "
                f"[{self.spec.min_replicas}, {self.spec.max_replicas}]"
            )

        logger.info(
            f"Elastically scaling PyTorchJob '{self.spec.job_name}' worker count to {target_replicas}..."
        )

        patch_body = {
            "spec": {
                "pytorchReplicaSpecs": {
                    "Worker": {
                        "replicas": target_replicas,
                    }
                }
            }
        }

        response = self.custom_api.patch_namespaced_custom_object(
            group=self.spec.crd_group,
            version=self.spec.crd_version,
            namespace=self.spec.namespace,
            plural=self.spec.crd_plural,
            name=self.spec.job_name,
            body=patch_body,
        )

        self.spec.target_replicas = target_replicas
        logger.info(f"Successfully scaled '{self.spec.job_name}' to {target_replicas} replicas.")
        return response if isinstance(response, dict) else {}

    def wait_for_job_completion(
        self,
        poll_interval_seconds: int = 10,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        """Polls Kubernetes Custom Object state until Succeeded, Failed, or Timed Out."""
        start_time = time.time()
        logger.info(f"Monitoring lifecycle for Kubernetes Job '{self.spec.job_name}'...")

        while True:
            try:
                job = self.custom_api.get_namespaced_custom_object(
                    group=self.spec.crd_group,
                    version=self.spec.crd_version,
                    namespace=self.spec.namespace,
                    plural=self.spec.crd_plural,
                    name=self.spec.job_name,
                )

                # Safely parse nested dict structure without triggering Pyright NoneType attribute errors
                job_dict: Dict[str, Any] = job if isinstance(job, dict) else {}
                status_raw = job_dict.get("status")
                status_dict: Dict[str, Any] = status_raw if isinstance(status_raw, dict) else {}

                conditions_raw = status_dict.get("conditions")
                conditions: List[Any] = conditions_raw if isinstance(conditions_raw, list) else []

                for cond in conditions:
                    if isinstance(cond, dict):
                        cond_type = str(cond.get("type", ""))
                        cond_status = str(cond.get("status", ""))

                        if cond_type == "Succeeded" and cond_status == "True":
                            logger.info(f"Job '{self.spec.job_name}' completed successfully.")
                            return "Succeeded"
                        if cond_type == "Failed" and cond_status == "True":
                            logger.error(f"Job '{self.spec.job_name}' execution failed.")
                            return "Failed"

            except ApiException as e:
                logger.warning(f"Error querying custom object status: {str(e)}")

            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                raise TimeoutError(
                    f"Job '{self.spec.job_name}' exceeded timeout of {timeout_seconds}s"
                )

            time.sleep(poll_interval_seconds)

    def delete_job(self) -> None:
        """Deletes custom object PyTorchJob deployment from cluster."""
        try:
            self.custom_api.delete_namespaced_custom_object(
                group=self.spec.crd_group,
                version=self.spec.crd_version,
                namespace=self.spec.namespace,
                plural=self.spec.crd_plural,
                name=self.spec.job_name,
            )
            logger.info(f"Successfully deleted Kubernetes Job '{self.spec.job_name}'.")
        except ApiException as e:
            logger.error(f"Failed to delete Kubernetes Job '{self.spec.job_name}': {str(e)}")