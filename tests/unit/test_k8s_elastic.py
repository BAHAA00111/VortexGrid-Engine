from pathlib import Path
import pytest

from vortexgrid.launcher.k8s_elastic import (
    HAS_K8S,
    K8sElasticJobSpec,
    K8sElasticLauncher,
)


@pytest.mark.unit
def test_k8s_elastic_job_spec_manifest_loading(tmp_path: Path):
    manifest_file = tmp_path / "k8s_job.yaml"
    manifest_file.write_text("""
metadata:
  name: "test-elastic-job"
  namespace: "vortexgrid-prod"
spec:
  minReplicas: 4
  maxReplicas: 16
  replicas: 8
  gpusPerReplica: 2
""")

    spec = K8sElasticJobSpec.from_manifest(manifest_file)
    assert spec.job_name == "test-elastic-job"
    assert spec.namespace == "vortexgrid-prod"
    assert spec.min_replicas == 4
    assert spec.max_replicas == 16
    assert spec.target_replicas == 8
    assert spec.gpus_per_replica == 2


@pytest.mark.unit
def test_k8s_pytorchjob_manifest_generation():
    if not HAS_K8S:
        pytest.skip("Kubernetes Python package not installed.")

    spec = K8sElasticJobSpec(
        job_name="unit-test-job",
        namespace="testing",
        min_replicas=2,
        max_replicas=4,
        target_replicas=2,
    )

    # Disable network API connection initialization for unit test isolation
    launcher = K8sElasticLauncher.__new__(K8sElasticLauncher)
    launcher.spec = spec

    manifest = launcher.generate_pytorchjob_manifest()

    assert manifest["apiVersion"] == "kubeflow.org/v1"
    assert manifest["kind"] == "PyTorchJob"
    assert manifest["metadata"]["name"] == "unit-test-job"
    assert manifest["spec"]["elasticPolicy"]["minReplicas"] == 2
    assert manifest["spec"]["elasticPolicy"]["maxReplicas"] == 4
