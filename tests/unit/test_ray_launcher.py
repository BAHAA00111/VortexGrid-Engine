from pathlib import Path
import pytest

from vortexgrid.launcher.ray_launcher import RayClusterSpec, RayLauncher, HAS_RAY


@pytest.mark.unit
def test_ray_cluster_spec_manifest_loading(tmp_path: Path):
    manifest_file = tmp_path / "ray_config.yaml"
    manifest_file.write_text(
        """
cluster:
  address: "127.0.0.1:6379"
  dashboard_url: "http://127.0.0.1:8265"
worker_group:
  num_workers: 2
  gpus_per_worker: 1
  cpus_per_worker: 4
  placement_strategy: "STRICT_SPREAD"
"""
    )

    spec = RayClusterSpec.from_manifest(manifest_file)
    assert spec.cluster_address == "127.0.0.1:6379"
    assert spec.num_workers == 2
    assert spec.cpus_per_worker == 4
    assert spec.placement_strategy == "STRICT_SPREAD"


@pytest.mark.unit
def test_ray_launcher_initialization():
    if not HAS_RAY:
        pytest.skip("Ray package not installed.")

    spec = RayClusterSpec(num_workers=1, gpus_per_worker=0, cpus_per_worker=1)
    launcher = RayLauncher(spec=spec)
    assert launcher.spec.num_workers == 1