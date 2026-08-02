from pathlib import Path
import pytest
import yaml


@pytest.mark.unit
def test_k8s_manifests_syntax() -> None:
    manifest_paths = [
        Path("manifests/k8s/base/custom-metrics-exporter.yaml"),
        Path("manifests/k8s/base/prometheus-service.yaml"),
        Path("manifests/k8s/base/ray-cluster.yaml"),
        Path("manifests/k8s/jobs/auto-recovery-cron.yaml"),
    ]

    for filepath in manifest_paths:
        assert filepath.exists(), f"Missing manifest file: {filepath}"

        with open(filepath, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))

        assert len(docs) > 0, f"Empty manifest file: {filepath}"

        for doc in docs:
            if doc is None:
                continue
            assert isinstance(doc, dict), f"Document in {filepath} must parse to a dict"
            assert "apiVersion" in doc
            assert "kind" in doc
            assert "metadata" in doc
