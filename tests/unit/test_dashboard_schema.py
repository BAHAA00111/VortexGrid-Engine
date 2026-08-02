import json
from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.mark.unit
def test_grafana_gpu_telemetry_json_parsing() -> None:
    # Check both possible paths (dashboards/ or configs/dashboards/)
    candidates = [
        Path("dashboards/grafana_gpu_telemetry.json"),
        Path("configs/dashboards/grafana_gpu_telemetry.json"),
    ]

    dashboard_file = next((p for p in candidates if p.exists()), None)
    assert dashboard_file is not None, "Missing Grafana telemetry JSON configuration file."

    with open(dashboard_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    assert isinstance(raw_data, dict), "Grafana dashboard must parse into a JSON dictionary"
    dashboard: Dict[str, Any] = raw_data

    assert dashboard.get("uid") == "vortexgrid-gpu-telemetry"
    assert "panels" in dashboard
    assert isinstance(dashboard["panels"], list)

    # Validate template variable bindings
    templating = dashboard.get("templating", {})
    assert isinstance(templating, dict)
    vars_list = templating.get("list", [])
    assert isinstance(vars_list, list)

    var_names = [v.get("name") for v in vars_list if isinstance(v, dict)]
    assert "DS_PROMETHEUS" in var_names
    assert "hostname" in var_names