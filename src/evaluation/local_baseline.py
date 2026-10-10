"""Load the current baseline's FACE component evidence for FACE-only comparisons.

The complete FACE + keypoint definition is registry.baseline_pipeline. Legacy
top-level fields retain the detector's evaluated contract; they do not certify
the new combined pipeline or change the deployed keypoint runtime.
"""
from pathlib import Path
import hashlib
import json

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs/model_registry/local_baseline.yaml"


def load_local_baseline(registry=REGISTRY, *, project_root=ROOT, verify=True):
    spec = yaml.safe_load(Path(registry).read_text())
    if spec.get("role") != "current_local_baseline":
        raise ValueError("Expected current local baseline registry")
    deployment = spec["deployment"]
    if (deployment["topology"] != "non-cascade"
            or deployment["stages"] != ["full-image FACE/ellipse"]
            or deployment["capabilities"] != ["face", "ellipse"]):
        raise ValueError("Unsupported local baseline topology/capabilities")
    if spec["operating_points"]["selected_on_split"] != "val":
        raise ValueError("Local baseline thresholds must be validation selected")
    for key in ("config", "checkpoint"):
        spec[key] = {**spec[key], "path": str(Path(project_root) / spec[key]["path"])}
    if verify:
        assets = [spec["config"], spec["checkpoint"]]
        for split in ("val", "test"):
            assets.extend(spec["evaluations"][split].values())
        for asset in assets:
            path = Path(project_root) / asset["path"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != asset["sha256"]:
                raise ValueError(f"Local baseline asset changed: {path}")
        for split in ("val", "test"):
            evidence = spec["evaluations"][split]
            marker = json.loads((Path(project_root) / evidence["success.json"]["path"]).read_text())
            if (marker.get("status") != "success" or not marker.get("inventory_verified")
                    or not marker.get("assertions") or not all(marker["assertions"].values())
                    or marker.get("metrics_sha256") != evidence["metrics.json"]["sha256"]):
                raise ValueError(f"Local baseline evaluation failed: {split}")
    return spec


def protect_historical_comparison(output_dir, baseline_id):
    """Never overwrite a completed report by silently replacing its comparator."""
    metrics = Path(output_dir) / "metrics.json"
    if metrics.exists():
        previous = json.loads(metrics.read_text()).get("primary_local_face_bbox", {})
        if previous.get("current_local_baseline_id") != baseline_id:
            raise ValueError("Historical baseline comparison exists; use a new output directory")
