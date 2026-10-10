"""Auditable initial-state identity and implementation acceptance helpers."""
import hashlib
import json
from pathlib import Path


def common_state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith("messenger_recovery.reference_memory.") or key.startswith("reference_memory."):
            continue
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def implementation_hashes():
    root = Path(__file__).resolve().parents[2]
    files = ["keypoint_comparison1/landmark_detection/" + name + ".py" for name in (
        "continuous_recovery", "continuous_teacher", "recovery_reference", "occlusion_materials", "occlusion_policy", "real_occlusion", "real_recovery_metrics",
        "model", "dataset", "config", "training", "recovery_acceptance", "losses", "metrics", "evaluation", "pipeline")]
    files += ["scripts/run_continuous_recovery_gate.py", "scripts/train_continuous_recovery.py",
              "scripts/run_multiscale_query_refinement_gate.py", "scripts/audit_occlusion_distribution.py",
              "scripts/opaque_boundary_gradient.py", "scripts/prepare_real_occlusion_bank.py",
              "scripts/audit_real_mixed_occlusion.py", "scripts/evaluate_real_mixed_recovery.py"]
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}


def verify_codino_evaluation(root):
    directory = Path(root) / "outputs/experiments/codino_official_sfp_ellipse/evaluation/test/baseline_aligned"
    marker = json.loads((directory / "success.json").read_text())
    if (marker.get("status") != "success" or marker.get("architecture_id") != "codino_official_sfp_ellipse"
            or marker.get("split") != "test" or marker.get("inventory_verified") is not True):
        raise RuntimeError("Co-DINO final evaluation has not passed its declared inventory")
    for name, key in (("metrics.json", "metrics_sha256"), ("face_threshold_selection.json", "face_threshold_selection_sha256")):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != marker.get(key):
            raise RuntimeError(f"Co-DINO evaluation artifact changed: {name}")
    return marker
