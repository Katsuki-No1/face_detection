from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import shlex
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.config_utils import load_yaml_config


LOGGER = logging.getLogger("yolo26_pose")
SPLITS = ("train", "val", "test")
VARIANTS = ("s", "m")


class PipelineError(RuntimeError):
    """Raised when the production experiment contract is not satisfied."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetSpec:
    version: str
    yaml_path: Path
    root: Path
    coco_annotations: dict[str, Path]
    expected_coco_sha256: dict[str, str]
    expected_images: dict[str, int]
    expected_category_id: int
    expected_names: dict[int, str]
    expected_kpt_shape: tuple[int, int]
    expected_flip_idx: tuple[int, ...]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    official_checkpoint: str
    pretrained_weights: Path
    expected_scale: str
    batch: int
    run_name: str
    official_coco_reference: dict[str, Any]


@dataclass(frozen=True)
class OutputSpec:
    pretrained_dir: Path
    train_project: Path
    evaluation_root: Path
    preflight_root: Path


@dataclass(frozen=True)
class WandbSpec:
    enabled: bool
    entity: str
    project: str
    group: str
    mode: str


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    project_root: Path
    experiment_name: str
    dataset: DatasetSpec
    models: dict[str, ModelSpec]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    selection: dict[str, Any]
    outputs: OutputSpec
    runtime: dict[str, Any]
    wandb: WandbSpec
    raw: dict[str, Any]

    def resolved_dict(self) -> dict[str, Any]:
        result = deepcopy(self.raw)
        result["project_root"] = str(self.project_root)
        result["dataset"]["yaml"] = str(self.dataset.yaml_path)
        result["dataset"]["root"] = str(self.dataset.root)
        result["dataset"]["coco_annotations"] = _plain(self.dataset.coco_annotations)
        for key, model in self.models.items():
            result["models"][key]["pretrained_weights"] = str(model.pretrained_weights)
        result["outputs"] = _plain(
            {
                "pretrained_dir": self.outputs.pretrained_dir,
                "train_project": self.outputs.train_project,
                "evaluation_root": self.outputs.evaluation_root,
                "preflight_root": self.outputs.preflight_root,
            }
        )
        result["wandb"] = _plain(self.wandb.__dict__)
        return result


def validate_production_training(training: Mapping[str, Any]) -> None:
    if training.get("production_mode") is not True:
        raise PipelineError("training.production_mode must be true")
    epochs = int(training.get("epochs", 0))
    minimum_epochs = int(training.get("minimum_epochs", 120))
    if epochs < minimum_epochs:
        raise PipelineError(f"Production training requires at least {minimum_epochs} epochs, got {epochs}")
    if float(training.get("fraction", 0.0)) != 1.0:
        raise PipelineError("Production training requires fraction=1.0; dataset subsampling is forbidden")
    if training.get("pretrained") is not True:
        raise PipelineError("Production training must start from the official pretrained checkpoint")
    if int(training.get("imgsz", 0)) < 960:
        raise PipelineError("Production face-pose comparison requires imgsz >= 960")
    if training.get("val") is not True:
        raise PipelineError("Validation must remain enabled")
    if int(training.get("save_period", 0)) <= 0:
        raise PipelineError("save_period must be positive so a val-selected checkpoint remains available")
    forbidden = {"max_batches", "max_train_images", "debug_subset", "limit_train_batches"}
    present = sorted(key for key in forbidden if key in training)
    if present:
        raise PipelineError(f"Demo/subset training keys are forbidden: {present}")


def validate_wandb_config(value: Mapping[str, Any]) -> None:
    if value.get("enabled") is not True:
        raise PipelineError("wandb.enabled must be true for the unfreeze-and-later production runs")
    for key in ("entity", "project", "group"):
        if not str(value.get(key, "")).strip():
            raise PipelineError(f"wandb.{key} must be a non-empty string")
    mode = str(value.get("mode", "online"))
    if mode not in {"online", "offline", "disabled"}:
        raise PipelineError(f"wandb.mode must be online, offline, or disabled; got {mode!r}")
    if mode != "online":
        raise PipelineError("Production comparison runs require wandb.mode=online")


def _int_key_dict(value: Mapping[Any, Any]) -> dict[int, str]:
    return {int(key): str(item) for key, item in value.items()}


def load_pipeline_config(
    config_path: str | Path,
    *,
    project_root_override: str | Path | None = None,
) -> PipelineConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise PipelineError(f"Config does not exist: {path}")
    raw_value = load_yaml_config(path)
    if not isinstance(raw_value, dict):
        raise PipelineError(f"Config must contain a YAML mapping: {path}")
    raw: dict[str, Any] = raw_value
    if int(raw.get("schema_version", 0)) != 1:
        raise PipelineError("Unsupported or missing schema_version; expected 1")

    if project_root_override is not None:
        project_root = Path(project_root_override).expanduser().resolve()
    else:
        configured_root = Path(str(raw.get("project_root", "."))).expanduser()
        project_root = configured_root.resolve() if configured_root.is_absolute() else (_repo_root() / configured_root).resolve()

    dataset_raw = raw.get("dataset")
    models_raw = raw.get("models")
    outputs_raw = raw.get("outputs")
    if not isinstance(dataset_raw, dict) or not isinstance(models_raw, dict) or not isinstance(outputs_raw, dict):
        raise PipelineError("dataset, models, and outputs mappings are required")

    coco_raw = dataset_raw.get("coco_annotations", {})
    sha_raw = dataset_raw.get("expected_coco_sha256", {})
    count_raw = dataset_raw.get("expected_images", {})
    if any(split not in coco_raw or split not in sha_raw or split not in count_raw for split in SPLITS):
        raise PipelineError("Dataset must define train/val/test annotation paths, hashes, and expected image counts")
    dataset_root = _resolve(project_root, dataset_raw["root"])
    dataset_version = str(dataset_raw.get("version") or dataset_root.name).strip()
    if not dataset_version:
        raise PipelineError("dataset.version could not be resolved")
    dataset = DatasetSpec(
        version=dataset_version,
        yaml_path=_resolve(project_root, dataset_raw["yaml"]),
        root=dataset_root,
        coco_annotations={split: _resolve(project_root, coco_raw[split]) for split in SPLITS},
        expected_coco_sha256={split: str(sha_raw[split]).lower() for split in SPLITS},
        expected_images={split: int(count_raw[split]) for split in SPLITS},
        expected_category_id=int(dataset_raw["expected_category_id"]),
        expected_names=_int_key_dict(dataset_raw["expected_names"]),
        expected_kpt_shape=tuple(int(item) for item in dataset_raw["expected_kpt_shape"]),  # type: ignore[arg-type]
        expected_flip_idx=tuple(int(item) for item in dataset_raw["expected_flip_idx"]),
    )
    if dataset.expected_kpt_shape != (5, 3):
        raise PipelineError(f"This pipeline requires the five-point face pose schema [5, 3], got {dataset.expected_kpt_shape}")

    models: dict[str, ModelSpec] = {}
    if set(models_raw) != set(VARIANTS):
        raise PipelineError(f"Exactly the YOLO26 Pose variants {list(VARIANTS)} must be configured")
    for key in VARIANTS:
        value = models_raw[key]
        if not isinstance(value, dict):
            raise PipelineError(f"models.{key} must be a mapping")
        expected_checkpoint = f"yolo26{key}-pose.pt"
        if value.get("official_checkpoint") != expected_checkpoint:
            raise PipelineError(f"models.{key}.official_checkpoint must be {expected_checkpoint}")
        if str(value.get("expected_scale")) != key:
            raise PipelineError(f"models.{key}.expected_scale must be {key}")
        batch = int(value.get("batch", 0))
        if batch <= 0:
            raise PipelineError(f"models.{key}.batch must be positive")
        models[key] = ModelSpec(
            key=key,
            display_name=str(value["display_name"]),
            official_checkpoint=str(value["official_checkpoint"]),
            pretrained_weights=_resolve(project_root, value["pretrained_weights"]),
            expected_scale=str(value["expected_scale"]),
            batch=batch,
            run_name=str(value["run_name"]),
            official_coco_reference=dict(value.get("official_coco_reference", {})),
        )

    training = dict(raw.get("training", {}))
    validate_production_training(training)
    wandb_raw = raw.get("wandb", {})
    if not isinstance(wandb_raw, dict):
        raise PipelineError("wandb must be a mapping")
    validate_wandb_config(wandb_raw)
    wandb = WandbSpec(
        enabled=bool(wandb_raw["enabled"]),
        entity=str(wandb_raw["entity"]),
        project=str(wandb_raw["project"]),
        group=str(wandb_raw["group"]),
        mode=str(wandb_raw.get("mode", "online")),
    )
    outputs = OutputSpec(
        pretrained_dir=_resolve(project_root, outputs_raw["pretrained_dir"]),
        train_project=_resolve(project_root, outputs_raw["train_project"]),
        evaluation_root=_resolve(project_root, outputs_raw["evaluation_root"]),
        preflight_root=_resolve(project_root, outputs_raw["preflight_root"]),
    )
    if any(model.pretrained_weights.parent != outputs.pretrained_dir for model in models.values()):
        raise PipelineError("All pretrained weights must be contained directly in outputs.pretrained_dir")

    return PipelineConfig(
        config_path=path,
        project_root=project_root,
        experiment_name=str(raw["experiment_name"]),
        dataset=dataset,
        models=models,
        training=training,
        evaluation=dict(raw.get("evaluation", {})),
        selection=dict(raw.get("selection", {})),
        outputs=outputs,
        runtime=dict(raw.get("runtime", {})),
        wandb=wandb,
        raw=raw,
    )


def select_model_keys(value: str | Sequence[str]) -> list[str]:
    items = [value] if isinstance(value, str) else list(value)
    tokens: list[str] = []
    for item in items:
        tokens.extend(part.strip().lower() for part in str(item).split(",") if part.strip())
    if not tokens or "all" in tokens:
        return list(VARIANTS)
    unknown = sorted(set(tokens) - set(VARIANTS))
    if unknown:
        raise PipelineError(f"Unknown model variants: {unknown}; choose s, m, or all")
    return [key for key in VARIANTS if key in tokens]


def _version_tuple(value: str) -> tuple[int, ...]:
    components: list[int] = []
    for part in value.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        components.append(int(digits))
    return tuple(components)


def _run_command(command: Sequence[str], *, cwd: Path, log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(list(command))}\n\n")
        log_file.flush()
        result = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise PipelineError(f"Command failed with exit code {result.returncode}; see {log_path}")
    return elapsed


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_resolved_config(config: PipelineConfig) -> Path:
    path = config.outputs.preflight_root / "config_resolved.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.resolved_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _audit_dataset(config: PipelineConfig) -> dict[str, Any]:
    dataset = config.dataset
    for required in (dataset.yaml_path, dataset.root, *dataset.coco_annotations.values()):
        if not required.exists():
            raise PipelineError(f"Required dataset path does not exist: {required}")

    dataset_yaml = yaml.safe_load(dataset.yaml_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_yaml, dict):
        raise PipelineError(f"Dataset YAML must contain a mapping: {dataset.yaml_path}")
    actual_names = _int_key_dict(dataset_yaml.get("names", {}))
    actual_kpt_shape = tuple(int(item) for item in dataset_yaml.get("kpt_shape", []))
    actual_flip_idx = tuple(int(item) for item in dataset_yaml.get("flip_idx", []))
    if actual_names != dataset.expected_names:
        raise PipelineError(f"Dataset class names changed: expected {dataset.expected_names}, got {actual_names}")
    if actual_kpt_shape != dataset.expected_kpt_shape:
        raise PipelineError(f"Dataset keypoint shape changed: expected {dataset.expected_kpt_shape}, got {actual_kpt_shape}")
    if actual_flip_idx != dataset.expected_flip_idx:
        raise PipelineError(f"Dataset flip_idx changed: expected {dataset.expected_flip_idx}, got {actual_flip_idx}")
    for split in SPLITS:
        if split not in dataset_yaml:
            raise PipelineError(f"Dataset YAML is missing the {split} split")

    hashes: dict[str, str] = {}
    for split, annotation_path in dataset.coco_annotations.items():
        hashes[split] = sha256_file(annotation_path)
        expected = dataset.expected_coco_sha256[split]
        if hashes[split] != expected:
            raise PipelineError(
                f"Fixed {split} annotation changed: expected SHA-256 {expected}, got {hashes[split]}"
            )

    checker_report = config.outputs.preflight_root / "yolo_label_audit.json"
    _run_command(
        [
            sys.executable,
            str(config.project_root / "scripts_YOLO" / "check_yolo_pose_dataset.py"),
            "--data-root",
            str(dataset.root),
            "--yaml",
            str(dataset.yaml_path),
            "--output",
            str(checker_report),
        ],
        cwd=config.project_root,
        log_path=config.outputs.preflight_root / "logs" / "dataset_label_audit.log",
    )
    label_audit = json.loads(checker_report.read_text(encoding="utf-8"))
    if label_audit.get("status") != "pass":
        raise PipelineError(f"YOLO label audit failed; see {checker_report}")
    for split in SPLITS:
        actual = int(label_audit["images_by_split"].get(split, -1))
        expected = dataset.expected_images[split]
        if actual != expected:
            raise PipelineError(f"Fixed {split} image count changed: expected {expected}, got {actual}")

    if str(config.project_root) not in sys.path:
        sys.path.insert(0, str(config.project_root))
    from src.data.coco_split_audit import audit_coco_splits

    split_audit = audit_coco_splits(
        dataset.root,
        annotation_paths=dataset.coco_annotations,
        expected_category_id=dataset.expected_category_id,
    )
    for split in SPLITS:
        actual = int(split_audit["splits"][split]["images"])
        expected = dataset.expected_images[split]
        if actual != expected:
            raise PipelineError(f"COCO {split} image count changed: expected {expected}, got {actual}")
    _write_json(config.outputs.preflight_root / "coco_split_audit.json", split_audit)
    return {
        "status": "pass",
        "dataset_yaml": str(dataset.yaml_path),
        "coco_sha256": hashes,
        "images_by_split": label_audit["images_by_split"],
        "labels_by_split": label_audit["labels_by_split"],
        "objects": label_audit["objects"],
        "kpt_shape": list(dataset.expected_kpt_shape),
        "split_leakage": split_audit["cross_split_overlap_counts"],
    }


def _download_weight(model: ModelSpec, allow_download: bool) -> Path:
    target = model.pretrained_weights
    if target.is_file():
        return target
    if not allow_download:
        raise PipelineError(
            f"Official checkpoint is missing: {target}. Re-run without --skip-weight-download or place "
            f"{model.official_checkpoint} there."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset

    LOGGER.info("Downloading official %s checkpoint to %s", model.display_name, target)
    downloaded = Path(attempt_download_asset(str(target))).resolve()
    if downloaded != target.resolve() and downloaded.is_file():
        raise PipelineError(f"Ultralytics resolved an unexpected checkpoint path: {downloaded} (wanted {target})")
    if not target.is_file():
        raise PipelineError(f"Ultralytics did not create the requested checkpoint: {target}")
    return target


def _verify_model_transfer(model: ModelSpec, dataset: DatasetSpec) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import PoseModel

    checkpoint = YOLO(str(model.pretrained_weights), task="pose")
    if checkpoint.task != "pose":
        raise PipelineError(f"{model.pretrained_weights} is not a pose checkpoint")
    source_model = checkpoint.model
    source_yaml = deepcopy(source_model.yaml)
    actual_scale = str(source_yaml.get("scale", ""))
    if actual_scale != model.expected_scale:
        raise PipelineError(f"{model.display_name} scale mismatch: expected {model.expected_scale}, got {actual_scale}")
    source_head = source_model.model[-1]
    source_kpt_shape = tuple(int(item) for item in getattr(source_head, "kpt_shape", ()))
    source_state = source_model.state_dict()

    adapted = PoseModel(
        source_yaml,
        ch=3,
        nc=len(dataset.expected_names),
        data_kpt_shape=list(dataset.expected_kpt_shape),
        verbose=False,
    )
    target_state = adapted.state_dict()
    compatible = {
        key
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    adapted.load(source_model, verbose=False)
    target_head = adapted.model[-1]
    target_kpt_shape = tuple(int(item) for item in getattr(target_head, "kpt_shape", ()))
    if target_kpt_shape != dataset.expected_kpt_shape:
        raise PipelineError(
            f"Face keypoint head adaptation failed for {model.display_name}: got {target_kpt_shape}"
        )
    if not bool(getattr(adapted, "end2end", False)):
        raise PipelineError(f"{model.display_name} did not retain YOLO26 end-to-end mode")

    report = {
        "status": "pass",
        "checkpoint": str(model.pretrained_weights),
        "checkpoint_sha256": sha256_file(model.pretrained_weights),
        "ultralytics_task": checkpoint.task,
        "scale": actual_scale,
        "source_kpt_shape": list(source_kpt_shape),
        "adapted_kpt_shape": list(target_kpt_shape),
        "classes": dataset.expected_names,
        "end2end": True,
        "source_parameters": sum(parameter.numel() for parameter in source_model.parameters()),
        "adapted_parameters": sum(parameter.numel() for parameter in adapted.parameters()),
        "compatible_state_tensors": len(compatible),
        "target_state_tensors": len(target_state),
        "compatible_state_tensor_fraction": len(compatible) / max(1, len(target_state)),
        "torch_version": torch.__version__,
    }
    del adapted, checkpoint, source_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _write_command_plan(config: PipelineConfig) -> None:
    script = config.project_root / "scripts_YOLO" / "yolo26_pose" / "run_yolo26_pose.py"
    common = [sys.executable, str(script), "--config", str(config.config_path)]
    train_commands = [
        shlex.join([*common, "--stage", "train", "--models", key, "--device", str(config.runtime.get("device", "0"))])
        for key in VARIANTS
    ]
    eval_commands = [
        shlex.join([*common, "--stage", "evaluate", "--models", key, "--device", str(config.runtime.get("device", "0"))])
        for key in VARIANTS
    ]
    (config.outputs.preflight_root / "train_commands.sh").write_text("\n".join(train_commands) + "\n", encoding="utf-8")
    (config.outputs.preflight_root / "evaluation_commands.sh").write_text(
        "\n".join([*eval_commands, shlex.join([*common, "--stage", "compare"])]) + "\n",
        encoding="utf-8",
    )


def _audit_wandb(config: PipelineConfig) -> dict[str, Any]:
    """Verify W&B authentication without exposing credentials or creating a run."""

    if not config.wandb.enabled:
        raise PipelineError("W&B must be enabled before starting production training")
    try:
        import wandb
    except ImportError as error:
        raise PipelineError("W&B logging is enabled, but wandb is not installed") from error

    report: dict[str, Any] = {
        "status": "pass",
        "version": wandb.__version__,
        "entity": config.wandb.entity,
        "project": config.wandb.project,
        "group": config.wandb.group,
        "mode": config.wandb.mode,
        "api_key_configured": False,
        "project_accessible": False,
    }
    if config.wandb.mode != "online":
        raise PipelineError("Production comparison runs require W&B online mode")
    try:
        api = wandb.Api(timeout=30)
        report["api_key_configured"] = bool(api.api_key)
        if not report["api_key_configured"]:
            raise PipelineError("W&B API key is not configured; run `wandb login` before training")
        project = api.project(config.wandb.project, entity=config.wandb.entity)
        report["project_accessible"] = bool(project is not None)
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(
            f"W&B authentication/project preflight failed ({type(error).__name__}): {error}"
        ) from error
    if not report["project_accessible"]:
        raise PipelineError(
            f"W&B project is not accessible: {config.wandb.entity}/{config.wandb.project}"
        )
    return report


def run_preflight(
    config: PipelineConfig,
    model_keys: Sequence[str],
    *,
    allow_weight_download: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    config.outputs.preflight_root.mkdir(parents=True, exist_ok=True)
    resolved_path = _write_resolved_config(config)
    LOGGER.info(
        "Auditing all %d fixed-dataset frames and split provenance for %s",
        sum(config.dataset.expected_images.values()),
        config.dataset.version,
    )
    dataset_report = _audit_dataset(config)

    try:
        import ultralytics
    except Exception as error:
        raise PipelineError("Ultralytics is required. Install requirements.txt in the active environment.") from error
    minimum_version = str(config.runtime.get("minimum_ultralytics_version", "8.4.0"))
    if _version_tuple(ultralytics.__version__) < _version_tuple(minimum_version):
        raise PipelineError(
            f"Ultralytics {minimum_version}+ is required for YOLO26; found {ultralytics.__version__}"
        )
    wandb_report = _audit_wandb(config)

    model_reports: dict[str, Any] = {}
    for index, key in enumerate(model_keys, start=1):
        model = config.models[key]
        LOGGER.info("[preflight:model] %d/%d %s", index, len(model_keys), model.display_name)
        _download_weight(model, allow_weight_download)
        model_reports[key] = _verify_model_transfer(model, config.dataset)

    _write_command_plan(config)
    report = {
        "status": "pass",
        "experiment_name": config.experiment_name,
        "config": str(config.config_path),
        "config_resolved": str(resolved_path),
        "ultralytics_version": ultralytics.__version__,
        "dataset": dataset_report,
        "models": model_reports,
        "wandb": wandb_report,
        "elapsed_seconds": time.monotonic() - started,
        "training_started": False,
    }
    _write_json(config.outputs.preflight_root / "preflight_report.json", report)
    LOGGER.info("Preflight passed in %.1fs; no training was started", report["elapsed_seconds"])
    return report


def _training_kwargs(config: PipelineConfig, model: ModelSpec, device: str) -> dict[str, Any]:
    excluded = {"production_mode", "minimum_epochs"}
    kwargs = {key: value for key, value in config.training.items() if key not in excluded and value is not None}
    kwargs.update(
        {
            "data": str(config.dataset.yaml_path),
            "batch": model.batch,
            "project": str(config.outputs.train_project),
            "name": model.run_name,
            "device": device,
            "exist_ok": False,
        }
    )
    return kwargs


def _configure_accelerator(config: PipelineConfig, device: str) -> dict[str, Any]:
    import torch

    uses_cuda = device not in {"cpu", "mps"} and torch.cuda.is_available()
    if config.training.get("production_mode") and not uses_cuda:
        raise PipelineError(
            f"Production training requires CUDA, but device={device!r} and cuda_available={torch.cuda.is_available()}"
        )
    tf32 = bool(config.runtime.get("enable_tf32", True)) and uses_cuda
    if uses_cuda:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    return {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tf32": tf32,
    }


def _wandb_run_id(config: PipelineConfig, model: ModelSpec) -> str:
    seed = int(config.training.get("seed", 0))
    identity = f"{config.experiment_name}:{model.run_name}:seed={seed}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _init_yolo_wandb_run(config: PipelineConfig, model: ModelSpec, *, resume: bool):
    try:
        import wandb
    except ImportError as error:
        raise PipelineError("W&B logging is enabled, but wandb is not installed") from error

    run_config = {
        "experiment_name": config.experiment_name,
        "model": model.display_name,
        "model_variant": model.key,
        "official_checkpoint": model.official_checkpoint,
        "dataset": config.dataset.version,
        "dataset_images": config.dataset.expected_images,
        "dataset_coco_sha256": config.dataset.expected_coco_sha256,
        "training": deepcopy(config.training),
        "batch": model.batch,
        "seed": int(config.training.get("seed", 0)),
    }
    local_dir = config.outputs.train_project / "_wandb"
    local_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        group=config.wandb.group,
        name=f"{model.run_name}-seed{run_config['seed']}",
        id=_wandb_run_id(config, model),
        resume="allow" if resume else "never",
        job_type="training",
        mode=config.wandb.mode,
        dir=str(local_dir),
        config=run_config,
        tags=["YOLO26", "pose", model.key, config.dataset.version, "production"],
    )
    LOGGER.info("W&B run started: %s", run.url or run.id)
    return run


def run_training(
    config: PipelineConfig,
    model_keys: Sequence[str],
    *,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    import ultralytics
    import wandb

    # The callback registry is constructed from this setting when YOLO is imported.
    # W&B itself is initialized explicitly below so project/group/run identity never
    # depends on Ultralytics' implicit directory-derived defaults.
    ultralytics.settings.update({"wandb": config.wandb.enabled})
    from ultralytics import YOLO
    from ultralytics.utils.callbacks import wb as ultralytics_wb

    if config.wandb.enabled and not ultralytics_wb.callbacks:
        raise PipelineError("Ultralytics W&B callbacks are unavailable although W&B is enabled")

    accelerator = _configure_accelerator(config, device)
    config.outputs.train_project.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for index, key in enumerate(model_keys, start=1):
        spec = config.models[key]
        run_dir = config.outputs.train_project / spec.run_name
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        LOGGER.info("[train] %d/%d %s", index, len(model_keys), spec.display_name)
        if best.is_file() and (run_dir / "results.csv").is_file() and not resume:
            LOGGER.info("Completed run exists; reusing %s", run_dir)
            summaries[key] = {"status": "reused", "run_dir": str(run_dir), "best_weights": str(best)}
            continue
        if run_dir.exists() and not resume:
            raise PipelineError(
                f"Partial or unrecognized run exists at {run_dir}. Use --resume if last.pt is valid; "
                "the pipeline will never overwrite it."
            )
        if resume:
            if not last.is_file():
                raise PipelineError(f"Cannot resume {spec.display_name}; missing {last}")
            model = YOLO(str(last), task="pose")
            launch = {"resume": True, "checkpoint": str(last), "accelerator": accelerator}
        else:
            if not spec.pretrained_weights.is_file():
                raise PipelineError(f"Missing preflighted official checkpoint: {spec.pretrained_weights}")
            model = YOLO(str(spec.pretrained_weights), task="pose")
            kwargs = _training_kwargs(config, spec, device)
            launch = {"resume": False, "checkpoint": str(spec.pretrained_weights), "kwargs": kwargs, "accelerator": accelerator}
        _write_json(config.outputs.preflight_root / f"train_launch_{key}.json", launch)
        wandb_run = _init_yolo_wandb_run(config, spec, resume=resume)
        wandb_metadata = {
            "entity": config.wandb.entity,
            "project": config.wandb.project,
            "group": config.wandb.group,
            "run_id": wandb_run.id,
            "run_url": wandb_run.url,
            "resume": resume,
        }
        launch["wandb"] = wandb_metadata
        _write_json(config.outputs.preflight_root / f"train_launch_{key}.json", launch)
        started = time.monotonic()
        try:
            if resume:
                model.train(resume=True)
            else:
                model.train(**kwargs)
        except Exception:
            if wandb.run is not None:
                wandb_run.finish(exit_code=1)
            raise
        if wandb.run is not None:
            # Ultralytics normally closes the run in on_train_end. Keep an explicit
            # fallback so a callback-version change cannot leave the run dangling.
            wandb_run.finish(exit_code=0)
        elapsed = time.monotonic() - started
        if not best.is_file():
            raise PipelineError(f"Training returned without best.pt: {best}")
        metadata = {
            "status": "complete",
            "model": spec.display_name,
            "run_dir": str(run_dir),
            "best_weights": str(best),
            "elapsed_seconds": elapsed,
            "config": str(config.config_path),
            "dataset_coco_sha256": config.dataset.expected_coco_sha256,
            "accelerator": accelerator,
            "wandb": wandb_metadata,
        }
        _write_json(run_dir / "yolo26_experiment_metadata.json", metadata)
        summaries[key] = metadata
    return summaries


def _best_weights(config: PipelineConfig, model: ModelSpec) -> Path:
    path = config.outputs.train_project / model.run_name / "weights" / "best.pt"
    if not path.is_file():
        raise PipelineError(f"Missing trained weights for {model.display_name}: {path}")
    return path


def run_evaluation(
    config: PipelineConfig,
    model_keys: Sequence[str],
    *,
    device: str,
    force: bool,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    eval_cfg = config.evaluation
    for index, key in enumerate(model_keys, start=1):
        model = config.models[key]
        weights = _best_weights(config, model)
        output_dir = config.outputs.evaluation_root / model.run_name
        metrics_summary = output_dir / "test_metrics" / "test_metrics_summary.json"
        LOGGER.info("[evaluate] %d/%d %s", index, len(model_keys), model.display_name)
        if metrics_summary.is_file() and not force:
            LOGGER.info("Evaluation exists; reusing %s", metrics_summary)
            summaries[key] = json.loads(metrics_summary.read_text(encoding="utf-8"))
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        logs = output_dir / "logs"
        val_dir = output_dir / "val_threshold"
        test_f1_dir = output_dir / "test_fixed_threshold"
        test_metrics_dir = output_dir / "test_metrics"
        common_f1 = [
            sys.executable,
            str(config.project_root / "scripts_YOLO" / "evaluate_yolo_pose_f1.py"),
            "--weights",
            str(weights),
            "--data-root",
            str(config.dataset.root),
            "--imgsz",
            str(eval_cfg["imgsz"]),
            "--device",
            device,
            "--batch-size",
            str(eval_cfg["batch_size"]),
            "--iou-threshold",
            str(eval_cfg["iou_threshold"]),
            "--prediction-conf-min",
            str(eval_cfg["prediction_conf_min"]),
        ]
        _run_command(
            [
                *common_f1,
                "--split",
                "val",
                "--thresholds",
                str(eval_cfg["thresholds"]),
                "--output-dir",
                str(val_dir),
                "--write-predictions",
            ],
            cwd=config.project_root,
            log_path=logs / "val_threshold_sweep.log",
        )
        best_val = json.loads((val_dir / "val_best_threshold.json").read_text(encoding="utf-8"))
        threshold = float(best_val["threshold"])
        baseline_annotation_root = (
            config.project_root
            / "outputs"
            / "dataset_cache"
            / "face_pose_drive_v2_balanced_811"
            / "annotations"
            / "face_dino_v1"
        )
        common_alignment = [
            sys.executable,
            str(config.project_root / "tools" / "evaluate_baseline_aligned_predictions.py"),
            "--format",
            "yolo_pose",
            "--architecture-id",
            f"{model.display_name}_face_pose_drive_v2_balanced_811",
            "--comparison-stage",
            "pose",
            "--config",
            str(config.config_path),
            "--checkpoint",
            str(weights),
            "--seed",
            str(int(config.training.get("seed", 0))),
            "--validation-face-threshold",
            str(threshold),
            "--validation-point-threshold",
            "0.70",
        ]
        _run_command(
            [
                *common_alignment,
                "--predictions",
                str(val_dir / "val_raw_predictions.jsonl"),
                "--annotations",
                str(baseline_annotation_root / "val.json"),
                "--split",
                "val",
                "--output-dir",
                str(output_dir / "evaluation" / "val" / "baseline_aligned"),
                "--source-metrics",
                str(val_dir / "val_best_threshold.json"),
            ],
            cwd=config.project_root,
            log_path=logs / "val_baseline_aligned.log",
        )
        _run_command(
            [
                *common_f1,
                "--split",
                "test",
                "--fixed-threshold",
                str(threshold),
                "--output-dir",
                str(test_f1_dir),
                "--write-predictions",
            ],
            cwd=config.project_root,
            log_path=logs / "test_fixed_threshold_f1.log",
        )
        metrics_command = [
            sys.executable,
            str(config.project_root / "scripts_YOLO" / "evaluate_yolo_pose_metrics.py"),
            "--weights",
            str(weights),
            "--data",
            str(config.dataset.yaml_path),
            "--data-root",
            str(config.dataset.root),
            "--split",
            "test",
            "--imgsz",
            str(eval_cfg["imgsz"]),
            "--device",
            device,
            "--batch-size",
            str(eval_cfg["batch_size"]),
            "--conf-threshold",
            str(threshold),
            "--prediction-conf-min",
            str(eval_cfg["prediction_conf_min"]),
            "--iou-threshold",
            str(eval_cfg["iou_threshold"]),
            "--keypoint-error-threshold",
            str(eval_cfg["keypoint_error_threshold"]),
            "--output-dir",
            str(test_metrics_dir),
            "--max-error-images",
            str(eval_cfg["max_error_images"]),
        ]
        if bool(eval_cfg.get("save_error_images", True)):
            metrics_command.append("--save-error-images")
        _run_command(metrics_command, cwd=config.project_root, log_path=logs / "test_metrics.log")
        _run_command(
            [
                *common_alignment,
                "--predictions",
                str(test_f1_dir / "test_raw_predictions.jsonl"),
                "--annotations",
                str(baseline_annotation_root / "test.json"),
                "--split",
                "test",
                "--output-dir",
                str(output_dir / "evaluation" / "test" / "baseline_aligned"),
                "--source-metrics",
                str(metrics_summary),
            ],
            cwd=config.project_root,
            log_path=logs / "test_baseline_aligned.log",
        )
        summary = json.loads(metrics_summary.read_text(encoding="utf-8"))
        summary["validation_selected_threshold"] = best_val
        summary["model"] = model.display_name
        _write_json(output_dir / "evaluation_summary.json", summary)
        summaries[key] = summary
    return summaries


def _nested_number(value: Mapping[str, Any], *keys: str) -> float | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    try:
        return float(current) if current is not None else None
    except (TypeError, ValueError):
        return None


def recommend_variant(rows: Mapping[str, Mapping[str, Any]], selection: Mapping[str, Any]) -> dict[str, Any]:
    if not all(key in rows for key in VARIANTS):
        return {
            "status": "pending",
            "recommended": str(selection.get("current_default", "s")),
            "reason": "Both YOLO26s and YOLO26m require results from the same fixed validation/test protocol.",
        }
    s_row, m_row = rows["s"], rows["m"]
    s_pose = _nested_number(s_row, "map", "pose", "map50_95")
    m_pose = _nested_number(m_row, "map", "pose", "map50_95")
    s_box = _nested_number(s_row, "map", "box", "map50_95")
    m_box = _nested_number(m_row, "map", "box", "map50_95")
    s_nme = _nested_number(s_row, "keypoints", "nme")
    m_nme = _nested_number(m_row, "keypoints", "nme")
    if None in {s_pose, m_pose, s_box, m_box, s_nme, m_nme}:
        return {
            "status": "pending",
            "recommended": str(selection.get("current_default", "s")),
            "reason": "One or more required bbox/pose/NME metrics are missing.",
        }
    pose_gain = float(m_pose) - float(s_pose)
    bbox_regression = float(s_box) - float(m_box)
    nme_regression = float(m_nme) - float(s_nme)
    gates = {
        "pose_gain": pose_gain >= float(selection.get("pose_map50_95_min_gain_for_m", 0.01)),
        "bbox_regression": bbox_regression <= float(selection.get("bbox_map50_95_max_regression_for_m", 0.005)),
        "nme_regression": nme_regression <= float(selection.get("nme_max_regression_for_m", 0.002)),
    }
    use_m = all(gates.values())
    return {
        "status": "complete",
        "recommended": "m" if use_m else "s",
        "pose_map50_95_gain_m_minus_s": pose_gain,
        "bbox_map50_95_regression_s_minus_m": bbox_regression,
        "nme_regression_m_minus_s": nme_regression,
        "gates": gates,
        "reason": (
            "YOLO26m cleared the configured quality-gain gates."
            if use_m
            else "YOLO26m did not clear every configured gain/regression gate; retain YOLO26s for this dataset."
        ),
    }


def _comparison_row(key: str, config: PipelineConfig, summary: Mapping[str, Any] | None) -> dict[str, Any]:
    model = config.models[key]
    row: dict[str, Any] = {
        "variant": key,
        "model": model.display_name,
        "batch": model.batch,
        "run_dir": str(config.outputs.train_project / model.run_name),
        "status": "complete" if summary else "pending",
        "official_coco_pose_map50_95_reference": model.official_coco_reference.get("pose_map50_95"),
        "official_coco_parameters_reference": model.official_coco_reference.get("parameters"),
        "official_coco_flops_g_reference": model.official_coco_reference.get("flops_g"),
    }
    if summary:
        row.update(
            {
                "test_box_map50_95": _nested_number(summary, "map", "box", "map50_95"),
                "test_pose_map50_95": _nested_number(summary, "map", "pose", "map50_95"),
                "test_precision_fixed_val_threshold": _nested_number(summary, "fixed_threshold_detection", "precision"),
                "test_recall_fixed_val_threshold": _nested_number(summary, "fixed_threshold_detection", "recall"),
                "test_f1_fixed_val_threshold": _nested_number(summary, "fixed_threshold_detection", "f1"),
                "test_keypoint_nme": _nested_number(summary, "keypoints", "nme"),
                "test_pck_0_05": _nested_number(summary, "keypoints", "pck_0.05"),
                "test_pck_0_10": _nested_number(summary, "keypoints", "pck_0.10"),
            }
        )
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(config: PipelineConfig) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for key, model in config.models.items():
        path = config.outputs.evaluation_root / model.run_name / "evaluation_summary.json"
        if path.is_file():
            summaries[key] = json.loads(path.read_text(encoding="utf-8"))
    rows = [_comparison_row(key, config, summaries.get(key)) for key in VARIANTS]
    recommendation = recommend_variant(summaries, config.selection)
    report = {
        "experiment": config.experiment_name,
        "dataset_images": config.dataset.expected_images,
        "rows": rows,
        "recommendation": recommendation,
        "note": "Official COCO references are human-pose results and are not this face dataset's measured performance.",
    }
    root = config.outputs.evaluation_root
    _write_json(root / "comparison.json", report)
    _write_csv(root / "comparison.csv", rows)
    headers = ["variant", "status", "test_box_map50_95", "test_pose_map50_95", "test_keypoint_nme", "test_f1_fixed_val_threshold"]
    lines = [
        "# YOLO26 Pose fixed-dataset comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    lines.extend(
        [
            "",
            f"Recommendation status: `{recommendation['status']}`",
            f"Recommended variant: `{recommendation['recommended']}`",
            "",
            str(recommendation["reason"]),
            "",
            f"The official COCO numbers are reference data only; they are not {config.dataset.version} results.",
        ]
    )
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def configure_logging(config: PipelineConfig, verbose: bool = False) -> None:
    config.outputs.preflight_root.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    handlers.append(logging.FileHandler(config.outputs.preflight_root / "pipeline.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production YOLO26s/m Pose training and comparison pipeline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["preflight", "train", "evaluate", "compare", "all"], default="preflight")
    parser.add_argument("--models", nargs="+", default=["all"], help="s, m, all, or a comma-separated list")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or 0,1")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--skip-weight-download", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume selected partial runs from last.pt")
    parser.add_argument("--force-eval", action="store_true", help="Regenerate derived evaluation outputs")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_pipeline_config(args.config, project_root_override=args.project_root)
    configure_logging(config, args.verbose)
    model_keys = select_model_keys(args.models)
    device = str(args.device if args.device is not None else config.runtime.get("device", "0"))
    allow_download = bool(config.runtime.get("allow_weight_download", True)) and not args.skip_weight_download
    try:
        if args.stage in {"preflight", "all"}:
            run_preflight(config, model_keys, allow_weight_download=allow_download)
        if args.stage in {"train", "all"}:
            if args.stage == "train":
                run_preflight(config, model_keys, allow_weight_download=allow_download)
            run_training(config, model_keys, device=device, resume=args.resume)
        if args.stage in {"evaluate", "all"}:
            run_evaluation(config, model_keys, device=device, force=args.force_eval)
        if args.stage in {"compare", "all"}:
            report = run_comparison(config)
            LOGGER.info("Model recommendation: %s (%s)", report["recommendation"]["recommended"], report["recommendation"]["status"])
    except PipelineError as error:
        LOGGER.error("%s", error)
        return 2
    return 0


__all__ = [
    "DatasetSpec",
    "ModelSpec",
    "OutputSpec",
    "WandbSpec",
    "PipelineConfig",
    "PipelineError",
    "load_pipeline_config",
    "main",
    "recommend_variant",
    "run_comparison",
    "run_evaluation",
    "run_preflight",
    "run_training",
    "select_model_keys",
    "validate_production_training",
    "validate_wandb_config",
]
