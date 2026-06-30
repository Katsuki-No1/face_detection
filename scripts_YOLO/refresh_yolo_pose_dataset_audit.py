import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_VARIANT = "modified_100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh the current YOLO Pose dataset from CVAT zips and run the Phase 0 audit. "
            "This script prepares data and audits it; it does not train."
        )
    )
    parser.add_argument("--zip-root", type=Path, default=Path("data/output"))
    parser.add_argument(
        "--modified-zip",
        type=Path,
        action="append",
        default=[],
        help="Explicit CVAT zip. Can be supplied multiple times. Overrides --zip-root discovery.",
    )
    parser.add_argument(
        "--modified-zip-manifest",
        type=Path,
        default=None,
        help="Text file listing CVAT zips, one path per line. Used when --modified-zip is omitted.",
    )
    parser.add_argument("--old-data-root", type=Path, default=Path("datasets/face_pose"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs_experiment"))
    parser.add_argument("--run-name", default=None, help="Default: current timestamp.")
    parser.add_argument("--dataset-output-dir", type=Path, default=None)
    parser.add_argument("--audit-output-dir", type=Path, default=None)
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=["old_repro", "modified_50", "modified_100"])
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prepare-script", type=Path, default=Path("scripts_YOLO/run_yolo_pose_dataset_ablation.py"))
    parser.add_argument("--audit-script", type=Path, default=Path("scripts_YOLO/audit_yolo_pose_dataset.py"))
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Only discover zips and print the planned run.")
    parser.add_argument("--fail-on-warn", action="store_true", help="Exit non-zero when the audit status is warn.")
    parser.add_argument("--no-latest-links", action="store_true")
    parser.add_argument(
        "--latest-dataset-link",
        type=Path,
        default=Path("outputs_experiment/latest_yolo_pose_dataset_refresh"),
    )
    parser.add_argument(
        "--latest-audit-link",
        type=Path,
        default=Path("outputs_experiment/latest_phase0_dataset_audit"),
    )

    parser.add_argument("--small-face-area-threshold", type=float, default=0.01)
    parser.add_argument("--tiny-face-area-threshold", type=float, default=0.0025)
    parser.add_argument("--large-face-area-threshold", type=float, default=0.15)
    parser.add_argument("--edge-margin-ratio", type=float, default=0.02)
    parser.add_argument("--max-sample-weight", type=float, default=5.0)
    parser.add_argument("--max-sample-repeat", type=int, default=4)
    parser.add_argument("--near-frame-window", type=int, default=3)
    parser.add_argument("--near-time-window", type=float, default=2.0)
    parser.add_argument("--max-listed-issues", type=int, default=200)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_has_images(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS for child in path.iterdir())


def count_adjacent_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for child in path.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS)


def annotation_xml_name(zip_path: Path) -> str | None:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if Path(name).name == "annotations.xml":
                    return name
    except zipfile.BadZipFile:
        return None
    return None


def zip_contains_annotations_xml(zip_path: Path) -> bool:
    return annotation_xml_name(zip_path) is not None


def discover_cvat_zips(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"CVAT zip root not found: {root}")
    zips = []
    for zip_path in sorted(root.rglob("*.zip")):
        if not directory_has_images(zip_path.parent):
            continue
        if not zip_contains_annotations_xml(zip_path):
            continue
        zips.append(zip_path)
    if not zips:
        raise FileNotFoundError(f"No CVAT annotation zips were found under {root}")
    return zips


def read_zip_manifest(path: Path) -> list[Path]:
    zips = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = line.split("#", 1)[0].strip()
        if not value:
            continue
        zip_path = Path(value)
        if zip_path.suffix.lower() != ".zip":
            raise ValueError(f"Expected a .zip path at {path}:{line_number}: {value}")
        zips.append(zip_path)
    if not zips:
        raise ValueError(f"No zip paths were found in {path}")
    return zips


def modified_zip_inputs(args: argparse.Namespace) -> list[Path]:
    if args.modified_zip:
        return args.modified_zip
    if args.modified_zip_manifest is not None:
        return read_zip_manifest(args.modified_zip_manifest)
    return discover_cvat_zips(args.zip_root)


def cvat_zip_stats(zip_path: Path) -> dict[str, Any]:
    xml_name = annotation_xml_name(zip_path)
    if xml_name is None:
        return {
            "zip": str(zip_path),
            "annotations_xml": None,
            "annotation_images": 0,
            "heads": 0,
            "faces": 0,
            "occluded_faces": 0,
            "adjacent_images": count_adjacent_images(zip_path.parent),
            "sha1": sha1_file(zip_path),
        }

    with zipfile.ZipFile(zip_path) as archive:
        root = ElementTree.fromstring(archive.read(xml_name))

    image_count = 0
    heads = 0
    faces = 0
    occluded_faces = 0
    for image in root.iter("image"):
        image_count += 1
        for shape in image:
            label = shape.attrib.get("label")
            if label == "Head":
                heads += 1
            elif label == "Face":
                faces += 1
                occluded_faces += int(shape.attrib.get("occluded") == "1")

    return {
        "zip": str(zip_path),
        "resolved_zip": str(zip_path.resolve()),
        "annotations_xml": xml_name,
        "annotation_kind": zip_path.stem,
        "annotation_images": image_count,
        "adjacent_images": count_adjacent_images(zip_path.parent),
        "heads": heads,
        "faces": faces,
        "occluded_faces": occluded_faces,
        "sha1": sha1_file(zip_path),
    }


def run_command(command: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - started
    result = {
        "command": command,
        "log": str(log_path),
        "returncode": process.returncode,
        "elapsed_seconds": round(elapsed, 3),
    }
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}. See {log_path}")
    return result


def prepare_command(args: argparse.Namespace, zips: list[Path], dataset_output_dir: Path) -> list[str]:
    command = [
        args.python,
        str(args.prepare_script),
        "--output-dir",
        str(dataset_output_dir),
        "--old-data-root",
        str(args.old_data_root),
        "--prepare-only",
        "--train-fraction",
        str(args.train_fraction),
        "--sample-seed",
        str(args.sample_seed),
    ]
    command.append("--skip-existing" if args.skip_existing else "--no-skip-existing")
    for zip_path in zips:
        command.extend(["--modified-zip", str(zip_path)])
    return command


def audit_command(args: argparse.Namespace, data_root: Path, audit_output_dir: Path) -> list[str]:
    return [
        args.python,
        str(args.audit_script),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(audit_output_dir),
        "--small-face-area-threshold",
        str(args.small_face_area_threshold),
        "--tiny-face-area-threshold",
        str(args.tiny_face_area_threshold),
        "--large-face-area-threshold",
        str(args.large_face_area_threshold),
        "--edge-margin-ratio",
        str(args.edge_margin_ratio),
        "--max-sample-weight",
        str(args.max_sample_weight),
        "--max-sample-repeat",
        str(args.max_sample_repeat),
        "--near-frame-window",
        str(args.near_frame_window),
        "--near-time-window",
        str(args.near_time_window),
        "--max-listed-issues",
        str(args.max_listed_issues),
    ]


def update_latest_link(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.is_file():
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Refusing to replace existing directory: {link_path}")
    link_path.symlink_to(target.resolve(), target_is_directory=True)


def make_summary_markdown(summary: dict[str, Any]) -> str:
    audit = summary["audit"]
    audit_summary = audit.get("summary", {})
    checks = audit.get("checks", {})
    lines = [
        "# YOLO Pose Dataset Refresh",
        "",
        "This run prepared data and ran the Phase 0 audit. It did not train.",
        "",
        f"- run name: `{summary['run_name']}`",
        f"- status: `{audit.get('status')}`",
        f"- zip count: `{summary['source']['zip_count']}`",
        f"- source annotation images: `{summary['source']['annotation_images']}`",
        f"- dataset output: `{summary['dataset_output_dir']}`",
        f"- audit output: `{summary['audit_output_dir']}`",
        f"- data root: `{summary['data_root']}`",
        f"- balanced yaml: `{summary['balanced_yaml']}`",
        "",
        "## Audit Summary",
        "",
        "| item | value |",
        "| --- | ---: |",
    ]
    for key in [
        "images",
        "instances",
        "positive_images",
        "negative_images",
        "small_instances",
        "metadata_faces",
        "metadata_face_occluded",
        "metadata_ellipse_out_of_image",
        "metadata_objects_with_occluded_keypoint",
        "metadata_excluded_faces",
    ]:
        lines.append(f"| {key} | {audit_summary.get(key, 0)} |")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| check | value |",
            "| --- | ---: |",
            f"| failures | {checks.get('failure_count', 0)} |",
            f"| duplicate cross-split images | {checks.get('duplicate_image_cross_split_count', 0)} |",
            f"| near-frame cross-split pairs | {checks.get('near_frame_cross_split_pair_count', 0)} |",
            f"| metadata/object mismatch images | {checks.get('metadata_object_mismatch_image_count', 0)} |",
            f"| ellipse out of image count | {checks.get('ellipse_out_of_image_count', 0)} |",
            f"| balanced train unique images | {checks.get('balanced_train_unique_images', 0)} |",
            f"| balanced train repeated entries | {checks.get('balanced_train_repeated_entries', 0)} |",
            "",
            "## Source Zips",
            "",
            "| zip | annotation images | faces | heads | adjacent images |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["source"]["zips"]:
        lines.append(
            f"| `{row['zip']}` | {row['annotation_images']} | {row['faces']} | "
            f"{row['heads']} | {row['adjacent_images']} |"
        )
    return "\n".join(lines)


def output_paths(args: argparse.Namespace) -> tuple[str, Path, Path]:
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_output_dir = args.dataset_output_dir or args.output_root / f"yolo_pose_dataset_refresh_{run_name}"
    audit_output_dir = args.audit_output_dir or args.output_root / f"phase0_dataset_audit_{run_name}_{args.variant}"
    return run_name, dataset_output_dir, audit_output_dir


def main() -> int:
    args = parse_args()
    run_name, dataset_output_dir, audit_output_dir = output_paths(args)
    zips = modified_zip_inputs(args)
    missing_zips = [path for path in zips if not path.exists()]
    if missing_zips:
        raise FileNotFoundError("Missing CVAT zip(s): " + ", ".join(str(path) for path in missing_zips))
    if not args.old_data_root.exists():
        raise FileNotFoundError(f"Old/fixed evaluation data root not found: {args.old_data_root}")

    zip_rows = [cvat_zip_stats(path) for path in zips]
    source_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "zip_root": str(args.zip_root),
        "zip_count": len(zip_rows),
        "annotation_images": sum(int(row["annotation_images"]) for row in zip_rows),
        "adjacent_images": sum(int(row["adjacent_images"]) for row in zip_rows),
        "faces": sum(int(row["faces"]) for row in zip_rows),
        "heads": sum(int(row["heads"]) for row in zip_rows),
        "occluded_faces": sum(int(row["occluded_faces"]) for row in zip_rows),
        "zips": zip_rows,
    }

    plan = {
        "run_name": run_name,
        "dataset_output_dir": str(dataset_output_dir),
        "audit_output_dir": str(audit_output_dir),
        "variant": args.variant,
        "source": source_manifest,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    audit_output_dir.mkdir(parents=True, exist_ok=True)
    write_json(audit_output_dir / "source_zip_manifest.json", source_manifest)

    commands = []
    prepare_result = run_command(
        prepare_command(args, zips, dataset_output_dir),
        dataset_output_dir / "logs" / "refresh_prepare.log",
    )
    commands.append(prepare_result)

    data_root = dataset_output_dir / "datasets" / args.variant
    if not data_root.exists():
        raise FileNotFoundError(f"Prepared dataset variant was not found: {data_root}")

    audit_result = run_command(
        audit_command(args, data_root, audit_output_dir),
        audit_output_dir / "refresh_audit.log",
    )
    commands.append(audit_result)

    audit_report = read_json(audit_output_dir / "phase0_dataset_audit.json")
    balanced_yaml = audit_output_dir / "face_pose_class_balanced.yaml"
    summary = {
        "schema_version": "1.0",
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_output_dir": str(dataset_output_dir),
        "audit_output_dir": str(audit_output_dir),
        "data_root": str(data_root),
        "balanced_yaml": str(balanced_yaml),
        "source": source_manifest,
        "commands": commands,
        "audit": audit_report,
    }
    write_json(audit_output_dir / "refresh_summary.json", summary)
    write_text(audit_output_dir / "refresh_summary.md", make_summary_markdown(summary))

    if not args.no_latest_links:
        update_latest_link(args.latest_dataset_link, dataset_output_dir)
        update_latest_link(args.latest_audit_link, audit_output_dir)

    print(json.dumps({"refresh": summary}, ensure_ascii=False, indent=2))
    if args.fail_on_warn and audit_report.get("status") != "pass":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
