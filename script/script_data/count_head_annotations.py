import argparse
import csv
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

#コマンド：python script/script_data/count_head_annotations.py data/output/2026_06_21_1/task2_yk.zip

DEFAULT_WANDB_ENTITY = "sato-waseda-university"
DEFAULT_WANDB_PROJECT = "face_detection"


@dataclass
class ImageHeadCount:
    zip_path: Path
    xml_name: str
    image_name: str
    head_count: int


@dataclass
class ZipHeadCount:
    zip_path: Path
    xml_files: int
    images: int
    images_with_head: int
    heads: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count CVAT Head annotations in zip files under data/annotation_data."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=Path("data/annotation_data"),
        help="Zip file or directory that contains CVAT annotation zip files.",
    )
    parser.add_argument(
        "--label",
        default="Head",
        help="Box label to count. Default: Head.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        help="Optional CSV path for per-image counts.",
    )
    parser.add_argument(
        "--dataset-version",
        default="face_dataset_v7",
        help="Dataset version recorded in W&B. Default: face_dataset_v7.",
    )
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument(
        "--wandb",
        action="store_true",
        help="Log the annotation summary and tables to Weights & Biases.",
    )
    wandb_group.add_argument(
        "--wandb-entity",
        default=DEFAULT_WANDB_ENTITY,
        help=f"W&B user or team name. Default: {DEFAULT_WANDB_ENTITY}.",
    )
    wandb_group.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help=f"W&B project name. Default: {DEFAULT_WANDB_PROJECT}.",
    )
    wandb_group.add_argument(
        "--wandb-run-name",
        help="Optional display name for the W&B run.",
    )
    wandb_group.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B logging mode. Default: online.",
    )
    return parser.parse_args()


def find_zip_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".zip":
            raise ValueError(f"Input file is not a zip file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    return sorted(path for path in input_path.rglob("*.zip") if path.is_file())


def is_target_head_box(box: ET.Element, label: str) -> bool:
    return (box.get("label") or "").casefold() == label.casefold()


def count_heads_in_xml(
    xml_bytes: bytes,
    zip_path: Path,
    xml_name: str,
    label: str,
) -> list[ImageHeadCount]:
    root = ET.fromstring(xml_bytes)
    rows = []
    for image in root.findall(".//image"):
        head_count = sum(1 for box in image.findall("box") if is_target_head_box(box, label))
        rows.append(
            ImageHeadCount(
                zip_path=zip_path,
                xml_name=xml_name,
                image_name=image.get("name", ""),
                head_count=head_count,
            )
        )
    return rows


def count_heads_in_zip(zip_path: Path, label: str) -> tuple[ZipHeadCount, list[ImageHeadCount]]:
    image_rows = []
    xml_files = 0

    with zipfile.ZipFile(zip_path) as archive:
        xml_names = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".xml") and not name.endswith("/")
        )
        for xml_name in xml_names:
            xml_files += 1
            image_rows.extend(
                count_heads_in_xml(
                    xml_bytes=archive.read(xml_name),
                    zip_path=zip_path,
                    xml_name=xml_name,
                    label=label,
                )
            )

    zip_count = ZipHeadCount(
        zip_path=zip_path,
        xml_files=xml_files,
        images=len(image_rows),
        images_with_head=sum(1 for row in image_rows if row.head_count > 0),
        heads=sum(row.head_count for row in image_rows),
    )
    return zip_count, image_rows


def write_csv(path: Path, rows: list[ImageHeadCount]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["zip_path", "xml_name", "image_name", "head_count"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "zip_path": str(row.zip_path),
                    "xml_name": row.xml_name,
                    "image_name": row.image_name,
                    "head_count": row.head_count,
                }
            )


def log_counts_to_wandb(
    args: argparse.Namespace,
    zip_counts: list[ZipHeadCount],
) -> None:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B logging was requested, but wandb is not installed. "
            "Run: python -m pip install -r requirements.txt"
        ) from error

    scalar_metrics = {
        "dataset/zip_files": len(zip_counts),
        "dataset/xml_files": sum(count.xml_files for count in zip_counts),
        "dataset/images": sum(count.images for count in zip_counts),
        "dataset/images_with_head": sum(count.images_with_head for count in zip_counts),
        "dataset/heads": sum(count.heads for count in zip_counts),
    }
    init_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "job_type": "annotation-audit",
        "mode": args.wandb_mode,
        "config": {
            "input_path": str(args.input_path),
            "label": args.label,
            "dataset_version": args.dataset_version,
            "csv_output": str(args.csv_output) if args.csv_output else None,
        },
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity

    with wandb.init(**init_kwargs) as run:
        log_data = dict(scalar_metrics)
        if len(zip_counts) > 1:
            log_data["tables/per_zip"] = wandb.Table(
                columns=["zip_path", "xml_files", "images", "images_with_head", "heads"],
                data=[
                    [
                        str(count.zip_path),
                        count.xml_files,
                        count.images,
                        count.images_with_head,
                        count.heads,
                    ]
                    for count in zip_counts
                ],
            )

        run.log(log_data)
        if run.url:
            print(f"wandb_run: {run.url}")
        else:
            print(f"wandb_run: completed ({args.wandb_mode} mode)")


def main() -> int:
    args = parse_args()
    zip_files = find_zip_files(args.input_path)
    if not zip_files:
        print(f"No zip files found under {args.input_path}")
        return 0

    zip_counts = []
    image_rows = []
    for zip_path in zip_files:
        zip_count, rows = count_heads_in_zip(zip_path, args.label)
        zip_counts.append(zip_count)
        image_rows.extend(rows)

    print(f"input: {args.input_path}")
    print(f"label: {args.label}")
    print(f"zip_files: {len(zip_counts)}")
    print(f"xml_files: {sum(count.xml_files for count in zip_counts)}")
    print(f"images: {sum(count.images for count in zip_counts)}")
    print(f"images_with_head: {sum(count.images_with_head for count in zip_counts)}")
    print(f"heads: {sum(count.heads for count in zip_counts)}")

    if len(zip_counts) > 1:
        print("")
        print("per_zip:")
        for count in zip_counts:
            print(
                f"{count.zip_path}: heads={count.heads}, images={count.images}, "
                f"images_with_head={count.images_with_head}, xml_files={count.xml_files}"
            )

    if args.csv_output:
        write_csv(args.csv_output, image_rows)
        print(f"csv_output: {args.csv_output}")

    if args.wandb:
        log_counts_to_wandb(args, zip_counts)

    return 0


if __name__ == "__main__":
    sys.exit(main())
