import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from typing import NamedTuple


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class CliError(RuntimeError):
    pass


class DuplicateGroup(NamedTuple):
    sha256: str
    kept: str
    removed: tuple[str, ...]
    annotated_members: tuple[str, ...]


class DeduplicationResult(NamedTuple):
    source_images: int
    output_images: int
    duplicate_groups: int
    removed_images: int
    removed_annotated_images: int
    output_dir: Path
    output_zip: Path
    manifest_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove byte-identical images and their empty CVAT image entries, while "
            "writing a new dataset and leaving the source unchanged."
        )
    )
    parser.add_argument("image_dir", type=Path, help="Directory containing the source images.")
    parser.add_argument(
        "annotation_zip",
        type=Path,
        help="CVAT annotation zip containing annotations.xml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New output directory. Default: <image_dir>_deduplicated.",
    )
    parser.add_argument(
        "--xml-name",
        default="annotations.xml",
        help="XML member name inside the annotation zip. Default: annotations.xml.",
    )
    parser.add_argument(
        "--annotated-duplicate-policy",
        choices=("error", "keep-first"),
        default="error",
        help=(
            "How to handle a duplicate group with multiple annotated images. "
            "Default: error. keep-first retains the first image in XML order."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_image_path(image_name: str) -> Path:
    posix_path = PurePosixPath(image_name)
    if not image_name or posix_path.is_absolute() or ".." in posix_path.parts:
        raise CliError(f"Unsafe or empty image name in annotations.xml: {image_name!r}")
    return Path(*posix_path.parts)


def find_image_files(image_dir: Path) -> dict[str, Path]:
    image_paths: dict[str, Path] = {}
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_name = path.relative_to(image_dir).as_posix()
        image_paths[relative_name] = path
    return image_paths


def read_annotation_xml(annotation_zip: Path, xml_name: str) -> tuple[ET.ElementTree, bytes]:
    if not annotation_zip.is_file():
        raise CliError(f"Annotation zip does not exist: {annotation_zip}")

    with zipfile.ZipFile(annotation_zip) as archive:
        matching_names = [name for name in archive.namelist() if name == xml_name]
        if len(matching_names) != 1:
            raise CliError(
                f"Expected exactly one {xml_name!r} in {annotation_zip}, "
                f"found {len(matching_names)}."
            )
        xml_bytes = archive.read(xml_name)
    return ET.ElementTree(ET.fromstring(xml_bytes)), xml_bytes


def index_cvat_images(root: ET.Element) -> tuple[list[ET.Element], dict[str, ET.Element]]:
    images = root.findall(".//image")
    if not images:
        raise CliError("No <image> elements were found in annotations.xml.")

    images_by_name: dict[str, ET.Element] = {}
    for image in images:
        name = image.get("name", "")
        relative_image_path(name)
        if name in images_by_name:
            raise CliError(f"Duplicate image name in annotations.xml: {name}")
        images_by_name[name] = image
    return images, images_by_name


def validate_one_to_one_mapping(
    image_paths: dict[str, Path], images_by_name: dict[str, ET.Element]
) -> None:
    disk_names = set(image_paths)
    xml_names = set(images_by_name)
    missing_files = sorted(xml_names - disk_names)
    missing_xml = sorted(disk_names - xml_names)
    if not missing_files and not missing_xml:
        return

    details = []
    if missing_files:
        details.append(
            "missing image files: " + ", ".join(missing_files[:5])
            + (" ..." if len(missing_files) > 5 else "")
        )
    if missing_xml:
        details.append(
            "images missing from XML: " + ", ".join(missing_xml[:5])
            + (" ..." if len(missing_xml) > 5 else "")
        )
    raise CliError("Images and XML are not one-to-one; " + "; ".join(details))


def find_duplicate_groups(
    images: list[ET.Element],
    images_by_name: dict[str, ET.Element],
    image_paths: dict[str, Path],
    annotated_duplicate_policy: str = "error",
) -> list[DuplicateGroup]:
    if annotated_duplicate_policy not in {"error", "keep-first"}:
        raise ValueError(
            "annotated_duplicate_policy must be either 'error' or 'keep-first'"
        )

    names_by_digest: dict[str, list[str]] = {}
    for image in images:
        name = image.get("name", "")
        digest = sha256_file(image_paths[name])
        names_by_digest.setdefault(digest, []).append(name)

    groups = []
    for digest, names in names_by_digest.items():
        if len(names) < 2:
            continue

        annotated_members = [name for name in names if len(images_by_name[name]) > 0]
        if len(annotated_members) > 1 and annotated_duplicate_policy == "error":
            raise CliError(
                "A byte-identical group contains multiple annotated images; refusing to "
                f"choose one automatically: {', '.join(annotated_members)}"
            )

        if len(annotated_members) > 1:
            kept = names[0]
        else:
            kept = annotated_members[0] if annotated_members else names[0]
        removed = tuple(name for name in names if name != kept)
        annotated_removed = [name for name in removed if len(images_by_name[name]) > 0]
        if annotated_removed and annotated_duplicate_policy == "error":
            raise CliError(
                "Deduplication would remove annotated images: " + ", ".join(annotated_removed)
            )
        groups.append(
            DuplicateGroup(
                sha256=digest,
                kept=kept,
                removed=removed,
                annotated_members=tuple(annotated_members),
            )
        )
    return groups


def update_cvat_xml(tree: ET.ElementTree, removed_names: set[str]) -> bytes:
    root = tree.getroot()
    images, images_by_name = index_cvat_images(root)
    parent_by_child = {child: parent for parent in root.iter() for child in parent}

    for name in removed_names:
        image = images_by_name[name]
        parent = parent_by_child.get(image)
        if parent is None:
            raise CliError(f"Could not find parent element for XML image: {name}")
        parent.remove(image)

    remaining_images = root.findall(".//image")
    for new_id, image in enumerate(remaining_images):
        image.set("id", str(new_id))

    image_count = len(remaining_images)
    stop_frame = max(0, image_count - 1)
    for metadata_root in root.findall(".//meta/job") + root.findall(".//meta/task"):
        size = metadata_root.find("size")
        if size is not None:
            size.text = str(image_count)
        start = metadata_root.find("start_frame")
        if start is not None:
            start.text = "0"
        stop = metadata_root.find("stop_frame")
        if stop is not None:
            stop.text = str(stop_frame)

        segments = metadata_root.findall("./segments/segment")
        if len(segments) > 1:
            raise CliError("CVAT exports with multiple segments are not supported safely.")
        if segments:
            segment_start = segments[0].find("start")
            segment_stop = segments[0].find("stop")
            if segment_start is not None:
                segment_start.text = "0"
            if segment_stop is not None:
                segment_stop.text = str(stop_frame)

    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def write_annotation_zip(
    source_zip: Path, output_zip: Path, xml_name: str, cleaned_xml: bytes
) -> None:
    with zipfile.ZipFile(source_zip) as source, zipfile.ZipFile(output_zip, "w") as output:
        for info in source.infolist():
            data = cleaned_xml if info.filename == xml_name else source.read(info.filename)
            output.writestr(info, data)


def write_manifest(
    path: Path,
    image_dir: Path,
    annotation_zip: Path,
    source_images: int,
    groups: list[DuplicateGroup],
    annotated_duplicate_policy: str,
) -> None:
    removed_images = sum(len(group.removed) for group in groups)
    removed_annotated_images = sum(
        name in group.annotated_members for group in groups for name in group.removed
    )
    keeper_policy = (
        "the sole annotated image, otherwise the first XML image"
        if annotated_duplicate_policy == "error"
        else (
            "the sole annotated image; for multiple annotated duplicates, "
            "the first image in XML order"
        )
    )
    manifest = {
        "schema_version": 1,
        "source_image_dir": str(image_dir),
        "source_annotation_zip": str(annotation_zip),
        "matching_method": "exact file SHA-256",
        "keeper_policy": keeper_policy,
        "summary": {
            "source_images": source_images,
            "output_images": source_images - removed_images,
            "duplicate_groups": len(groups),
            "removed_images": removed_images,
            "removed_annotated_images": removed_annotated_images,
        },
        "duplicate_groups": [
            {
                "sha256": group.sha256,
                "kept": group.kept,
                "removed": list(group.removed),
                "annotated_members": list(group.annotated_members),
            }
            for group in groups
        ],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")


def deduplicate_dataset(
    image_dir: Path,
    annotation_zip: Path,
    output_dir: Path,
    xml_name: str = "annotations.xml",
    annotated_duplicate_policy: str = "error",
) -> DeduplicationResult:
    if not image_dir.is_dir():
        raise CliError(f"Image directory does not exist: {image_dir}")
    if output_dir.exists():
        raise CliError(f"Output path already exists: {output_dir}")

    source_resolved = image_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == source_resolved or source_resolved in output_resolved.parents:
        raise CliError("Output directory must not be the source directory or inside it.")

    tree, _ = read_annotation_xml(annotation_zip, xml_name)
    images, images_by_name = index_cvat_images(tree.getroot())
    image_paths = find_image_files(image_dir)
    validate_one_to_one_mapping(image_paths, images_by_name)
    groups = find_duplicate_groups(
        images,
        images_by_name,
        image_paths,
        annotated_duplicate_policy=annotated_duplicate_policy,
    )
    removed_names = {name for group in groups for name in group.removed}
    removed_annotated_images = sum(
        name in group.annotated_members for group in groups for name in group.removed
    )
    cleaned_xml = update_cvat_xml(tree, removed_names)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}_staging_", dir=output_dir.parent)
    )
    try:
        for name, source_path in image_paths.items():
            if name in removed_names:
                continue
            destination = staging_dir / relative_image_path(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)

        output_zip = staging_dir / annotation_zip.name
        write_annotation_zip(annotation_zip, output_zip, xml_name, cleaned_xml)
        manifest_path = staging_dir / "deduplication_manifest.json"
        write_manifest(
            manifest_path,
            image_dir=image_dir,
            annotation_zip=annotation_zip,
            source_images=len(images),
            groups=groups,
            annotated_duplicate_policy=annotated_duplicate_policy,
        )
        staging_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return DeduplicationResult(
        source_images=len(images),
        output_images=len(images) - len(removed_names),
        duplicate_groups=len(groups),
        removed_images=len(removed_names),
        removed_annotated_images=removed_annotated_images,
        output_dir=output_dir,
        output_zip=output_dir / annotation_zip.name,
        manifest_path=output_dir / "deduplication_manifest.json",
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.image_dir.with_name(
        f"{args.image_dir.name}_deduplicated"
    )
    try:
        result = deduplicate_dataset(
            image_dir=args.image_dir,
            annotation_zip=args.annotation_zip,
            output_dir=output_dir,
            xml_name=args.xml_name,
            annotated_duplicate_policy=args.annotated_duplicate_policy,
        )
    except (CliError, ET.ParseError, OSError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"source_images: {result.source_images}")
    print(f"duplicate_groups: {result.duplicate_groups}")
    print(f"removed_images: {result.removed_images}")
    print(f"output_images: {result.output_images}")
    print(f"removed_annotated_images: {result.removed_annotated_images}")
    print(f"output_dir: {result.output_dir}")
    print(f"output_zip: {result.output_zip}")
    print(f"manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
