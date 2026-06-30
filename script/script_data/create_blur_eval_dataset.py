from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is available in the project env.
    yaml = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CliError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a YOLO-format evaluation dataset with blur/pixelation applied to images or face boxes."
    )
    parser.add_argument("input_root", type=Path, help="YOLO dataset root that contains images/{split} and labels/{split}.")
    parser.add_argument("output_root", type=Path, help="Output YOLO dataset root for blurred evaluation.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Dataset split(s) to process. Default: test.",
    )
    parser.add_argument(
        "--target",
        choices=["face", "image"],
        default="face",
        help="Apply blur to YOLO face bbox regions or to the whole image. Default: face.",
    )
    parser.add_argument(
        "--mode",
        choices=["gaussian", "box", "pixelate", "pixelate-gaussian"],
        default="gaussian",
        help="Blur mode. pixelate is closest to mosaic. Default: gaussian.",
    )
    parser.add_argument("--radius", type=float, default=3.0, help="Blur radius for gaussian/box modes. Default: 3.0.")
    parser.add_argument("--block-size", type=int, default=12, help="Pixel block size for pixelate modes. Default: 12.")
    parser.add_argument("--bbox-padding", type=float, default=0.10, help="Relative padding around each face bbox. Default: 0.10.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG/WebP save quality. Default: 95.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional smoke limit per split. 0 means all images.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    return parser.parse_args()


def iter_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise CliError(f"Image split directory does not exist: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_yolo_boxes(label_path: Path, width: int, height: int, padding: float) -> list[tuple[int, int, int, int]]:
    if not label_path.exists():
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cx = float(parts[1]) * width
            cy = float(parts[2]) * height
            box_w = float(parts[3]) * width
            box_h = float(parts[4]) * height
        except ValueError:
            continue
        pad_x = box_w * padding
        pad_y = box_h * padding
        x1 = max(0, math.floor(cx - box_w * 0.5 - pad_x))
        y1 = max(0, math.floor(cy - box_h * 0.5 - pad_y))
        x2 = min(width, math.ceil(cx + box_w * 0.5 + pad_x))
        y2 = min(height, math.ceil(cy + box_h * 0.5 + pad_y))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def pixelate(image: Image.Image, block_size: int) -> Image.Image:
    block_size = max(1, int(block_size))
    width, height = image.size
    small_size = (max(1, math.ceil(width / block_size)), max(1, math.ceil(height / block_size)))
    small = image.resize(small_size, Image.Resampling.BILINEAR)
    return small.resize((width, height), Image.Resampling.NEAREST)


def apply_effect(image: Image.Image, *, mode: str, radius: float, block_size: int) -> Image.Image:
    if mode == "gaussian":
        return image.filter(ImageFilter.GaussianBlur(radius=max(0.0, float(radius))))
    if mode == "box":
        return image.filter(ImageFilter.BoxBlur(radius=max(0.0, float(radius))))
    if mode == "pixelate":
        return pixelate(image, block_size)
    if mode == "pixelate-gaussian":
        return pixelate(image, block_size).filter(ImageFilter.GaussianBlur(radius=max(0.0, float(radius))))
    raise ValueError(f"Unsupported mode: {mode}")


def process_image(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    *,
    target: str,
    mode: str,
    radius: float,
    block_size: int,
    bbox_padding: float,
    jpeg_quality: int,
) -> int:
    image = Image.open(image_path).convert("RGB")
    if target == "image":
        processed = apply_effect(image, mode=mode, radius=radius, block_size=block_size)
        affected_regions = 1
    else:
        processed = image.copy()
        boxes = read_yolo_boxes(label_path, image.width, image.height, bbox_padding)
        for box in boxes:
            crop = processed.crop(box)
            processed.paste(apply_effect(crop, mode=mode, radius=radius, block_size=block_size), box)
        affected_regions = len(boxes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {}
    if output_path.suffix.lower() in {".jpg", ".jpeg", ".webp"}:
        save_kwargs["quality"] = int(jpeg_quality)
    processed.save(output_path, **save_kwargs)
    return affected_regions


def copy_labels(input_root: Path, output_root: Path, split: str) -> int:
    src = input_root / "labels" / split
    dst = output_root / "labels" / split
    if not src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for label_path in sorted(src.glob("*.txt")):
        shutil.copy2(label_path, dst / label_path.name)
        count += 1
    return count


def copy_metadata_and_root_files(input_root: Path, output_root: Path, splits: list[str]) -> None:
    metadata_dir = input_root / "metadata"
    if metadata_dir.exists():
        shutil.copytree(metadata_dir, output_root / "metadata", dirs_exist_ok=True)

    for path in input_root.iterdir():
        if path.name in {"images", "labels", "metadata"} or path.suffix == ".cache":
            continue
        if path.is_file():
            target = output_root / path.name
            if path.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                data["path"] = str(output_root)
                for split in splits:
                    data[split] = f"images/{split}"
                target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            else:
                shutil.copy2(path, target)


def ensure_output_root(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise CliError(f"Output directory is not empty. Use --overwrite to continue: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        raise CliError(f"Input dataset root does not exist: {input_root}")
    if args.radius < 0:
        raise CliError("--radius must be >= 0")
    if args.block_size <= 0:
        raise CliError("--block-size must be > 0")
    if args.jpeg_quality <= 0 or args.jpeg_quality > 100:
        raise CliError("--jpeg-quality must be in 1..100")

    splits = [str(split) for split in args.splits]
    ensure_output_root(output_root, overwrite=bool(args.overwrite))
    copy_metadata_and_root_files(input_root, output_root, splits)

    summary: dict[str, Any] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "splits": splits,
        "target": args.target,
        "mode": args.mode,
        "radius": args.radius,
        "block_size": args.block_size,
        "bbox_padding": args.bbox_padding,
        "jpeg_quality": args.jpeg_quality,
        "split_counts": {},
    }

    total_images = 0
    total_regions = 0
    for split in splits:
        image_paths = iter_images(input_root / "images" / split)
        if args.max_images > 0:
            image_paths = image_paths[: args.max_images]
        labels_copied = copy_labels(input_root, output_root, split)
        split_regions = 0
        for image_path in image_paths:
            label_path = input_root / "labels" / split / f"{image_path.stem}.txt"
            output_path = output_root / "images" / split / image_path.name
            split_regions += process_image(
                image_path,
                label_path,
                output_path,
                target=args.target,
                mode=args.mode,
                radius=args.radius,
                block_size=args.block_size,
                bbox_padding=args.bbox_padding,
                jpeg_quality=args.jpeg_quality,
            )
        total_images += len(image_paths)
        total_regions += split_regions
        summary["split_counts"][split] = {
            "images": len(image_paths),
            "labels_copied": labels_copied,
            "affected_regions": split_regions,
        }

    summary["images"] = total_images
    summary["affected_regions"] = total_regions
    (output_root / "blur_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CliError as error:
        raise SystemExit(f"Error: {error}") from None
