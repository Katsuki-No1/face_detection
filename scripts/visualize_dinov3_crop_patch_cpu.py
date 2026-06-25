import argparse
import csv
import html
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.script_model.run_dinov3 import (
    extract_features,
    load_backbone_from_repo,
    load_batch,
    make_transform,
    resolve_weights,
)
from src.phase1_dino.dataset import (
    box_iou,
    clip_box,
    crop_image,
    expand_box,
    head_box_for_face,
    metadata_for_image,
    metadata_index,
    random_background_box,
    read_label_objects,
    valid_metadata_objects,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create crop galleries and DINOv3 patch similarity maps on CPU.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("outputs_experiment/yolo_pose_dataset_ablation_20260625_all_zips/datasets/_modified_all_train/metadata/conversion_metadata.json"),
    )
    parser.add_argument("--dinov3-repo", type=Path, default=Path("/home/katsuki/.cache/torch/hub/facebookresearch_dinov3_main"))
    parser.add_argument("--weights", default="script/script_model/DINO/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-patch-images", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["image_path"] = Path(row["image"])
        row["label_path"] = Path(row["label"])
        row["objects"] = int(row.get("objects") or 0)
        row["classes_list"] = [item for item in row.get("classes", "").split("|") if item]
    return rows


def make_crop_record(
    source_row: dict[str, Any],
    crop_type: str,
    bbox: list[float],
    image_size: tuple[int, int],
    object_index: int,
    crop_index: int,
    tags: list[str],
) -> dict[str, Any]:
    image_path = source_row["image_path"]
    crop_id = f"{Path(image_path).stem}_{crop_type}_{object_index}_{crop_index}"
    return {
        "crop_id": crop_id,
        "source": source_row["source"],
        "selection_stratum": source_row["selection_stratum"],
        "image": str(image_path),
        "label": str(source_row["label_path"]),
        "crop_type": crop_type,
        "object_index": object_index,
        "crop_index": crop_index,
        "bbox_xyxy": [round(float(value), 3) for value in bbox],
        "width": image_size[0],
        "height": image_size[1],
        "tags": tags,
    }


def generate_crops(rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], output_dir: Path, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    crop_records: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    crop_root = output_dir / "crops"
    for crop_type in ["face", "head", "background", "hard_negative"]:
        (crop_root / crop_type).mkdir(parents=True, exist_ok=True)

    for row_index, row in enumerate(rows, start=1):
        image_path = row["image_path"]
        with Image.open(image_path) as image:
            width, height = image.size
        label_objects = read_label_objects(row["label_path"], width, height)
        face_boxes = [item["bbox_xyxy"] for item in label_objects]
        meta_record = metadata_for_image(metadata, image_path)
        meta_objects = valid_metadata_objects(meta_record)
        image_records.append(
            {
                "image_index": row_index,
                "image": str(image_path),
                "source": row["source"],
                "selection_stratum": row["selection_stratum"],
                "classes": row.get("classes", ""),
                "width": width,
                "height": height,
                "face_bboxes": face_boxes,
            }
        )

        for object_index, item in enumerate(label_objects):
            face_box = expand_box(item["bbox_xyxy"], width, height, 0.15) or item["bbox_xyxy"]
            crop_records.append(
                make_crop_record(
                    row,
                    "face",
                    face_box,
                    (width, height),
                    object_index,
                    0,
                    ["face_like"],
                )
            )
            head_box = head_box_for_face(meta_record, item["bbox_xyxy"], width, height)
            if head_box is not None:
                crop_records.append(
                    make_crop_record(
                        row,
                        "head",
                        head_box,
                        (width, height),
                        object_index,
                        0,
                        ["face_like", "head_context"],
                    )
                )
            elif object_index < len(meta_objects):
                raw = meta_objects[object_index].get("training_bbox", {})
                fallback = clip_box(
                    [
                        float(raw.get("xtl", item["bbox_xyxy"][0])) - 20,
                        float(raw.get("ytl", item["bbox_xyxy"][1])) - 30,
                        float(raw.get("xbr", item["bbox_xyxy"][2])) + 20,
                        float(raw.get("ybr", item["bbox_xyxy"][3])) + 30,
                    ],
                    width,
                    height,
                )
                if fallback is not None:
                    crop_records.append(
                        make_crop_record(row, "head", fallback, (width, height), object_index, 0, ["face_like", "head_fallback"])
                    )

        background_box = random_background_box(rng, width, height, face_boxes, 32, 0.05, False)
        if background_box is not None:
            crop_records.append(
                make_crop_record(row, "background", background_box, (width, height), -1, 0, ["non_face", "random_background"])
            )
        hard_box = random_background_box(rng, width, height, face_boxes, 32, 0.05, True)
        if hard_box is not None:
            crop_records.append(
                make_crop_record(row, "hard_negative", hard_box, (width, height), -1, 0, ["non_face", "face_sized_background"])
            )

    for index, record in enumerate(crop_records, start=1):
        crop_type = record["crop_type"]
        crop_path = crop_root / crop_type / f"{index:04d}_{record['crop_id']}.jpg"
        crop = crop_image(Path(record["image"]), record["bbox_xyxy"])
        crop.save(crop_path, quality=92)
        record["crop_path"] = str(crop_path)

    return crop_records, image_records


def write_crop_metadata(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "crop_id",
        "source",
        "selection_stratum",
        "image",
        "label",
        "crop_type",
        "object_index",
        "crop_index",
        "bbox_xyxy",
        "width",
        "height",
        "tags",
        "crop_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["bbox_xyxy"] = json.dumps(row["bbox_xyxy"])
            row["tags"] = "|".join(row["tags"])
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_contact_sheet(records: list[dict[str, Any]], crop_type: str, output_path: Path, max_items: int = 60) -> None:
    items = [record for record in records if record["crop_type"] == crop_type][:max_items]
    if not items:
        return
    thumb_w, thumb_h = 180, 150
    label_h = 46
    columns = 5
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, record in enumerate(items):
        x = (idx % columns) * thumb_w
        y = (idx // columns) * (thumb_h + label_h)
        with Image.open(record["crop_path"]) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            sheet.paste(image, (x + (thumb_w - image.width) // 2, y))
        draw.text((x + 4, y + thumb_h + 2), record["source"], fill=(0, 0, 0))
        draw.text((x + 4, y + thumb_h + 18), record["selection_stratum"][:24], fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def load_dinov3(repo: Path, weights: str, torch: Any) -> Any:
    model = load_backbone_from_repo(repo, "dinov3_vitb16", resolve_weights(weights))
    model.eval().to("cpu")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def feature_batch(paths: list[Path], model: Any, transform: Any, ImageModule: Any, torch: Any) -> tuple[np.ndarray, np.ndarray]:
    batch = load_batch(paths, transform, ImageModule, torch).to("cpu")
    with torch.inference_mode():
        cls_tokens, patch_tokens, _ = extract_features(model, batch, torch)
    cls = cls_tokens.detach().float().cpu().numpy()
    patches = patch_tokens.detach().float().cpu().numpy() if patch_tokens is not None else np.empty((len(paths), 0, cls.shape[-1]))
    cls = normalize(cls, axis=1)
    patches = normalize(patches, axis=-1) if patches.size else patches
    return cls, patches


def normalize(values: np.ndarray, axis: int, eps: float = 1e-12) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=axis, keepdims=True), eps)


def patch_inside_mask(grid_shape: tuple[int, int], width: int, height: int, face_boxes: list[list[float]]) -> np.ndarray:
    grid_h, grid_w = grid_shape
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for gy in range(grid_h):
        cy = (gy + 0.5) / grid_h * height
        for gx in range(grid_w):
            cx = (gx + 0.5) / grid_w * width
            for box in face_boxes:
                if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                    mask[gy, gx] = True
                    break
    return mask


def save_overlay(image_record: dict[str, Any], similarity: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    with Image.open(image_record["image"]) as image:
        width, height = image.size
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(image.convert("RGB"))
        ax.imshow(similarity, cmap="jet", alpha=0.45, extent=[0, width, height, 0], vmin=-1.0, vmax=1.0)
        for box in image_record["face_bboxes"]:
            x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.8, edgecolor="lime", facecolor="none"))
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def write_html_gallery(crop_records: list[dict[str, Any]], image_records: list[dict[str, Any]], output_path: Path) -> None:
    base = output_path.parent
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>DINOv3 crop and patch visual check</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:14px}.card img{max-width:160px;max-height:130px;border:1px solid #bbb}.meta{font-size:12px;color:#444;word-break:break-word}</style>",
        "</head><body><h1>DINOv3 crop and patch visual check</h1>",
        "<p>These files are for visual inspection. Hard negative is currently face-sized random background that does not overlap GT faces.</p>",
    ]
    for crop_type in ["face", "head", "background", "hard_negative"]:
        lines.append(f"<h2>{html.escape(crop_type)}</h2><div class='grid'>")
        for record in [item for item in crop_records if item["crop_type"] == crop_type][:60]:
            rel = rel_path(record["crop_path"], base)
            lines.append(
                "<div class='card'>"
                f"<img src='{html.escape(rel)}'>"
                f"<div class='meta'>{html.escape(record['source'])}<br>{html.escape(record['selection_stratum'])}<br>{html.escape(record['crop_id'])}</div>"
                "</div>"
            )
        lines.append("</div>")
    lines.append("<h2>patch similarity maps</h2><div class='grid'>")
    for record in image_records:
        overlay = record.get("overlay_path")
        if overlay:
            lines.append(
                "<div class='card'>"
                f"<img src='{html.escape(rel_path(overlay, base))}'>"
                f"<div class='meta'>{html.escape(record['source'])}<br>{html.escape(record['selection_stratum'])}</div>"
                "</div>"
            )
    lines.append("</div></body></html>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rel_path(path: str | Path, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def main() -> int:
    args = parse_args()
    import torch
    from PIL import Image as ImageModule
    from torchvision.transforms import v2

    rows = read_manifest(args.manifest)
    if args.max_patch_images:
        patch_rows = rows[: args.max_patch_images]
    else:
        patch_rows = rows
    metadata = metadata_index(Path(rows[0]["image_path"]).parents[2], args.metadata)
    crop_records, image_records = generate_crops(rows, metadata, args.output_dir, args.seed)
    write_crop_metadata(crop_records, args.output_dir / "crop_metadata.csv")

    figures_dir = args.output_dir / "figures"
    for crop_type in ["face", "head", "background", "hard_negative"]:
        write_contact_sheet(crop_records, crop_type, figures_dir / f"contact_sheet_{crop_type}.png")

    face_crop_paths = [Path(record["crop_path"]) for record in crop_records if record["crop_type"] == "face"]
    model = load_dinov3(args.dinov3_repo, args.weights, torch)
    transform = make_transform(args.resize_size, torch, v2)

    face_features = []
    for start in range(0, len(face_crop_paths), args.batch_size):
        cls, _ = feature_batch(face_crop_paths[start : start + args.batch_size], model, transform, ImageModule, torch)
        face_features.append(cls)
    if not face_features:
        raise RuntimeError("No face crops were available for a DINOv3 face prototype.")
    face_prototype = normalize(np.concatenate(face_features, axis=0).mean(axis=0, keepdims=True), axis=1)[0]

    metrics_rows = []
    image_paths = [Path(record["image"]) for record in image_records[: len(patch_rows)]]
    overlay_dir = args.output_dir / "overlays" / "patch_similarity_maps"
    for start in range(0, len(image_paths), args.batch_size):
        batch_records = image_records[start : start + args.batch_size]
        batch_paths = image_paths[start : start + args.batch_size]
        _, patches = feature_batch(batch_paths, model, transform, ImageModule, torch)
        for offset, image_record in enumerate(batch_records):
            token_count = patches.shape[1]
            side = int(round(token_count**0.5))
            grid = patches[offset].reshape(side, side, patches.shape[-1])
            similarity = grid @ face_prototype
            mask = patch_inside_mask((side, side), int(image_record["width"]), int(image_record["height"]), image_record["face_bboxes"])
            outside = ~mask
            inside_mean = float(similarity[mask].mean()) if mask.any() else None
            outside_mean = float(similarity[outside].mean()) if outside.any() else None
            margin = inside_mean - outside_mean if inside_mean is not None and outside_mean is not None else None
            overlay_path = overlay_dir / f"{int(image_record['image_index']):02d}_{Path(image_record['image']).stem}.png"
            save_overlay(image_record, similarity, overlay_path)
            image_record["overlay_path"] = str(overlay_path)
            metrics_rows.append(
                {
                    "image": image_record["image"],
                    "source": image_record["source"],
                    "selection_stratum": image_record["selection_stratum"],
                    "face_count": len(image_record["face_bboxes"]),
                    "bbox_inside_similarity_mean": inside_mean,
                    "bbox_outside_similarity_mean": outside_mean,
                    "inside_outside_margin": margin,
                    "overlay_path": str(overlay_path),
                }
            )

    metrics_path = args.output_dir / "patch_similarity_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "image",
            "source",
            "selection_stratum",
            "face_count",
            "bbox_inside_similarity_mean",
            "bbox_outside_similarity_mean",
            "inside_outside_margin",
            "overlay_path",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    write_html_gallery(crop_records, image_records, args.output_dir / "crop_patch_gallery.html")
    summary = {
        "crop_count": len(crop_records),
        "face_crop_count": sum(1 for record in crop_records if record["crop_type"] == "face"),
        "head_crop_count": sum(1 for record in crop_records if record["crop_type"] == "head"),
        "background_crop_count": sum(1 for record in crop_records if record["crop_type"] == "background"),
        "hard_negative_crop_count": sum(1 for record in crop_records if record["crop_type"] == "hard_negative"),
        "patch_similarity_map_count": len(metrics_rows),
        "output_dir": str(args.output_dir),
        "gallery": str(args.output_dir / "crop_patch_gallery.html"),
        "patch_similarity_metrics": str(metrics_path),
        "note": "DINOv3 is frozen and runs on CPU. Hard negatives are face-sized random background crops with low IoU to GT faces.",
    }
    (args.output_dir / "crop_patch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
