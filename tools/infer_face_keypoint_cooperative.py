from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_face_keypoint_cooperative import (  # noqa: E402
    image_tensor,
    scale_decoded,
    tensor_lists,
)
from face_keypoint_cooperative_common import (  # noqa: E402
    autocast_context,
    build_model,
    configure_device_math,
    load_cooperative_config,
    resolve_path,
)
from src.head_pose_geometry.heads import (  # noqa: E402
    CachedHeadDetector,
    HeadDetection,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = ("cyan", "lime", "yellow", "magenta", "orange")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save independent and refined cooperative predictions."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--head-cache",
        type=Path,
        help=(
            "Schema-v1 predicted HEAD cache keyed by input path or file name; "
            "required when head-pose geometry is enabled"
        ),
    )
    parser.add_argument("--save-overlays", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    return sorted(
        value
        for value in path.rglob("*")
        if value.is_file() and value.suffix.lower() in IMAGE_EXTENSIONS
    )


def draw_output(
    image: Image.Image,
    decoded: dict[str, Any],
    *,
    refined: bool,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    face = decoded["refined_face" if refined else "independent_face"]
    pose = decoded["refined_pose" if refined else "independent_pose"]
    for box, score in zip(face["boxes"], face["scores"], strict=True):
        draw.rectangle(tuple(float(value) for value in box.tolist()), outline="red", width=3)
        draw.text((float(box[0]), float(box[1])), f"face {float(score):.2f}", fill="red")
    for keypoints, score in zip(pose["keypoints"], pose["scores"], strict=True):
        for point_index, point in enumerate(keypoints):
            x, y = (float(value) for value in point.tolist())
            radius = 3
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=COLORS[point_index],
            )
        draw.text(
            tuple(float(value) for value in keypoints[0].tolist()),
            f"pose {float(score):.2f}",
            fill="cyan",
        )
    return result


def main() -> int:
    args = parse_args()
    config = load_cooperative_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    configure_device_math(device, config["train"])
    model = build_model(
        config,
        device=device,
        checkpoint=resolve_path(args.checkpoint),
    ).eval()
    head_detector = (
        CachedHeadDetector(resolve_path(args.head_cache))
        if args.head_cache is not None
        else None
    )
    if (
        (
            model.head_pose_geometry is not None
            or model.external_head_cascade is not None
        )
        and model.shared_head_branch is None
    ) and head_detector is None:
        raise ValueError(
            "--head-cache is required when the three-stage cascade is enabled"
        )
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = image_paths(resolve_path(args.input))
    input_size = int(config["data"].get("image_size", 960))
    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            image = Image.open(path).convert("RGB")
            tensor, original_hw = image_tensor(image, input_size)
            head_detections = None
            if head_detector is not None:
                cache_key = next(
                    (
                        key
                        for key in (str(path), path.name, path.stem)
                        if key in head_detector.entries
                    ),
                    None,
                )
                if cache_key is None:
                    raise KeyError(f"No HEAD cache entry for {path}")
                original_h, original_w = original_hw
                detected = head_detector.detect(
                    torch.empty((3, original_h, original_w), device="meta"),
                    image_id=cache_key,
                )
                scale_x = input_size / float(original_w)
                scale_y = input_size / float(original_h)
                head_detections = [[
                    HeadDetection(
                        box_xyxy=(
                            item.box_xyxy[0] * scale_x,
                            item.box_xyxy[1] * scale_y,
                            item.box_xyxy[2] * scale_x,
                            item.box_xyxy[3] * scale_y,
                        ),
                        confidence=item.confidence,
                        instance_id=item.instance_id,
                        valid_image_region_xyxy=(
                            item.valid_image_region_xyxy[0] * scale_x,
                            item.valid_image_region_xyxy[1] * scale_y,
                            item.valid_image_region_xyxy[2] * scale_x,
                            item.valid_image_region_xyxy[3] * scale_y,
                        ),
                        source=item.source,
                    )
                    for item in detected
                ]]
            with torch.inference_mode(), autocast_context(device, config["train"]):
                raw = model(
                    tensor.to(device),
                    return_raw=True,
                    head_detections=head_detections,
                )
                decoded = model.decode(raw, image_size=(input_size, input_size))[0]
            decoded = scale_decoded(
                decoded,
                input_size=input_size,
                original_hw=original_hw,
            )
            row = {
                "path": str(path),
                "independent_face": tensor_lists(decoded["independent_face"]),
                "independent_pose": tensor_lists(decoded["independent_pose"]),
                "refined_face": tensor_lists(decoded["refined_face"]),
                "refined_pose": tensor_lists(decoded["refined_pose"]),
                "associations": tensor_lists(decoded["associations"]),
            }
            if "head_pose_geometry" in decoded:
                row["head_pose_geometry"] = tensor_lists(
                    decoded["head_pose_geometry"]
                )
            if "external_head_cascade" in decoded:
                row["external_head_cascade"] = tensor_lists(
                    decoded["external_head_cascade"]
                )
            if "shared_head" in decoded:
                row["shared_head"] = tensor_lists(decoded["shared_head"])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if args.save_overlays:
                draw_output(image, decoded, refined=False).save(
                    output_dir / f"{path.stem}_independent.jpg"
                )
                draw_output(image, decoded, refined=True).save(
                    output_dir / f"{path.stem}_refined.jpg"
                )
    print(
        json.dumps(
            {
                "images": len(paths),
                "predictions": str(prediction_path),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
