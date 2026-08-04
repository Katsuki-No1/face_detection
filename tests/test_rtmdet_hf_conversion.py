from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script.script_data.convert_face_pose_metadata_to_rtmdet_hf import (  # noqa: E402
    ConversionError,
    convert_dataset,
)


def shape(label: str, bbox: tuple[float, float, float, float]) -> dict[str, object]:
    x1, y1, x2, y2 = bbox
    return {
        "label": label,
        "bbox": {"xtl": x1, "ytl": y1, "xbr": x2, "ybr": y2},
    }


def make_source(root: Path, *, shared_video: bool = False) -> None:
    records = []
    for index, split in enumerate(("train", "val", "test"), start=1):
        name = f"video_{split}_frame_00001.jpg"
        relative = Path("images") / split / name
        image_path = root / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 80), (index * 40, index * 30, index * 20)).save(image_path)
        records.append(
            {
                "annotation_kind": f"task_{split}",
                "image_stem": (
                    "shared_video" if shared_video and split != "test" else f"video_{split}"
                ),
                "image_name": name,
                "image_path": str(image_path),
                "output_image": relative.as_posix(),
                "split": split,
                "negative_sample": split == "val",
                "heads": [shape("Head", (-10, 5, 40, 60))] if split != "test" else [],
                "faces": [shape("Face", (10, 15, 30, 45))] if split != "val" else [],
            }
        )
    metadata = root / "metadata" / "conversion_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"records": records}), encoding="utf-8")


class RtmDetHfConversionTest(unittest.TestCase):
    def test_converts_head_and_face_and_keeps_head_only_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            output = base / "output"
            make_source(source)

            summary = convert_dataset(source, output)

            self.assertEqual(summary["totals"]["images"], 3)
            self.assertEqual(summary["totals"]["head_annotations"], 2)
            self.assertEqual(summary["totals"]["face_annotations"], 2)
            self.assertEqual(summary["leakage_audit"]["expected_category_ids"], [1, 2])
            val = json.loads((output / "val.json").read_text(encoding="utf-8"))
            self.assertEqual(val["categories"][0]["name"], "head")
            self.assertEqual([row["category_id"] for row in val["annotations"]], [1])
            self.assertEqual(val["annotations"][0]["bbox"], [0.0, 5.0, 40.0, 55.0])
            self.assertTrue(val["images"][0]["metadata"]["source_negative_sample"])
            self.assertTrue((output / val["images"][0]["file_name"]).is_file())

    def test_rejects_cross_split_video_without_publishing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            output = base / "output"
            make_source(source, shared_video=True)

            with self.assertRaisesRegex(ConversionError, "cross dataset splits"):
                convert_dataset(source, output)
            self.assertFalse(output.exists())

    def test_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            output = base / "output"
            make_source(source)
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ConversionError, "refusing to overwrite"):
                convert_dataset(source, output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_reference_mode_writes_annotations_inside_source_without_copying_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = source / "annotations" / "rtmdet_hf"
            make_source(source)

            summary = convert_dataset(
                source,
                output,
                image_mode="reference",
                dataset_version="combined_v2",
            )

            self.assertEqual(summary["image_materialization"], {"reference": 3})
            self.assertEqual(summary["leakage_audit"]["data_root"], str(source.resolve()))
            self.assertFalse((output / "images").exists())
            train = json.loads((output / "train.json").read_text(encoding="utf-8"))
            self.assertEqual(train["images"][0]["file_name"], "images/train/video_train_frame_00001.jpg")
            self.assertEqual(
                train["images"][0]["metadata"]["source_image"],
                "images/train/video_train_frame_00001.jpg",
            )
            self.assertIn("annotations only", (output / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
