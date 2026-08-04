from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script.script_data.convert_face_pose_yolo_to_coco import ConversionError, convert_dataset
from src.data.coco_split_audit import CocoSplitLeakageError, audit_coco_splits, sha256_file
from tools import train_dinov3_deimv2


YOLO_LINE = "0 0.5 0.5 0.4 0.5 0.4 0.4 2 0.6 0.4 2 0.5 0.5 2 0.45 0.6 2 0.55 0.6 2"


def make_dataset(root: Path, *, shared_video: bool = False) -> None:
    records = []
    colors = {"train": (255, 0, 0), "val": (0, 255, 0), "test": (0, 0, 255)}
    for index, split in enumerate(("train", "val", "test"), start=1):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        name = f"video_{split}_frame_00001.jpg"
        Image.new("RGB", (100, 80), colors[split]).save(image_dir / name)
        (label_dir / f"{Path(name).stem}.txt").write_text(
            "" if split == "val" else YOLO_LINE + "\n",
            encoding="utf-8",
        )
        records.append(
            {
                "annotation_kind": f"task_{index}",
                "image_stem": "same_video" if shared_video and split != "test" else f"video_{split}",
                "image_name": name,
                "image_path": str(image_dir / name),
                "output_image": f"images/{split}/{name}",
                "split": split,
            }
        )
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    (metadata_dir / "conversion_metadata.json").write_text(
        json.dumps({"records": records}),
        encoding="utf-8",
    )


class FacePoseCocoConversionTest(unittest.TestCase):
    def test_conversion_preserves_splits_negative_images_and_category_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            summary = convert_dataset(root)

            self.assertEqual(summary["totals"], {"images": 3, "annotations": 2, "source_videos": 3})
            self.assertEqual(summary["leakage_audit"]["status"], "ok")
            train = json.loads((root / "train.json").read_text(encoding="utf-8"))
            val = json.loads((root / "val.json").read_text(encoding="utf-8"))
            self.assertEqual(train["categories"], [{"id": 2, "name": "face", "supercategory": "person"}])
            self.assertEqual(train["annotations"][0]["category_id"], 2)
            self.assertEqual(train["annotations"][0]["bbox"], [30.0, 20.0, 40.0, 40.0])
            self.assertEqual(len(val["images"]), 1)
            self.assertEqual(val["annotations"], [])
            self.assertEqual(train["images"][0]["metadata"]["source_video_id"], "video_train")
            self.assertEqual(
                train["images"][0]["metadata"]["sha256"],
                sha256_file(root / train["images"][0]["file_name"]),
            )

            first = (root / "train.json").read_text(encoding="utf-8")
            convert_dataset(root)
            self.assertEqual((root / "train.json").read_text(encoding="utf-8"), first)

    def test_conversion_rejects_same_video_across_annotation_kinds_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root, shared_video=True)
            with self.assertRaisesRegex(ConversionError, "Source video ids cross"):
                convert_dataset(root)
            self.assertFalse((root / "train.json").exists())

    def test_audit_rejects_cross_split_hash_and_post_conversion_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            convert_dataset(root)
            train = json.loads((root / "train.json").read_text(encoding="utf-8"))
            val = json.loads((root / "val.json").read_text(encoding="utf-8"))
            train_path = root / train["images"][0]["file_name"]
            val_path = root / val["images"][0]["file_name"]
            val_path.write_bytes(train_path.read_bytes())
            val["images"][0]["metadata"]["sha256"] = sha256_file(val_path)
            (root / "val.json").write_text(json.dumps(val), encoding="utf-8")
            with self.assertRaisesRegex(CocoSplitLeakageError, "sha256"):
                audit_coco_splits(root)

            make_dataset(root)
            convert_dataset(root)
            val = json.loads((root / "val.json").read_text(encoding="utf-8"))
            val_path = root / val["images"][0]["file_name"]
            val_path.write_bytes(b"changed-after-conversion")
            with self.assertRaisesRegex(CocoSplitLeakageError, "SHA-256 changed"):
                audit_coco_splits(root)

    def test_conversion_failure_keeps_existing_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            (root / "train.json").write_text("existing", encoding="utf-8")
            metadata_path = root / "metadata" / "conversion_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["records"][0]["split"] = "val"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ConversionError, "Split mismatch"):
                convert_dataset(root)
            self.assertEqual((root / "train.json").read_text(encoding="utf-8"), "existing")


class TrainingLeakagePreflightTest(unittest.TestCase):
    def test_audit_failure_happens_before_output_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "training-output"
            args = Namespace(
                config=Path("config.yaml"),
                data_root=None,
                output_dir=output_dir,
                device="cpu",
                amp=None,
                seed=None,
                epochs=None,
                batch_size=None,
                num_workers=None,
                max_train_batches=None,
                resume_from=None,
                save_every=10,
                val_every=None,
                val_split=None,
                val_fixed_threshold=None,
                val_score_min=None,
                val_max_images=None,
                dry_run=True,
                synthetic=False,
                tiny_model=False,
                allow_dummy_backbone=False,
            )
            config = {
                "data": {
                    "data_root": "dataset",
                    "category_ids": [2],
                    "leakage_audit": {"enabled": True},
                },
                "train": {},
            }
            with (
                patch.object(train_dinov3_deimv2, "parse_args", return_value=args),
                patch.object(train_dinov3_deimv2, "load_config", return_value=config),
                patch.object(
                    train_dinov3_deimv2,
                    "resolve_data_paths",
                    return_value=(config, False),
                ),
                patch.object(
                    train_dinov3_deimv2,
                    "run_configured_leakage_audit",
                    side_effect=CocoSplitLeakageError("intentional leakage"),
                ),
                patch.object(train_dinov3_deimv2.train_impl, "dry_run") as dry_run,
            ):
                with self.assertRaisesRegex(CocoSplitLeakageError, "intentional leakage"):
                    train_dinov3_deimv2.main()
            self.assertFalse(output_dir.exists())
            dry_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
