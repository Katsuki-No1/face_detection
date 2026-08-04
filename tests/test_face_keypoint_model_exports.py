from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

from script.script_data.export_face_keypoints_deim_dino import export_deim_dino
from script.script_data.export_face_keypoints_rtmpose import export_rtmpose


def _canonical_dataset(root: Path) -> None:
    image_path = root / "images/train/sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), (128, 128, 128)).save(image_path)
    metadata = {
        "schema_version": 1,
        "keypoint_names": ["e1", "e2", "n", "m1", "m2"],
        "records": [
            {
                "annotation_kind": "unit",
                "image_stem": "source_video",
                "image_path": "source/sample.jpg",
                "output_image": "images/train/sample.jpg",
                "split": "train",
                "objects": [
                    {
                        "face_index": 0,
                        "face": {"occluded": True},
                        "cvat_ellipse": {
                            "cx": 50.0,
                            "cy": 40.0,
                            "rx": 20.0,
                            "ry": 30.0,
                            "rotation": 0.0,
                        },
                        "training_bbox": {
                            "xtl": 20.0,
                            "ytl": 10.0,
                            "xbr": 80.0,
                            "ybr": 70.0,
                        },
                        "keypoints": {
                            "e1": {"x": 35.0, "y": 30.0, "v": 2, "occluded": False},
                            "e2": {"x": 65.0, "y": 30.0, "v": 2, "occluded": True},
                            "n": {"x": 101.0, "y": 40.0, "v": 2, "occluded": True},
                            "m1": {"x": 40.0, "y": 55.0, "v": 2, "occluded": False},
                            "m2": {"x": 60.0, "y": 55.0, "v": 2, "occluded": False},
                        },
                    }
                ],
            }
        ],
    }
    metadata_path = root / "metadata/conversion_metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_model_specific_exports_preserve_occlusion_and_deim_ellipse(tmp_path: Path) -> None:
    _canonical_dataset(tmp_path)
    rtm_summary = export_rtmpose(tmp_path)
    deim_summary = export_deim_dino(tmp_path)

    rtm = json.loads(
        (tmp_path / "annotations/rtmpose/person_keypoints_train.json").read_text()
    )
    deim = json.loads(
        (tmp_path / "annotations/deim_dino/face_keypoints_train.json").read_text()
    )
    rtm_annotation = rtm["annotations"][0]
    deim_annotation = deim["annotations"][0]

    assert rtm_annotation["category_id"] == 1
    assert deim_annotation["category_id"] == 2
    assert rtm_annotation["keypoints"] == deim_annotation["keypoints"]
    assert rtm_annotation["keypoints"][2::3] == [2, 1, 0, 2, 2]
    assert deim_annotation["attributes"]["face_occluded"] is True
    assert deim_annotation["attributes"]["occluded_keypoint_names"] == ["right_eye"]
    assert deim_annotation["ellipse"] == [
        50.0,
        40.0,
        30.0,
        20.0,
        -math.pi * 0.5,
    ]
    assert rtm_annotation["ellipse"] == deim_annotation["ellipse"]
    assert rtm_summary["splits"]["train"]["visibility_counts"]["1"] == 1
    assert deim_summary["splits"]["train"]["ellipses"] == 1
    assert not (tmp_path / "labels").exists()
