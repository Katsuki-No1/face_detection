import json
import tempfile
import unittest
from pathlib import Path

from src.phase1_dino.dataset import write_crop_metadata
from src.phase1_dino.make_report import make_report


class DinoPhase1ReportTest(unittest.TestCase):
    def test_report_html_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory) / "phase1"
            (out_dir / "features").mkdir(parents=True)
            (out_dir / "metrics").mkdir(parents=True)
            (out_dir / "figures").mkdir(parents=True)
            (out_dir / "retrieval").mkdir(parents=True)
            (out_dir / "features" / "feature_run_info.json").write_text(
                json.dumps(
                    {
                        "model_name": "mock_dino",
                        "device": "mock",
                        "frozen_backbone": True,
                        "fine_tuned": False,
                        "detection_head_trained": False,
                        "keypoint_head_trained": False,
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "extract_features_summary.json").write_text(
                json.dumps({"crop_features_shape": [2, 4]}),
                encoding="utf-8",
            )
            (out_dir / "analysis_summary.json").write_text(json.dumps({"feature_count": 2}), encoding="utf-8")
            (out_dir / "metrics" / "knn_metrics.json").write_text(
                json.dumps({"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}),
                encoding="utf-8",
            )
            (out_dir / "metrics" / "retrieval_metrics.json").write_text(
                json.dumps({"query_count": 1, "top1_precision": 1.0}),
                encoding="utf-8",
            )
            (out_dir / "metrics" / "similarity_metrics.csv").write_text(
                "pair_type,count,mean,std,median,p25,p75,separation_margin\nface-face,1,1,0,1,1,1,0.5\n",
                encoding="utf-8",
            )
            (out_dir / "retrieval" / "knn_retrieval.html").write_text("<html></html>", encoding="utf-8")
            write_crop_metadata(
                out_dir / "features" / "crop_features_meta.csv",
                [
                    {
                        "image_id": "img",
                        "image_path": "img.jpg",
                        "crop_id": "face_0",
                        "crop_type": "face",
                        "label": "face_like",
                        "bbox_xyxy": [0, 0, 10, 10],
                        "original_width": 100,
                        "original_height": 100,
                        "face_size_category": "medium",
                        "tags": ["face"],
                        "split": "test",
                        "object_index": 0,
                        "crop_path": "face_0.jpg",
                    }
                ],
            )
            summary = make_report({"output": {"output_dir": str(out_dir)}, "dino": {"model_name": "mock_dino"}})
            report = Path(summary["report"])
            self.assertTrue(report.exists())
            text = report.read_text(encoding="utf-8")
            self.assertIn("fine-tuneしていません", text)
            self.assertIn("Precision / Recall / mAP", text)


if __name__ == "__main__":
    unittest.main()
