import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.phase1_dino.dataset import write_crop_metadata
from src.phase1_dino.metrics import write_knn_metrics, write_similarity_metrics


class DinoPhase1MetricsTest(unittest.TestCase):
    def records(self) -> list[dict]:
        rows = []
        specs = [
            ("face_0", "face", "face_like", [1.0, 0.0]),
            ("face_1", "face", "face_like", [0.9, 0.1]),
            ("background_0", "background", "non_face", [-1.0, 0.0]),
            ("hard_negative_0", "hard_negative", "non_face", [-0.8, 0.2]),
        ]
        for crop_id, crop_type, label, _feature in specs:
            rows.append(
                {
                    "image_id": "img",
                    "image_path": "img.jpg",
                    "crop_id": crop_id,
                    "crop_type": crop_type,
                    "label": label,
                    "bbox_xyxy": [0, 0, 10, 10],
                    "original_width": 100,
                    "original_height": 100,
                    "face_size_category": "medium" if label == "face_like" else "none",
                    "tags": [crop_type],
                    "split": "test",
                    "object_index": 0,
                    "crop_path": f"{crop_id}.jpg",
                }
            )
        return rows

    def test_knn_and_similarity_metrics_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self.records()
            features = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.8, 0.2]], dtype=np.float32)
            write_crop_metadata(root / "features" / "crop_features_meta.csv", records)
            knn = write_knn_metrics(features, records, root / "metrics" / "knn_metrics.json", k=1)
            self.assertTrue((root / "metrics" / "knn_metrics.json").exists())
            self.assertEqual(knn["confusion_matrix"]["fp"], 0)
            rows = write_similarity_metrics(
                features,
                records,
                root / "metrics" / "similarity_metrics.csv",
                root / "figures" / "similarity_histogram.png",
                max_pairs=100,
                seed=1,
            )
            self.assertTrue((root / "metrics" / "similarity_metrics.csv").exists())
            self.assertTrue((root / "figures" / "similarity_histogram.png").exists())
            self.assertEqual({row["pair_type"] for row in rows}, {"face-face", "face-background", "background-background", "face-hard_negative"})


if __name__ == "__main__":
    unittest.main()
