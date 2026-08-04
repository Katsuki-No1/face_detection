import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "script"
    / "script_data"
    / "recover_video_name_history.py"
)
SPEC = importlib.util.spec_from_file_location("recover_video_name_history", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
recovery = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recovery
SPEC.loader.exec_module(recovery)


class RecoverVideoNameHistoryTest(unittest.TestCase):
    def test_matches_history_name_to_extracted_frame_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "data" / "output" / "2026_01_01"
            history_dir = root / "history"
            output_dir.mkdir(parents=True)
            history_dir.mkdir()

            video_name = "ABC-001 recovered title.mp4"
            short_hash = hashlib.sha1(Path(video_name).stem.encode("utf-8")).hexdigest()[:8]
            matched_prefix = f"2026_01_01_ABC_{short_hash}"
            (output_dir / f"{matched_prefix}_frame_00001_000001.000s.jpg").touch()
            unresolved_prefix = "2026_01_01_XYZ_deadbeef"
            (output_dir / f"{unresolved_prefix}_frame_00001_000001.000s.jpg").touch()
            (history_dir / "session.jsonl").write_text(
                f'{{"output":"data/input/2026_01_01/{video_name}\\n"}}\n',
                encoding="utf-8",
            )

            candidates: dict[str, recovery.VideoCandidate] = {}
            prefix_locations: dict[str, set[str]] = {}
            recovery.scan_frame_prefixes(root, prefix_locations)
            recovery.scan_text_histories([history_dir], candidates)
            rows, unresolved_count = recovery.build_rows(
                candidates,
                prefix_locations,
                "2026-01-01T00:00:00+09:00",
            )

            recovered_row = next(row for row in rows if row["video_name"] == video_name)
            self.assertEqual(recovered_row["video_prefix"], matched_prefix)
            self.assertEqual(recovered_row["recovery_status"], "exact")
            self.assertEqual(unresolved_count, 1)
            self.assertTrue(
                any(
                    row["video_prefix"] == unresolved_prefix
                    and row["recovery_status"] == "unresolved"
                    for row in rows
                )
            )


if __name__ == "__main__":
    unittest.main()
