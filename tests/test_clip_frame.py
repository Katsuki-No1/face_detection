import hashlib
import importlib.util
import io
import shutil
import subprocess
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "script" / "script_data" / "clip_frame.py"
SPEC = importlib.util.spec_from_file_location("clip_frame", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
clip_frame = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clip_frame)


def make_args(input_dir: Path, output_dir: str | None = None, **overrides) -> Namespace:
    values = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "frame_count": 5,
        "recursive": False,
        "quality": 2,
        "full_decode_check": False,
        "video_history_csv": input_dir.parent / "video_name_history.csv",
        "allow_known_video_names": False,
    }
    values.update(overrides)
    return Namespace(**values)


class ClipFrameTest(unittest.TestCase):
    def test_normalizes_video_names_for_duplicate_checks(self) -> None:
        self.assertEqual(
            clip_frame.normalize_video_stem("ＡＢＣ-001.MP4"),
            clip_frame.normalize_video_stem("abc-001.mkv"),
        )

    def test_rejects_name_already_present_in_history(self) -> None:
        video = Path("input") / "Example Video.mp4"
        history_rows = [
            {
                "video_name": "example video.mkv",
                "normalized_stem": clip_frame.normalize_video_stem("example video.mkv"),
            }
        ]

        with self.assertRaisesRegex(clip_frame.CliError, "Previously seen video name"):
            clip_frame.check_video_name_duplicates(
                [video],
                history_rows,
                allow_known_names=False,
            )

    def test_rejects_renamed_video_when_content_hash_was_used_before(self) -> None:
        video = Path("input") / "renamed.mp4"
        fingerprints = {
            video: clip_frame.VideoFingerprint(
                content_sha256="a" * 64,
                file_size_bytes=123,
            )
        }
        history_rows = [
            {
                "video_name": "original-name.mkv",
                "normalized_stem": clip_frame.normalize_video_stem("original-name.mkv"),
                "content_sha256": "a" * 64,
            }
        ]

        with self.assertRaisesRegex(clip_frame.CliError, "重複あり"):
            clip_frame.check_video_duplicates(
                [video],
                fingerprints,
                history_rows,
                allow_known_videos=False,
            )

    def test_rejects_identical_content_twice_in_current_input(self) -> None:
        first = Path("input") / "first.mp4"
        second = Path("input") / "renamed-second.mp4"
        fingerprint = clip_frame.VideoFingerprint(
            content_sha256="b" * 64,
            file_size_bytes=456,
        )

        with self.assertRaisesRegex(clip_frame.CliError, "内容が同一"):
            clip_frame.check_video_duplicates(
                [first, second],
                {first: fingerprint, second: fingerprint},
                [],
                allow_known_videos=False,
            )

    def test_searches_video_csv_and_legacy_manifest_in_past_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_output = root / "data" / "output" / "csv-run"
            manifest_output = root / "data" / "output" / "manifest-run"
            csv_output.mkdir(parents=True)
            manifest_output.mkdir(parents=True)
            clip_frame.write_video_history(
                csv_output / "video_names.csv",
                [
                    {
                        "video_name": "from-csv.mp4",
                        "normalized_stem": "from-csv",
                    }
                ],
            )
            (manifest_output / "selection_manifest.jsonl").write_text(
                '{"video_name":"from-manifest.mov","video":"old/from-manifest.mov"}\n'
                '{"source_video":"from-active-learning.mp4"}\n',
                encoding="utf-8",
            )

            rows = clip_frame.search_past_clip_history(root)

        self.assertEqual(
            {row["video_name"] for row in rows},
            {
                "from-csv.mp4",
                "from-manifest.mov",
                "from-active-learning.mp4",
            },
        )

    def test_fingerprint_detects_file_content_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.write_bytes(b"same-video-bytes")

            fingerprint = clip_frame.fingerprint_video(video)

        self.assertEqual(fingerprint.file_size_bytes, 16)
        self.assertEqual(
            fingerprint.content_sha256,
            hashlib.sha256(b"same-video-bytes").hexdigest(),
        )

    def test_builds_interval_center_indices(self) -> None:
        self.assertEqual(
            clip_frame.build_uniform_frame_indices(total_frames=1_000, frame_count=10),
            [50, 150, 250, 350, 450, 550, 650, 750, 850, 950],
        )

    def test_rounding_keeps_indices_unique_and_evenly_spaced(self) -> None:
        indices = clip_frame.build_uniform_frame_indices(total_frames=10_003, frame_count=100)
        gaps = [right - left for left, right in zip(indices, indices[1:])]

        self.assertEqual(len(indices), 100)
        self.assertEqual(len(set(indices)), 100)
        self.assertEqual(indices, sorted(indices))
        self.assertLessEqual(max(gaps) - min(gaps), 1)

    def test_rejects_video_shorter_than_requested_frame_count(self) -> None:
        with self.assertRaisesRegex(clip_frame.CliError, "100 unique frames were requested"):
            clip_frame.build_uniform_frame_indices(total_frames=99, frame_count=100)

    def test_extract_command_decodes_once_and_selects_exact_indices(self) -> None:
        completed = subprocess.CompletedProcess(args=["ffmpeg"], returncode=0, stdout="", stderr="")
        with patch.object(clip_frame.subprocess, "run", return_value=completed) as run:
            clip_frame.extract_selected_frames(
                Path("video.mp4"),
                Path("frame_%05d.jpg"),
                [5, 15, 25],
                quality=2,
            )

        command = run.call_args.args[0]
        self.assertEqual(command.count("-i"), 1)
        self.assertIn("select=eq(n\\,5)+eq(n\\,15)+eq(n\\,25)", command)
        self.assertIn("-xerror", command)
        self.assertNotIn("-ss", command)

    def test_decode_error_is_reported(self) -> None:
        failed = subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=1,
            stdout="",
            stderr="Invalid NAL unit size",
        )
        stdout = io.StringIO()
        with patch.object(clip_frame.subprocess, "run", return_value=failed):
            with redirect_stdout(stdout):
                with self.assertRaisesRegex(clip_frame.CliError, "corrupt or incomplete"):
                    clip_frame.extract_selected_frames(
                        Path("broken.mp4"),
                        Path("frame_%05d.jpg"),
                        [0, 10],
                        quality=2,
                    )

        self.assertIn("Video problem detected", stdout.getvalue())
        self.assertIn("broken.mp4", stdout.getvalue())

    def test_packet_integrity_rejects_ffprobe_warning(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=None,
            stderr=None,
        )

        def write_warning(*_args, **kwargs):
            kwargs["stderr"].write("Invalid NAL unit size (0 > 12345)\n")
            return completed

        with patch.object(clip_frame.subprocess, "run", side_effect=write_warning) as run:
            with self.assertRaisesRegex(clip_frame.CliError, "packet/NAL integrity"):
                clip_frame.check_video_packet_integrity(Path("broken.mp4"))

        self.assertEqual(run.call_args.kwargs["stdout"], subprocess.DEVNULL)
        self.assertIn("-show_packets", run.call_args.args[0])

    def test_main_validates_all_videos_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            video = input_dir / "broken.mp4"
            video.write_bytes(b"broken")
            args = make_args(input_dir, "unused-test-output")
            with (
                patch.object(clip_frame, "parse_args", return_value=args),
                patch.object(clip_frame, "require_command"),
                patch.object(clip_frame, "find_videos", return_value=[video]),
                patch.object(
                    clip_frame,
                    "validate_videos",
                    side_effect=clip_frame.CliError("broken input"),
                ) as validate,
                patch.object(clip_frame, "extract_video_frames") as extract,
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    with self.assertRaisesRegex(clip_frame.CliError, "broken input"):
                        clip_frame.main()

            self.assertFalse(validate.call_args.kwargs["decode_check"])
            self.assertTrue(validate.call_args.kwargs["packet_check"])
            extract.assert_not_called()
            self.assertIn("0 new frames were written", stdout.getvalue())
            self.assertIn("existing output files were left unchanged", stdout.getvalue())

    def test_main_rejects_used_content_before_video_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "input"
            input_dir.mkdir()
            video = input_dir / "renamed.mp4"
            video.write_bytes(b"previously-used-content")
            args = make_args(input_dir, "unused-test-output")
            clip_frame.write_video_history(
                args.video_history_csv,
                [
                    {
                        "video_name": "original.mp4",
                        "normalized_stem": "original",
                        "content_sha256": hashlib.sha256(
                            b"previously-used-content"
                        ).hexdigest(),
                    }
                ],
            )

            with (
                patch.object(clip_frame, "parse_args", return_value=args),
                patch.object(clip_frame, "require_command"),
                patch.object(clip_frame, "search_past_clip_history", return_value=[]),
                patch.object(clip_frame, "validate_videos") as validate,
                patch.object(clip_frame, "extract_video_frames") as extract,
            ):
                with self.assertRaisesRegex(clip_frame.CliError, "重複あり"):
                    clip_frame.main()

            validate.assert_not_called()
            extract.assert_not_called()

    def test_main_reports_overlap_when_override_allows_reextraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "input"
            input_dir.mkdir()
            video = input_dir / "renamed.mp4"
            video.write_bytes(b"previously-used-content")
            args = make_args(
                input_dir,
                "unused-test-output",
                allow_known_video_names=True,
            )
            clip_frame.write_video_history(
                args.video_history_csv,
                [
                    {
                        "video_name": "original.mp4",
                        "normalized_stem": "original",
                        "content_sha256": hashlib.sha256(
                            b"previously-used-content"
                        ).hexdigest(),
                    }
                ],
            )

            stdout = io.StringIO()
            with (
                patch.object(clip_frame, "parse_args", return_value=args),
                patch.object(clip_frame, "require_command"),
                patch.object(clip_frame, "search_past_clip_history", return_value=[]),
                patch.object(
                    clip_frame,
                    "validate_videos",
                    side_effect=clip_frame.CliError("stop after duplicate report"),
                ),
                redirect_stdout(stdout),
            ):
                with self.assertRaisesRegex(clip_frame.CliError, "stop after duplicate report"):
                    clip_frame.main()

            report = stdout.getvalue()
            self.assertIn("重複あり", report)
            self.assertIn("renamed.mp4", report)
            self.assertIn("original.mp4", report)
            self.assertIn("再抽出を続行します", report)

    def test_staging_failure_leaves_existing_output_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "final-output"
            input_dir.mkdir()
            output_dir.mkdir()
            first = input_dir / "first.mp4"
            second = input_dir / "second.mp4"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            old_path = output_dir / "existing.jpg"
            old_path.write_bytes(b"existing")
            args = make_args(input_dir, str(output_dir))

            def stage_then_fail(**kwargs):
                staging_root = kwargs["output_root"]
                if kwargs["video_path"] == first:
                    for index in range(args.frame_count):
                        (staging_root / f"temporary_{index}.jpg").write_bytes(b"temporary")
                    return args.frame_count
                raise clip_frame.CliError("second video is corrupt")

            with (
                patch.object(clip_frame, "parse_args", return_value=args),
                patch.object(clip_frame, "require_command"),
                patch.object(clip_frame, "find_videos", return_value=[first, second]),
                patch.object(
                    clip_frame,
                    "validate_videos",
                    return_value={first: 10.0, second: 10.0},
                ),
                patch.object(clip_frame, "get_frame_count", return_value=300),
                patch.object(clip_frame, "extract_video_frames", side_effect=stage_then_fail),
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    with self.assertRaisesRegex(clip_frame.CliError, "second video is corrupt"):
                        clip_frame.main()

            self.assertEqual(old_path.read_bytes(), b"existing")
            self.assertEqual(list(output_dir.glob("*.jpg")), [old_path])
            self.assertFalse(any(root.glob(".final-output_staging_*")))
            self.assertIn("all temporary frames were discarded", stdout.getvalue())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is required")
class ClipFrameIntegrationTest(unittest.TestCase):
    @staticmethod
    def create_video(video_path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=30:duration=4",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(video_path),
            ],
            check=True,
        )

    def test_extracts_evenly_spaced_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            video_path = input_dir / "sample.mp4"
            self.create_video(video_path)

            duration = clip_frame.get_duration_seconds(video_path)
            total_frames = clip_frame.get_frame_count(video_path)
            extracted_count = clip_frame.extract_video_frames(
                video_path=video_path,
                input_dir=input_dir,
                output_root=output_dir,
                frame_count=5,
                quality=2,
                duration=duration,
                total_frames=total_frames,
            )

            output_paths = sorted(output_dir.glob("*.jpg"))
            selected_indices = [
                int(path.stem.rsplit("_source_", maxsplit=1)[1]) for path in output_paths
            ]
            self.assertEqual(extracted_count, 5)
            self.assertEqual(len(output_paths), 5)
            self.assertEqual(selected_indices, [12, 36, 60, 84, 108])

    def test_main_commits_images_and_video_name_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            video_path = input_dir / "sample.mp4"
            self.create_video(video_path)
            old_manifest = output_dir / "selection_manifest.jsonl"
            old_manifest.write_text('{"dino": true}\n', encoding="utf-8")
            args = make_args(input_dir, str(output_dir))

            stdout = io.StringIO()
            with patch.object(clip_frame, "parse_args", return_value=args):
                with redirect_stdout(stdout):
                    clip_frame.main()

            self.assertEqual(len(list(output_dir.glob("*.jpg"))), 5)
            self.assertTrue((output_dir / "video_names.csv").is_file())
            self.assertTrue(args.video_history_csv.is_file())
            history_text = args.video_history_csv.read_text(encoding="utf-8-sig")
            self.assertIn("sample.mp4", history_text)
            self.assertIn("content_sha256", history_text.splitlines()[0])
            self.assertFalse(old_manifest.exists())
            self.assertFalse(any(root.glob(".output_staging_*")))
            self.assertIn("Packet check OK", stdout.getvalue())
            self.assertIn("Committed 5 frame(s)", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
