import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".wmv",
    ".webm",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_HISTORY_CSV = PROJECT_ROOT / "data" / "video_name_history.csv"
VIDEO_HISTORY_FIELDS = [
    "video_name",
    "normalized_stem",
    "stem_sha1",
    "video_prefix",
    "source_path",
    "output_dir",
    "recorded_at",
    "evidence_source",
    "recovery_status",
    "content_sha256",
    "file_size_bytes",
]
FINGERPRINT_CHUNK_SIZE = 8 * 1024 * 1024


class CliError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoFingerprint:
    content_sha256: str
    file_size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract evenly spaced frames from each video in a directory."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        # 対象動画を指定したい場合は、実行時にこの引数へ動画が入ったディレクトリを渡します。
        # 例: python3 ./script/script_data/clip_frame.py ./data/input/2026_06_24
        help="Directory that contains target videos.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        # 省略時は入力フォルダ名と同じ名前で data/output 配下に出力します。
        # 例: 2026_06_17 を入力すると data/output/2026_06_17 に出力します。
        help="Directory name to create/use under ./data/output. Defaults to input_dir name.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=100,
        help="Number of evenly spaced frames to extract from each video. Default: 100.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search videos recursively under input_dir.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=2,
        help="JPEG quality for ffmpeg -q:v. Lower is better. Default: 2.",
    )
    parser.add_argument(
        "--full-decode-check",
        action="store_true",
        help=(
            "Also run a full decode before extraction. Normally the packet check and "
            "extraction decode are sufficient."
        ),
    )
    parser.add_argument(
        "--video-history-csv",
        type=Path,
        default=DEFAULT_VIDEO_HISTORY_CSV,
        help=(
            "Persistent CSV used to reject previously seen video names. "
            f"Default: {DEFAULT_VIDEO_HISTORY_CSV}."
        ),
    )
    parser.add_argument(
        "--allow-known-videos",
        "--allow-known-video-names",
        dest="allow_known_video_names",
        action="store_true",
        help=(
            "Allow intentional re-extraction of videos found in past clip records. "
            "--allow-known-video-names is kept as a compatibility alias."
        ),
    )
    return parser.parse_args()


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise CliError(f"{name} is not available. Please install ffmpeg.")


def find_videos(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def safe_name(value: str) -> str:
    return "".join("_" if char in '<>:"/\\|?*' else char for char in value).strip(" ._")


def frame_prefix(video_path: Path, input_dir: Path) -> str:
    folder_name = video_path.parent.name if video_path.parent != input_dir else input_dir.name
    date_part = safe_name(folder_name) or "unknown"
    video_part = safe_name(video_path.stem[:3]) or "vid"
    video_hash = hashlib.sha1(video_path.stem.encode("utf-8")).hexdigest()[:8]
    return f"{date_part}_{video_part}_{video_hash}"


def normalize_video_stem(video_name: str) -> str:
    """Normalize a filename for duplicate checks, ignoring its extension."""
    stem = Path(video_name).stem
    return unicodedata.normalize("NFKC", stem).casefold().strip()


def video_stem_sha1(video_name: str) -> str:
    return hashlib.sha1(Path(video_name).stem.encode("utf-8")).hexdigest()


def project_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def read_video_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        required = {"video_name"}
        if not required.issubset(fieldnames):
            raise CliError(
                f"Video history CSV has an unsupported header: {path}. "
                f"Required columns: {', '.join(sorted(required))}."
            )
        rows = []
        for source_row in reader:
            row = {field: source_row.get(field, "") for field in VIDEO_HISTORY_FIELDS}
            if row["video_name"] and not row["normalized_stem"]:
                row["normalized_stem"] = normalize_video_stem(row["video_name"])
            rows.append(row)
        return rows


def write_video_history(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8-sig",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=VIDEO_HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field, "") for field in VIDEO_HISTORY_FIELDS} for row in rows
            )
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_video_history_rows(
    videos: list[Path],
    input_dir: Path,
    output_root: Path,
    recorded_at: str,
    fingerprints: dict[Path, VideoFingerprint] | None = None,
    evidence_source: str = "clip_frame",
) -> list[dict[str, str]]:
    fingerprints = fingerprints or {}
    return [
        {
            "video_name": video_path.name,
            "normalized_stem": normalize_video_stem(video_path.name),
            "stem_sha1": video_stem_sha1(video_path.name),
            "video_prefix": frame_prefix(video_path, input_dir),
            "source_path": project_relative_path(video_path),
            "output_dir": project_relative_path(output_root),
            "recorded_at": recorded_at,
            "evidence_source": evidence_source,
            "recovery_status": "exact",
            "content_sha256": (
                fingerprints[video_path].content_sha256 if video_path in fingerprints else ""
            ),
            "file_size_bytes": (
                str(fingerprints[video_path].file_size_bytes)
                if video_path in fingerprints
                else ""
            ),
        }
        for video_path in videos
    ]


def read_selection_manifest_history(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        relative_output_dir = project_relative_path(path.parent)
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CliError(
                        f"Invalid JSON in past clip manifest {path}:{line_number}: {error}"
                    ) from error
                video_name = str(
                    record.get("video_name", "")
                    or record.get("source_video", "")
                ).strip()
                if not video_name:
                    source_path = str(
                        record.get("video", "")
                        or record.get("source_path", "")
                    ).strip()
                    video_name = Path(source_path).name
                if not video_name:
                    continue
                rows.append(
                    {
                        "video_name": video_name,
                        "normalized_stem": normalize_video_stem(video_name),
                        "stem_sha1": video_stem_sha1(video_name),
                        "video_prefix": str(record.get("video_prefix", "")),
                        "source_path": str(
                            record.get("video", "")
                            or record.get("source_path", "")
                        ),
                        "output_dir": relative_output_dir,
                        "recorded_at": "",
                        "evidence_source": "selection_manifest",
                        "recovery_status": "exact",
                        "content_sha256": str(record.get("content_sha256", "")),
                        "file_size_bytes": str(record.get("file_size_bytes", "")),
                    }
                )
    except OSError as error:
        raise CliError(f"Could not inspect past clip manifest {path}: {error}") from error
    return rows


def search_past_clip_history(project_root: Path) -> list[dict[str, str]]:
    """Search durable records created by previous clip runs."""
    output_root = project_root / "data" / "output"
    if not output_root.is_dir():
        return []

    rows: list[dict[str, str]] = []
    for csv_path in sorted(output_root.rglob("video_names.csv")):
        try:
            rows.extend(read_video_history(csv_path))
        except OSError as error:
            raise CliError(f"Could not inspect past clip CSV {csv_path}: {error}") from error
    for manifest_path in sorted(output_root.rglob("selection_manifest.jsonl")):
        rows.extend(read_selection_manifest_history(manifest_path))
    return rows


def fingerprint_video(video_path: Path) -> VideoFingerprint:
    digest = hashlib.sha256()
    try:
        before = video_path.stat()
        with video_path.open("rb") as file:
            while chunk := file.read(FINGERPRINT_CHUNK_SIZE):
                digest.update(chunk)
        after = video_path.stat()
    except OSError as error:
        raise CliError(f"Could not fingerprint video {video_path}: {error}") from error

    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise CliError(
            f"Video changed while it was being inspected: {video_path}. "
            "Wait until the copy/download is complete and retry."
        )
    return VideoFingerprint(
        content_sha256=digest.hexdigest(),
        file_size_bytes=after.st_size,
    )


def fingerprint_videos(videos: list[Path]) -> dict[Path, VideoFingerprint]:
    fingerprints: dict[Path, VideoFingerprint] = {}
    print("Fingerprinting input videos for duplicate detection...", flush=True)
    for video_path in videos:
        fingerprints[video_path] = fingerprint_video(video_path)
        print(f"Fingerprint OK: {video_path.name}", flush=True)
    return fingerprints


def find_batch_duplicate_names(videos: list[Path]) -> list[list[Path]]:
    paths_by_name: dict[str, list[Path]] = {}
    for video_path in videos:
        paths_by_name.setdefault(normalize_video_stem(video_path.name), []).append(video_path)
    return [paths for paths in paths_by_name.values() if len(paths) > 1]


def find_known_video_names(
    videos: list[Path],
    history_rows: list[dict[str, str]],
) -> list[tuple[Path, str]]:
    names_in_history = {
        row["normalized_stem"]: row["video_name"]
        for row in history_rows
        if row.get("normalized_stem") and row.get("video_name")
    }
    return [
        (video_path, names_in_history[normalized])
        for video_path in videos
        if (normalized := normalize_video_stem(video_path.name)) in names_in_history
    ]


def find_batch_duplicate_contents(
    videos: list[Path],
    fingerprints: dict[Path, VideoFingerprint],
) -> list[list[Path]]:
    paths_by_digest: dict[str, list[Path]] = {}
    for video_path in videos:
        paths_by_digest.setdefault(
            fingerprints[video_path].content_sha256,
            [],
        ).append(video_path)
    return [paths for paths in paths_by_digest.values() if len(paths) > 1]


def find_known_video_contents(
    videos: list[Path],
    fingerprints: dict[Path, VideoFingerprint],
    history_rows: list[dict[str, str]],
) -> list[tuple[Path, str]]:
    names_by_digest: dict[str, str] = {}
    for row in history_rows:
        digest = row.get("content_sha256", "").strip().casefold()
        if digest:
            names_by_digest.setdefault(digest, row.get("video_name", "") or "(unknown name)")
    return [
        (video_path, names_by_digest[fingerprint.content_sha256])
        for video_path in videos
        if (fingerprint := fingerprints[video_path]).content_sha256 in names_by_digest
    ]


def check_video_name_duplicates(
    videos: list[Path],
    history_rows: list[dict[str, str]],
    allow_known_names: bool,
) -> list[tuple[Path, str]]:
    batch_duplicates = find_batch_duplicate_names(videos)
    if batch_duplicates:
        detail = "\n".join(
            "  - " + ", ".join(str(path) for path in paths) for paths in batch_duplicates
        )
        raise CliError(
            "重複あり: 今回の入力内で動画名が重複しています。\n"
            f"Duplicate video names exist in the current input:\n{detail}"
        )

    known_names = find_known_video_names(videos, history_rows)
    if known_names and not allow_known_names:
        detail = "\n".join(
            f"  - {video_path.name} (history: {history_name})"
            for video_path, history_name in known_names
        )
        raise CliError(
            "重複あり: 過去に使用した動画名と一致しました。\n"
            "Previously seen video name(s) found. Frame extraction was not started:\n"
            f"{detail}\n"
            "Use --allow-known-video-names only when re-extraction is intentional."
        )
    return known_names


def format_known_video_matches(
    known_names: list[tuple[Path, str]],
    known_contents: list[tuple[Path, str]],
) -> str:
    matches_by_path: dict[Path, list[str]] = {}
    for video_path, history_name in known_names:
        matches_by_path.setdefault(video_path, []).append(
            f"動画名が一致（過去: {history_name}）"
        )
    for video_path, history_name in known_contents:
        matches_by_path.setdefault(video_path, []).append(
            f"動画内容のSHA-256が一致（過去: {history_name}）"
        )
    return "\n".join(
        f"  - {video_path.name}: {'、'.join(reasons)}"
        for video_path, reasons in matches_by_path.items()
    )


def check_video_duplicates(
    videos: list[Path],
    fingerprints: dict[Path, VideoFingerprint],
    history_rows: list[dict[str, str]],
    allow_known_videos: bool,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    batch_name_duplicates = find_batch_duplicate_names(videos)
    if batch_name_duplicates:
        detail = "\n".join(
            "  - " + ", ".join(str(path) for path in paths)
            for paths in batch_name_duplicates
        )
        raise CliError(
            "重複あり: 今回の入力内で動画名が重複しています。"
            "クリップ作成を開始しません。\n"
            f"{detail}"
        )

    batch_duplicates = find_batch_duplicate_contents(videos, fingerprints)
    if batch_duplicates:
        detail = "\n".join(
            "  - " + ", ".join(str(path) for path in paths) for paths in batch_duplicates
        )
        raise CliError(
            "重複あり: 今回の入力内に内容が同一の動画があります。"
            "クリップ作成を開始しません。\n"
            f"{detail}"
        )

    known_names = find_known_video_names(videos, history_rows)
    known_contents = find_known_video_contents(videos, fingerprints, history_rows)
    if (known_names or known_contents) and not allow_known_videos:
        detail = format_known_video_matches(known_names, known_contents)
        raise CliError(
            "重複あり: 過去に使用した動画と一致しました。"
            "クリップ作成を開始しません。\n"
            f"{detail}\n"
            "意図した再抽出の場合のみ --allow-known-videos を指定してください。"
        )
    return known_names, known_contents


def merge_semicolon_values(current: str, additional: str) -> str:
    return ";".join(
        dict.fromkeys(
            value
            for source in (current, additional)
            for value in source.split(";")
            if value
        )
    )


def merge_history_metadata(target: dict[str, str], source: dict[str, str]) -> None:
    for field in VIDEO_HISTORY_FIELDS:
        source_value = source.get(field, "")
        if not source_value:
            continue
        if field in {"source_path", "output_dir", "evidence_source", "video_prefix"}:
            target[field] = merge_semicolon_values(target.get(field, ""), source_value)
        elif not target.get(field):
            target[field] = source_value


def merge_video_history(
    existing_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    merged_rows = [
        {field: row.get(field, "") for field in VIDEO_HISTORY_FIELDS} for row in existing_rows
    ]
    added = 0
    for source_row in new_rows:
        row = {field: source_row.get(field, "") for field in VIDEO_HISTORY_FIELDS}
        normalized = row["normalized_stem"]
        digest = row["content_sha256"].casefold()

        same_digest = next(
            (
                candidate
                for candidate in merged_rows
                if digest and candidate.get("content_sha256", "").casefold() == digest
                and candidate.get("normalized_stem", "") == normalized
            ),
            None,
        )
        same_name_without_digest = next(
            (
                candidate
                for candidate in merged_rows
                if normalized
                and candidate.get("normalized_stem", "") == normalized
                and not candidate.get("content_sha256")
            ),
            None,
        )
        existing_match = same_digest or same_name_without_digest
        if existing_match is not None:
            merge_history_metadata(existing_match, row)
            continue

        if not digest:
            same_name = next(
                (
                    candidate
                    for candidate in merged_rows
                    if normalized and candidate.get("normalized_stem", "") == normalized
                ),
                None,
            )
            if same_name is not None:
                merge_history_metadata(same_name, row)
                continue

        merged_rows.append(row)
        added += 1
    return merged_rows, added


def command_error_message(result: subprocess.CompletedProcess[str]) -> str:
    message = (result.stderr or result.stdout or "").strip()
    if not message:
        return f"exit status {result.returncode}"

    lines = message.splitlines()
    if len(lines) > 8:
        lines = lines[:8] + ["..."]
    return "\n".join(lines)


def get_duration_seconds(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CliError(f"ffprobe failed:\n{command_error_message(result)}")

    try:
        duration = float(result.stdout.strip())
    except ValueError as error:
        raise CliError(f"ffprobe returned invalid duration: {result.stdout.strip()!r}") from error

    if not math.isfinite(duration) or duration <= 0:
        raise CliError(f"ffprobe returned invalid duration: {duration}")

    return duration


def get_frame_count(video_path: Path) -> int:
    """Return the decoded video frame count, using metadata when available."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CliError(
            f"ffprobe failed while reading frame count:\n{command_error_message(result)}"
        )

    value = result.stdout.strip()
    if value.isdigit() and int(value) > 0:
        return int(value)

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CliError(f"ffprobe failed while counting frames:\n{command_error_message(result)}")

    value = result.stdout.strip()
    if not value.isdigit() or int(value) <= 0:
        raise CliError(f"ffprobe returned invalid frame count: {value!r}")
    return int(value)


def build_uniform_frame_indices(total_frames: int, frame_count: int) -> list[int]:
    """Choose the center frame of each equally sized interval.

    Integer rounding can make adjacent gaps differ by one frame, but the
    indices are unique and cover the video uniformly.
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be greater than 0")
    if total_frames < frame_count:
        raise CliError(
            f"Video has {total_frames} frame(s), but {frame_count} unique frames were requested. "
            "Use a longer video or a smaller --frame-count."
        )

    return [
        ((2 * interval_index + 1) * total_frames) // (2 * frame_count)
        for interval_index in range(frame_count)
    ]


def check_video_decodable(video_path: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CliError(f"ffmpeg decode check failed:\n{command_error_message(result)}")


def check_video_packet_integrity(video_path: Path) -> None:
    """Scan every packet and reject parser warnings such as invalid NAL data."""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as warning_output:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "warning",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_entries",
                "packet=pts_time",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=warning_output,
            text=True,
        )
        warning_size = warning_output.tell()
        warning_output.seek(0)
        warning_lines = warning_output.read(8192).strip().splitlines()

    if result.returncode != 0 or warning_size:
        if len(warning_lines) > 8:
            warning_lines = warning_lines[:8] + ["..."]
        detail = "\n".join(warning_lines) or f"exit status {result.returncode}"
        raise CliError(
            "ffprobe packet/NAL integrity check failed. "
            f"The source video may be corrupt or incomplete.\n{detail}"
        )


def format_video_errors(errors: list[tuple[Path, Exception]]) -> str:
    detail = "\n\n".join(f"{path.name}\n{error}" for path, error in errors)
    return f"Video validation failed. Stop before extracting frames.\n\n{detail}"


def validate_videos(
    videos: list[Path],
    decode_check: bool,
    packet_check: bool = True,
) -> dict[Path, float]:
    durations: dict[Path, float] = {}
    metadata_errors: list[tuple[Path, Exception]] = []

    print("Checking video metadata...", flush=True)
    for video_path in videos:
        try:
            durations[video_path] = get_duration_seconds(video_path)
        except CliError as error:
            metadata_errors.append((video_path, error))

    if metadata_errors:
        message = format_video_errors(metadata_errors)
        print(f"Video problem detected: {message}", flush=True)
        raise CliError(message)

    if packet_check:
        packet_errors: list[tuple[Path, Exception]] = []
        print("Checking video packet/NAL integrity...", flush=True)
        for video_path in videos:
            try:
                check_video_packet_integrity(video_path)
            except CliError as error:
                packet_errors.append((video_path, error))
            else:
                print(f"Packet check OK: {video_path.name}", flush=True)

        if packet_errors:
            message = format_video_errors(packet_errors)
            print(f"Video problem detected: {message}", flush=True)
            raise CliError(message)

    if not decode_check:
        return durations

    decode_errors: list[tuple[Path, Exception]] = []
    print("Checking full video decode...", flush=True)
    for video_path in videos:
        try:
            check_video_decodable(video_path)
        except CliError as error:
            decode_errors.append((video_path, error))
        else:
            print(f"Decode check OK: {video_path.name}", flush=True)

    if decode_errors:
        message = format_video_errors(decode_errors)
        print(f"Video problem detected: {message}", flush=True)
        raise CliError(message)

    return durations


def extract_selected_frames(
    video_path: Path,
    output_pattern: Path,
    frame_indices: list[int],
    quality: int,
) -> None:
    """Decode once from the beginning and extract exact source-frame indices."""
    select_expression = "+".join(f"eq(n\\,{index})" for index in frame_indices)
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"select={select_expression}",
            "-fps_mode",
            "passthrough",
            "-q:v",
            str(quality),
            "-start_number",
            "1",
            "-y",
            str(output_pattern),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = (
            f"Video decode failed for {video_path.name}. "
            "The source video may be corrupt or incomplete.\n"
            f"{command_error_message(result)}"
        )
        print(f"Video problem detected: {message}", flush=True)
        raise CliError(message)


def validate_extracted_count(paths: list[Path], expected_count: int) -> None:
    if len(paths) != expected_count:
        message = (
            f"ffmpeg produced {len(paths)} frame(s), expected {expected_count}. "
            "The video's frame-count metadata may be incorrect or the video may be truncated."
        )
        print(f"Video problem detected: {message}", flush=True)
        raise CliError(message)


def extract_video_frames(
    video_path: Path,
    input_dir: Path,
    output_root: Path,
    frame_count: int,
    quality: int,
    duration: float,
    total_frames: int | None = None,
) -> int:
    prefix = frame_prefix(video_path, input_dir)
    total_frames = total_frames if total_frames is not None else get_frame_count(video_path)
    frame_indices = build_uniform_frame_indices(total_frames, frame_count)

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{prefix}_", dir=output_root) as temp_dir:
        temp_root = Path(temp_dir)
        extract_selected_frames(
            video_path,
            temp_root / "frame_%05d.jpg",
            frame_indices,
            quality,
        )
        temp_paths = sorted(temp_root.glob("frame_*.jpg"))
        validate_extracted_count(temp_paths, frame_count)

        for old_path in output_root.glob(f"{prefix}_frame_*.jpg"):
            old_path.unlink()

        for output_index, (temp_path, source_index) in enumerate(
            zip(temp_paths, frame_indices, strict=True), start=1
        ):
            estimated_time = duration * source_index / total_frames
            frame_name = (
                f"{prefix}_frame_{output_index:05d}_{estimated_time:010.3f}s_"
                f"source_{source_index:09d}.jpg"
            )
            temp_path.replace(output_root / frame_name)

    return frame_count


def main() -> None:
    args = parse_args()
    require_command("ffmpeg")
    require_command("ffprobe")

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if args.frame_count <= 0:
        raise ValueError("--frame-count must be greater than 0.")

    output_dir_name = args.output_dir or input_dir.name
    output_root = PROJECT_ROOT / "data" / "output" / output_dir_name

    videos = find_videos(input_dir, args.recursive)
    if not videos:
        print(f"No videos found in {input_dir}")
        return

    history_csv = args.video_history_csv
    if not history_csv.is_absolute():
        history_csv = PROJECT_ROOT / history_csv
    print("Searching past clip records for previously used videos...", flush=True)
    history_rows = read_video_history(history_csv)
    past_clip_rows = search_past_clip_history(PROJECT_ROOT)
    searchable_history_rows, _ = merge_video_history(history_rows, past_clip_rows)
    fingerprints = fingerprint_videos(videos)

    known_names, known_contents = check_video_duplicates(
        videos,
        fingerprints,
        searchable_history_rows,
        allow_known_videos=args.allow_known_video_names,
    )
    if known_names or known_contents:
        print(
            "重複あり: --allow-known-videos が指定されているため再抽出を続行します。\n"
            f"{format_known_video_matches(known_names, known_contents)}",
            flush=True,
        )
    print(
        f"Duplicate search OK: {len(videos)} input video(s), "
        f"{len(searchable_history_rows)} past record(s), "
        f"{len(known_names)} allowed name match(es), "
        f"{len(known_contents)} allowed content match(es).",
        flush=True,
    )

    try:
        durations = validate_videos(
            videos,
            decode_check=args.full_decode_check,
            packet_check=True,
        )
    except CliError:
        print(
            "Frame extraction aborted before output: 0 new frames were written. "
            "Any existing output files were left unchanged.",
            flush=True,
        )
        raise

    frame_counts: dict[Path, int] = {}
    frame_count_errors: list[tuple[Path, Exception]] = []
    print("Checking video frame counts...", flush=True)
    for video_path in videos:
        try:
            total_frames = get_frame_count(video_path)
            build_uniform_frame_indices(total_frames, args.frame_count)
            frame_counts[video_path] = total_frames
        except CliError as error:
            frame_count_errors.append((video_path, error))

    if frame_count_errors:
        message = format_video_errors(frame_count_errors)
        print(f"Video problem detected: {message}", flush=True)
        print(
            "Frame extraction aborted before output: 0 new frames were written. "
            "Any existing output files were left unchanged.",
            flush=True,
        )
        raise CliError(message)

    print(f"All {len(videos)} video(s) passed validation. Starting staged extraction.", flush=True)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_prefix = f".{safe_name(output_root.name) or 'frames'}_staging_"
    recorded_at = datetime.now().astimezone().isoformat(timespec="seconds")
    new_history_rows = build_video_history_rows(
        videos=videos,
        input_dir=input_dir,
        output_root=output_root,
        recorded_at=recorded_at,
        fingerprints=fingerprints,
    )

    try:
        with tempfile.TemporaryDirectory(
            prefix=staging_prefix,
            dir=output_root.parent,
        ) as directory:
            staging_root = Path(directory)
            for video_path in videos:
                extracted_count = extract_video_frames(
                    video_path=video_path,
                    input_dir=input_dir,
                    output_root=staging_root,
                    frame_count=args.frame_count,
                    quality=args.quality,
                    duration=durations[video_path],
                    total_frames=frame_counts[video_path],
                )
                print(f"{video_path.name}: staged {extracted_count} frame(s)")

            staged_paths = sorted(staging_root.glob("*.jpg"))
            expected_count = len(videos) * args.frame_count
            if len(staged_paths) != expected_count:
                raise CliError(
                    f"Staging contains {len(staged_paths)} frame(s), expected {expected_count}. "
                    "No output was committed."
                )

            staged_video_names_csv = staging_root / "video_names.csv"
            write_video_history(staged_video_names_csv, new_history_rows)

            output_root.mkdir(parents=True, exist_ok=True)
            prefixes = {frame_prefix(video_path, input_dir) for video_path in videos}
            for prefix in prefixes:
                for old_path in output_root.glob(f"{prefix}_frame_*.jpg"):
                    old_path.unlink()
            old_manifest = output_root / "selection_manifest.jsonl"
            if old_manifest.exists():
                old_manifest.unlink()
            for staged_path in staged_paths:
                staged_path.replace(output_root / staged_path.name)
            staged_video_names_csv.replace(output_root / staged_video_names_csv.name)
    except CliError:
        print(
            "Staged extraction aborted: all temporary frames were discarded. "
            "The data/output result was left unchanged.",
            flush=True,
        )
        raise

    merged_history_rows, _ = merge_video_history(history_rows, past_clip_rows)
    merged_history_rows, added_history_rows = merge_video_history(
        merged_history_rows,
        new_history_rows,
    )
    write_video_history(history_csv, merged_history_rows)
    print(f"Committed {len(videos) * args.frame_count} frame(s) to: {output_root}")
    print(f"Video names CSV: {output_root / 'video_names.csv'}")
    print(f"Video history CSV: {history_csv} ({added_history_rows} new video record(s))")


if __name__ == "__main__":
    try:
        main()
    except (CliError, NotADirectoryError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from None
