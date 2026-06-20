import argparse
import hashlib
import random
import shutil
import subprocess
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a fixed number of frames from each video in a directory."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        # 対象動画を指定したい場合は、実行時にこの引数へ動画が入ったディレクトリを渡します。
        # 例: python .\script\script_data\clip_frame.py .\data\input\2026_06_17
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
        help="Number of segments/frames to extract from each video. Default: 100.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible frame selection.",
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
    return parser.parse_args()


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} is not available. Please install ffmpeg.")


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
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def extract_frame(video_path: Path, output_path: Path, time_seconds: float, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{time_seconds:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            str(quality),
            "-y",
            str(output_path),
        ],
        check=True,
    )


def extract_video_frames(
    video_path: Path,
    input_dir: Path,
    output_root: Path,
    frame_count: int,
    quality: int,
    rng: random.Random,
) -> int:
    duration = get_duration_seconds(video_path)
    prefix = frame_prefix(video_path, input_dir)

    for count in range(1, frame_count + 1):
        segment_start = duration * (count - 1) / frame_count
        segment_end = duration * count / frame_count
        time_seconds = min(rng.uniform(segment_start, segment_end), max(duration - 0.001, 0.0))
        frame_name = f"{prefix}_frame_{count:05d}_{time_seconds:010.3f}s.jpg"
        extract_frame(video_path, output_root / frame_name, time_seconds, quality)

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

    project_root = Path(__file__).resolve().parents[2]
    output_dir_name = args.output_dir or input_dir.name
    output_root = project_root / "data" / "output" / output_dir_name
    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    videos = find_videos(input_dir, args.recursive)
    if not videos:
        print(f"No videos found in {input_dir}")
        return

    print(f"Output: {output_root}")
    for video_path in videos:
        frame_count = extract_video_frames(
            video_path=video_path,
            input_dir=input_dir,
            output_root=output_root,
            frame_count=args.frame_count,
            quality=args.quality,
            rng=rng,
        )
        print(f"{video_path.name}: extracted {frame_count} frame(s)")


if __name__ == "__main__":
    main()
