import argparse
import csv
import hashlib
import json
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
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
VIDEO_SUFFIX_PATTERN = "|".join(extension.removeprefix(".") for extension in VIDEO_EXTENSIONS)
HISTORY_VIDEO_PATH_RE = re.compile(
    rf"(?:\./)?data/input/(?P<folder>[^/\\\r\n\"]+)/"
    rf"(?P<name>[^/\\\r\n\"]+?\.(?:{VIDEO_SUFFIX_PATTERN}))",
    re.IGNORECASE,
)
FRAME_PREFIX_RE = re.compile(r"^(?P<prefix>.+)_frame_\d{5}(?:_|\.)")
HASH_SUFFIX_RE = re.compile(r"_(?P<hash>[0-9a-f]{8})$")
HISTORY_TEXT_SUFFIXES = {".jsonl", ".log", ".txt", ".history"}


@dataclass
class VideoCandidate:
    video_name: str
    normalized_stem: str
    priority: int
    evidence_sources: set[str] = field(default_factory=set)
    source_paths: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)
    content_sha256: str = ""
    file_size_bytes: str = ""


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Recover video names from existing inputs, frame manifests, extracted-frame "
            "prefixes, and optional text history roots."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help=f"Project root to scan. Default: {project_root}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/video_name_history.csv"),
        help="CSV to create or refresh, relative to project root by default.",
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional text file or directory to scan for past data/input video paths. "
            "May be repeated; for Codex history, pass ~/.codex."
        ),
    )
    return parser.parse_args()


def normalize_video_stem(video_name: str) -> str:
    return unicodedata.normalize("NFKC", Path(video_name).stem).casefold().strip()


def video_stem_sha1(video_name: str) -> str:
    return hashlib.sha1(Path(video_name).stem.encode("utf-8")).hexdigest()


def is_plausible_video_name(video_name: str) -> bool:
    if any(character in video_name for character in "*?[]"):
        return False
    stem = Path(video_name).stem.strip()
    return bool(stem) and any(character != "." for character in stem) and stem != "対象ファイル"


def add_candidate(
    candidates: dict[str, VideoCandidate],
    video_name: str,
    evidence_source: str,
    source_path: str = "",
    priority: int = 0,
) -> None:
    video_name = Path(video_name.removesuffix(":Zone.Identifier")).name.strip()
    if not is_plausible_video_name(video_name):
        return
    normalized = normalize_video_stem(video_name)
    if not normalized:
        return

    candidate = candidates.get(normalized)
    if candidate is None:
        candidate = VideoCandidate(
            video_name=video_name,
            normalized_stem=normalized,
            priority=priority,
        )
        candidates[normalized] = candidate
    elif priority > candidate.priority or (
        priority == candidate.priority
        and not Path(candidate.video_name).suffix
        and Path(video_name).suffix
    ):
        candidate.video_name = video_name
        candidate.priority = priority

    candidate.aliases.add(video_name)
    candidate.evidence_sources.add(evidence_source)
    if source_path:
        candidate.source_paths.add(source_path)


def add_prefix(
    prefix_locations: dict[str, set[str]],
    prefix: str,
    output_dir: str = "",
) -> None:
    if HASH_SUFFIX_RE.search(prefix) is None:
        return
    locations = prefix_locations.setdefault(prefix, set())
    if output_dir:
        locations.add(output_dir)


def read_existing_csv(
    csv_path: Path,
    candidates: dict[str, VideoCandidate],
    prefix_locations: dict[str, set[str]],
) -> None:
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_name = row.get("video_name", "")
            if video_name:
                add_candidate(
                    candidates,
                    video_name,
                    "existing_csv",
                    priority=110,
                )
                candidate = candidates.get(normalize_video_stem(video_name))
                if candidate is not None:
                    candidate.source_paths.update(
                        path for path in row.get("source_path", "").split(";") if path
                    )
                    if not candidate.content_sha256:
                        candidate.content_sha256 = row.get("content_sha256", "")
                    if not candidate.file_size_bytes:
                        candidate.file_size_bytes = row.get("file_size_bytes", "")
            for prefix in row.get("video_prefix", "").split(";"):
                if prefix:
                    for output_dir in row.get("output_dir", "").split(";") or [""]:
                        add_prefix(prefix_locations, prefix, output_dir)


def scan_current_inputs(project_root: Path, candidates: dict[str, VideoCandidate]) -> None:
    input_root = project_root / "data" / "input"
    if not input_root.is_dir():
        return
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            add_candidate(
                candidates,
                path.name,
                "current_input",
                str(path.relative_to(project_root)),
                priority=100,
            )


def scan_video_name_csvs(
    project_root: Path,
    candidates: dict[str, VideoCandidate],
    prefix_locations: dict[str, set[str]],
) -> None:
    output_root = project_root / "data" / "output"
    if not output_root.is_dir():
        return
    for csv_path in sorted(output_root.rglob("video_names.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                video_name = row.get("video_name", "")
                if video_name:
                    add_candidate(
                        candidates,
                        video_name,
                        "video_names_csv",
                        row.get("source_path", ""),
                        priority=105,
                    )
                prefix = row.get("video_prefix", "")
                if prefix:
                    add_prefix(
                        prefix_locations,
                        prefix,
                        str(csv_path.parent.relative_to(project_root)),
                    )


def scan_selection_manifests(
    project_root: Path,
    candidates: dict[str, VideoCandidate],
    prefix_locations: dict[str, set[str]],
) -> None:
    output_root = project_root / "data" / "output"
    if not output_root.is_dir():
        return
    for manifest_path in sorted(output_root.rglob("selection_manifest.jsonl")):
        with manifest_path.open("r", encoding="utf-8", errors="replace") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                video_name = str(record.get("video_name", ""))
                source_path = str(record.get("video", ""))
                if video_name:
                    add_candidate(
                        candidates,
                        video_name,
                        "selection_manifest",
                        source_path,
                        priority=95,
                    )
                prefix = str(record.get("video_prefix", ""))
                if prefix:
                    add_prefix(
                        prefix_locations,
                        prefix,
                        str(manifest_path.parent.relative_to(project_root)),
                    )


def scan_legacy_frame_directories(
    project_root: Path,
    candidates: dict[str, VideoCandidate],
) -> None:
    legacy_root = project_root / "output"
    if not legacy_root.is_dir():
        return
    for directory in sorted(path for path in legacy_root.glob("*/*") if path.is_dir()):
        if any(directory.glob("*.jpg")):
            add_candidate(
                candidates,
                directory.name,
                "legacy_frame_directory",
                str(directory.relative_to(project_root)),
                priority=70,
            )


def scan_frame_prefixes(
    project_root: Path,
    prefix_locations: dict[str, set[str]],
) -> None:
    output_root = project_root / "data" / "output"
    if not output_root.is_dir():
        return
    for image_path in output_root.rglob("*.jpg"):
        match = FRAME_PREFIX_RE.match(image_path.name)
        if match:
            add_prefix(
                prefix_locations,
                match.group("prefix"),
                str(image_path.parent.relative_to(project_root)),
            )


def iter_history_files(history_root: Path):
    if history_root.is_file():
        yield history_root
        return
    if not history_root.is_dir():
        return
    for path in history_root.rglob("*"):
        if path.is_file() and (
            path.suffix.lower() in HISTORY_TEXT_SUFFIXES
            or path.name in {".bash_history", ".zsh_history"}
        ):
            yield path


def scan_text_histories(
    history_roots: list[Path],
    candidates: dict[str, VideoCandidate],
) -> None:
    for history_root in history_roots:
        for history_path in iter_history_files(history_root.expanduser()):
            try:
                content = history_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in HISTORY_VIDEO_PATH_RE.finditer(content):
                folder = match.group("folder")
                video_name = match.group("name")
                add_candidate(
                    candidates,
                    video_name,
                    "text_history",
                    f"data/input/{folder}/{video_name}",
                    priority=80,
                )


def matching_prefixes(
    candidate: VideoCandidate,
    prefix_locations: dict[str, set[str]],
) -> list[str]:
    short_hashes = {video_stem_sha1(alias)[:8] for alias in candidate.aliases}
    return sorted(
        prefix
        for prefix in prefix_locations
        if (match := HASH_SUFFIX_RE.search(prefix)) and match.group("hash") in short_hashes
    )


def build_rows(
    candidates: dict[str, VideoCandidate],
    prefix_locations: dict[str, set[str]],
    recorded_at: str,
) -> tuple[list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    matched_prefixes: set[str] = set()
    for candidate in sorted(candidates.values(), key=lambda item: item.normalized_stem):
        prefixes = matching_prefixes(candidate, prefix_locations)
        matched_prefixes.update(prefixes)
        output_dirs = sorted(
            {
                output_dir
                for prefix in prefixes
                for output_dir in prefix_locations.get(prefix, set())
                if output_dir
            }
        )
        rows.append(
            {
                "video_name": candidate.video_name,
                "normalized_stem": candidate.normalized_stem,
                "stem_sha1": video_stem_sha1(candidate.video_name),
                "video_prefix": ";".join(prefixes),
                "source_path": ";".join(sorted(candidate.source_paths)),
                "output_dir": ";".join(output_dirs),
                "recorded_at": recorded_at,
                "evidence_source": ";".join(sorted(candidate.evidence_sources)),
                "recovery_status": (
                    "exact" if Path(candidate.video_name).suffix else "exact_stem"
                ),
                "content_sha256": candidate.content_sha256,
                "file_size_bytes": candidate.file_size_bytes,
            }
        )

    unresolved_prefixes = sorted(set(prefix_locations) - matched_prefixes)
    for prefix in unresolved_prefixes:
        rows.append(
            {
                "video_name": "",
                "normalized_stem": "",
                "stem_sha1": "",
                "video_prefix": prefix,
                "source_path": "",
                "output_dir": ";".join(sorted(prefix_locations[prefix])),
                "recorded_at": recorded_at,
                "evidence_source": "extracted_frame",
                "recovery_status": "unresolved",
                "content_sha256": "",
                "file_size_bytes": "",
            }
        )
    return rows, len(unresolved_prefixes)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
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
            writer.writerows(rows)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path

    candidates: dict[str, VideoCandidate] = {}
    prefix_locations: dict[str, set[str]] = {}
    read_existing_csv(output_path, candidates, prefix_locations)
    scan_current_inputs(project_root, candidates)
    scan_video_name_csvs(project_root, candidates, prefix_locations)
    scan_selection_manifests(project_root, candidates, prefix_locations)
    scan_legacy_frame_directories(project_root, candidates)
    scan_frame_prefixes(project_root, prefix_locations)
    scan_text_histories(args.history_root, candidates)

    recorded_at = datetime.now().astimezone().isoformat(timespec="seconds")
    rows, unresolved_count = build_rows(candidates, prefix_locations, recorded_at)
    write_csv(output_path, rows)

    exact_count = sum(1 for row in rows if row["recovery_status"] != "unresolved")
    matched_count = len(prefix_locations) - unresolved_count
    print(f"Recovered exact video names: {exact_count}")
    print(f"Matched extracted-frame prefixes: {matched_count}")
    print(f"Unresolved extracted-frame prefixes: {unresolved_count}")
    print(f"Video history CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
