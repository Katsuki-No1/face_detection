from __future__ import annotations

import csv
import hashlib
import html
import json
import logging
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
import yaml

from script.script_data.clip_frame import VIDEO_EXTENSIONS
from src.active_learning.scoring import deterministic_kmeans

from .scoring import (
    build_window_starts,
    invalid_penalty,
    normal_band_points,
    presence_points,
    representative_points,
    representative_raw_score,
    robust_variation,
    stable_percentiles,
)

LOGGER = logging.getLogger(__name__)
KEYPOINT_NAMES = ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")


@dataclass(slots=True)
class NormalPerformanceConfig:
    project_root: Path
    config_path: Path
    input_dir: Path
    output_dir: Path
    weights: Path
    device: str = "0"
    imgsz: int = 960
    confidence: float = 0.44
    max_detections: int = 100
    batch_size: int = 8
    anchor_interval_sec: float = 60.0
    motion_offsets_sec: tuple[float, ...] = (0.0, 0.5, 1.0)
    window_duration_sec: float = 600.0
    window_step_sec: float = 60.0
    cluster_count: int = 6
    top_review_count: int = 5
    random_seed: int = 20260718
    scene_change_threshold: float = 0.32
    black_brightness_threshold: float = 8.0
    white_brightness_threshold: float = 247.0
    contrast_threshold: float = 5.0
    blur_threshold: float = 20.0
    frozen_change_threshold: float = 0.002
    keypoint_visibility_threshold: float = 0.50
    output_video_crf: int = 20
    source_clip_crf: int = 18
    output_video_preset: str = "veryfast"
    source_clip_preset: str = "medium"
    recursive: bool = False

    def validate(self) -> None:
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"input_dir does not exist: {self.input_dir}")
        if not self.weights.is_file():
            raise FileNotFoundError(f"weights do not exist: {self.weights}")
        if self.imgsz <= 0 or self.batch_size <= 0 or self.max_detections <= 0:
            raise ValueError("imgsz, batch_size, and max_detections must be positive")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if self.anchor_interval_sec <= 0 or self.window_duration_sec <= 0 or self.window_step_sec <= 0:
            raise ValueError("time intervals must be positive")
        if not self.motion_offsets_sec or self.motion_offsets_sec[0] != 0.0:
            raise ValueError("motion_offsets_sec must start with 0.0")
        if tuple(sorted(self.motion_offsets_sec)) != self.motion_offsets_sec:
            raise ValueError("motion_offsets_sec must be sorted")
        spacing = np.diff(np.asarray(self.motion_offsets_sec, dtype=np.float64))
        if len(spacing) and not np.allclose(spacing, spacing[0]):
            raise ValueError("motion_offsets_sec must use equal spacing")
        ratio = self.window_duration_sec / self.anchor_interval_sec
        if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("window_duration_sec must be divisible by anchor_interval_sec")

    def resolved_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for key, value in list(row.items()):
            if isinstance(value, Path):
                row[key] = value.resolve().as_posix()
        row["motion_offsets_sec"] = list(self.motion_offsets_sec)
        return row


@dataclass(slots=True)
class VideoMetadata:
    path: Path
    video_id: str
    duration_sec: float
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    file_size: int
    mtime_ns: int


@dataclass(slots=True)
class Detection:
    confidence: float
    bbox: tuple[float, float, float, float]
    keypoints: tuple[tuple[float, float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": "face",
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox),
            "keypoints": {
                name: [float(point[0]), float(point[1]), float(point[2])]
                for name, point in zip(KEYPOINT_NAMES, self.keypoints, strict=True)
            },
        }


@dataclass(slots=True)
class AnchorFeature:
    index: int
    timestamp_sec: float
    image_path: str
    brightness: float
    contrast: float
    blur_score: float
    max_scene_change: float
    face_count: int
    face_present: bool
    invalid: bool
    invalid_reasons: list[str]
    primary_pose_size: tuple[float, float, float, float] | None
    motion_raw: float | None
    detections: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["primary_pose_size"] = list(self.primary_pose_size) if self.primary_pose_size is not None else None
        return row


@dataclass(slots=True)
class WindowScore:
    rank: int
    start_sec: float
    end_sec: float
    anchor_indices: list[int]
    representative_raw: float
    representative_percentile: float
    representative_points: int
    presence_ratio: float
    presence_points: int
    pose_size_raw: float
    pose_size_percentile: float
    pose_size_points: int
    temporal_motion_raw: float
    temporal_motion_percentile: float
    temporal_motion_points: int
    invalid_anchor_count: int
    invalid_penalty: int
    total_score: int
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> NormalPerformanceConfig:
    path = path.expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError("config root must be a mapping")
    project_root = _resolve_path(raw.get("project_root", ".."), path.parent)
    fields = dict(raw)
    fields.pop("project_root", None)
    fields["project_root"] = project_root
    fields["config_path"] = path
    for name in ("input_dir", "output_dir", "weights"):
        fields[name] = _resolve_path(fields[name], project_root)
    fields["motion_offsets_sec"] = tuple(float(value) for value in fields.get("motion_offsets_sec", (0.0, 0.5, 1.0)))
    config = NormalPerformanceConfig(**fields)
    config.validate()
    return config


def _rational(value: str | None) -> float:
    if not value or value in {"N/A", "0/0"}:
        return 0.0
    numerator, separator, denominator = value.partition("/")
    return float(numerator) / float(denominator) if separator else float(value)


def probe_video(path: Path) -> VideoMetadata:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
            "-of", "json", path.as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path.name}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    fps = _rational(stream.get("r_frame_rate") or stream.get("avg_frame_rate"))
    width, height = int(stream["width"]), int(stream["height"])
    if duration <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video metadata for {path.name}")
    stat = path.stat()
    frame_count_text = str(stream.get("nb_frames", ""))
    frame_count = int(frame_count_text) if frame_count_text.isdigit() else int(round(duration * fps))
    video_id = path.stem.split()[0].strip().replace("_", "-") or hashlib.sha1(path.name.encode()).hexdigest()[:12]
    return VideoMetadata(
        path=path.resolve(),
        video_id=video_id,
        duration_sec=duration,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        codec=str(stream.get("codec_name", "")),
        file_size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def find_input_videos(config: NormalPerformanceConfig) -> list[Path]:
    pattern = "**/*" if config.recursive else "*"
    return sorted(
        path
        for path in config.input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _parse_detections(result: Any) -> list[Detection]:
    boxes = result.boxes
    if boxes is None:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidence = boxes.conf.detach().cpu().numpy()
    keypoint_data = (
        result.keypoints.data.detach().cpu().numpy()
        if result.keypoints is not None
        else np.zeros((len(xyxy), len(KEYPOINT_NAMES), 3), dtype=np.float32)
    )
    detections: list[Detection] = []
    for index, box in enumerate(xyxy):
        points = keypoint_data[index] if index < len(keypoint_data) else np.zeros((len(KEYPOINT_NAMES), 3))
        if points.shape[-1] == 2:
            points = np.concatenate([points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)
        detections.append(Detection(
            confidence=float(confidence[index]),
            bbox=tuple(float(value) for value in box),
            keypoints=tuple(tuple(float(value) for value in point[:3]) for point in points[: len(KEYPOINT_NAMES)]),
        ))
    return detections


def _primary_detection(detections: Sequence[Detection]) -> Detection | None:
    if not detections:
        return None
    return max(
        detections,
        key=lambda row: ((row.bbox[2] - row.bbox[0]) * (row.bbox[3] - row.bbox[1]), row.confidence),
    )


def _pose_size_feature(detection: Detection | None, width: int, height: int, visibility: float) -> tuple[float, float, float, float] | None:
    if detection is None:
        return None
    x1, y1, x2, y2 = detection.bbox
    area_ratio = max(1e-9, (x2 - x1) * (y2 - y1) / float(width * height))
    center_x = (x1 + x2) / 2.0 / width
    left_eye, right_eye, nose = detection.keypoints[:3]
    if min(left_eye[2], right_eye[2], nose[2]) < visibility:
        return (math.log(area_ratio), center_x, 0.0, 0.0)
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    eye_distance = max(1e-6, math.hypot(eye_dx, eye_dy))
    roll = math.atan2(eye_dy, eye_dx) / math.pi
    eye_midpoint_x = (left_eye[0] + right_eye[0]) / 2.0
    yaw = (nose[0] - eye_midpoint_x) / eye_distance
    return (math.log(area_ratio), center_x, roll, yaw)


def _frame_quality(frame: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return (
        float(np.mean(gray)),
        float(np.std(gray)),
        float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )


def _scene_change(left: np.ndarray, right: np.ndarray) -> float:
    left_gray = cv2.resize(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
    right_gray = cv2.resize(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(left_gray, right_gray)) / 255.0)


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _match_detection(previous: Detection | None, candidates: Sequence[Detection], width: int, height: int) -> Detection | None:
    if not candidates:
        return None
    if previous is None:
        return _primary_detection(candidates)
    by_iou = max(candidates, key=lambda row: _bbox_iou(previous.bbox, row.bbox))
    if _bbox_iou(previous.bbox, by_iou.bbox) >= 0.10:
        return by_iou
    previous_center = np.asarray([(previous.bbox[0] + previous.bbox[2]) / 2, (previous.bbox[1] + previous.bbox[3]) / 2])
    diagonal = max(1.0, math.hypot(width, height))
    by_distance = min(
        candidates,
        key=lambda row: float(np.linalg.norm(
            previous_center - np.asarray([(row.bbox[0] + row.bbox[2]) / 2, (row.bbox[1] + row.bbox[3]) / 2])
        )),
    )
    center = np.asarray([(by_distance.bbox[0] + by_distance.bbox[2]) / 2, (by_distance.bbox[1] + by_distance.bbox[3]) / 2])
    return by_distance if float(np.linalg.norm(previous_center - center)) / diagonal <= 0.25 else None


def _model_motion(left: Detection, right: Detection, visibility: float) -> float:
    x1, y1, x2, y2 = left.bbox
    face_diagonal = max(1.0, math.hypot(x2 - x1, y2 - y1))
    left_center = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2])
    right_center = np.asarray([(right.bbox[0] + right.bbox[2]) / 2, (right.bbox[1] + right.bbox[3]) / 2])
    values = [float(np.linalg.norm(right_center - left_center)) / face_diagonal]
    for left_point, right_point in zip(left.keypoints, right.keypoints, strict=True):
        if left_point[2] >= visibility and right_point[2] >= visibility:
            values.append(math.hypot(right_point[0] - left_point[0], right_point[1] - left_point[1]) / face_diagonal)
    return float(np.median(values))


def _flow_motion(left_frame: np.ndarray, right_frame: np.ndarray, detection: Detection) -> float:
    target_width = 320
    scale = target_width / left_frame.shape[1]
    target_height = max(1, int(round(left_frame.shape[0] * scale)))
    left_gray = cv2.resize(cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY), (target_width, target_height), interpolation=cv2.INTER_AREA)
    right_gray = cv2.resize(cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY), (target_width, target_height), interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(left_gray, right_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    x1, y1, x2, y2 = detection.bbox
    sx1 = int(np.clip(math.floor(x1 * scale), 0, target_width - 1))
    sy1 = int(np.clip(math.floor(y1 * scale), 0, target_height - 1))
    sx2 = int(np.clip(math.ceil(x2 * scale), sx1 + 1, target_width))
    sy2 = int(np.clip(math.ceil(y2 * scale), sy1 + 1, target_height))
    magnitude = np.linalg.norm(flow[sy1:sy2, sx1:sx2], axis=2)
    face_diagonal = max(1.0, math.hypot(sx2 - sx1, sy2 - sy1))
    return float(np.median(magnitude) / face_diagonal) if magnitude.size else 0.0


def _burst_motion(
    frames: Sequence[np.ndarray],
    detections: Sequence[Sequence[Detection]],
    config: NormalPerformanceConfig,
) -> tuple[float | None, float]:
    primary = _primary_detection(detections[0])
    previous = primary
    values: list[float] = []
    scene_scores: list[float] = []
    for index in range(len(frames) - 1):
        scene = _scene_change(frames[index], frames[index + 1])
        scene_scores.append(scene)
        current = previous if index == 0 else _match_detection(previous, detections[index], frames[index].shape[1], frames[index].shape[0])
        following = _match_detection(current, detections[index + 1], frames[index].shape[1], frames[index].shape[0])
        if scene < config.scene_change_threshold and current is not None and following is not None:
            flow = _flow_motion(frames[index], frames[index + 1], current)
            model = _model_motion(current, following, config.keypoint_visibility_threshold)
            values.append(0.50 * flow + 0.50 * model)
        previous = following
    return (float(np.median(values)) if values else None, max(scene_scores, default=0.0))


def iter_motion_bursts(video: VideoMetadata, config: NormalPerformanceConfig) -> Iterator[tuple[float, list[np.ndarray]]]:
    offsets = config.motion_offsets_sec
    if len(offsets) < 2:
        sampling_fps = 1.0
    else:
        sampling_fps = 1.0 / (offsets[1] - offsets[0])
    period_frames = int(round(config.anchor_interval_sec * sampling_fps))
    selected_frames = len(offsets)
    video_frame_bytes = video.width * video.height * 3
    select_filter = f"fps={sampling_fps:.12g}:start_time=0,select='lt(mod(n\\,{period_frames})\\,{selected_frames})'"
    command = [
        "ffmpeg", "-v", "error", "-i", video.path.as_posix(), "-map", "0:v:0",
        "-vf", select_filter, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    group: list[np.ndarray] = []
    anchor_index = 0
    while True:
        raw = process.stdout.read(video_frame_bytes)
        if not raw:
            break
        if len(raw) != video_frame_bytes:
            process.kill()
            raise RuntimeError(f"short raw frame for {video.path.name}: {len(raw)}/{video_frame_bytes}")
        group.append(np.frombuffer(raw, dtype=np.uint8).reshape(video.height, video.width, 3).copy())
        if len(group) == selected_frames:
            timestamp = anchor_index * config.anchor_interval_sec
            if timestamp + offsets[-1] < video.duration_sec + 1e-6:
                yield timestamp, group
            group = []
            anchor_index += 1
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg sparse decode failed for {video.path.name}: {stderr[:4000]}")


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def scan_anchors(
    video: VideoMetadata,
    config: NormalPerformanceConfig,
    model: Any,
    embedding_model: Any,
    video_dir: Path,
) -> tuple[list[AnchorFeature], np.ndarray]:
    selection_dir = video_dir / "selection"
    image_dir = selection_dir / "anchor_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    anchors: list[AnchorFeature] = []
    embedding_rows: list[np.ndarray] = []
    pending: list[tuple[float, list[np.ndarray]]] = []
    started = time.perf_counter()

    def flush() -> None:
        if not pending:
            return
        flat_frames = [frame for _, burst in pending for frame in burst]
        results = model.predict(
            source=flat_frames,
            imgsz=config.imgsz,
            conf=config.confidence,
            max_det=config.max_detections,
            device=config.device,
            verbose=False,
        )
        parsed = [_parse_detections(result) for result in results]
        anchor_frames = [burst[0] for _, burst in pending]
        # Ultralytics reuses predictor overrides between calls. Keep embedding
        # extraction on a dedicated instance so it cannot switch later pose
        # predictions into tensor-only embed mode.
        embeddings = embedding_model.embed(source=anchor_frames, imgsz=config.imgsz, device=config.device, verbose=False)
        if len(embeddings) != len(pending):
            raise RuntimeError(f"YOLO embedding count mismatch: {len(embeddings)} != {len(pending)}")
        stride = len(config.motion_offsets_sec)
        for local_index, ((timestamp, burst), embedding) in enumerate(zip(pending, embeddings, strict=True)):
            prediction_groups = parsed[local_index * stride : (local_index + 1) * stride]
            anchor_detections = prediction_groups[0]
            motion_raw, max_scene = _burst_motion(burst, prediction_groups, config)
            brightness, contrast, blur = _frame_quality(burst[0])
            reasons: list[str] = []
            if brightness < config.black_brightness_threshold:
                reasons.append("black")
            if brightness > config.white_brightness_threshold:
                reasons.append("white")
            if contrast < config.contrast_threshold:
                reasons.append("low_contrast")
            if blur < config.blur_threshold:
                reasons.append("severe_blur")
            if not anchor_detections and max_scene < config.frozen_change_threshold:
                reasons.append("no_face_frozen")
            anchor_index = len(anchors)
            image_path = image_dir / f"anchor_{anchor_index:04d}_{int(round(timestamp)):06d}s.jpg"
            if not cv2.imwrite(image_path.as_posix(), burst[0], [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"failed to write anchor image: {image_path}")
            primary = _primary_detection(anchor_detections)
            anchors.append(AnchorFeature(
                index=anchor_index,
                timestamp_sec=timestamp,
                image_path=image_path.relative_to(video_dir).as_posix(),
                brightness=brightness,
                contrast=contrast,
                blur_score=blur,
                max_scene_change=max_scene,
                face_count=len(anchor_detections),
                face_present=bool(anchor_detections),
                invalid=bool(reasons),
                invalid_reasons=reasons,
                primary_pose_size=_pose_size_feature(primary, video.width, video.height, config.keypoint_visibility_threshold),
                motion_raw=motion_raw,
                detections=[row.to_dict() for row in anchor_detections],
            ))
            embedding_rows.append(_tensor_to_numpy(embedding).reshape(-1))
        LOGGER.info("[%s] scanned %d anchors", video.video_id, len(anchors))
        pending.clear()

    for timestamp, burst in iter_motion_bursts(video, config):
        pending.append((timestamp, burst))
        if len(pending) >= config.batch_size:
            flush()
    flush()
    if not anchors:
        raise RuntimeError(f"no complete anchor bursts decoded from {video.path.name}")
    embeddings_array = np.stack(embedding_rows).astype(np.float32)
    if embeddings_array.shape[1] != 512:
        raise RuntimeError(f"expected 512-dimensional YOLO embedding, got {embeddings_array.shape}")
    _write_jsonl(selection_dir / "anchors.jsonl", [row.to_dict() for row in anchors])
    np.save(selection_dir / "anchor_embeddings.npy", embeddings_array)
    LOGGER.info("[%s] anchor scan completed in %.1fs", video.video_id, time.perf_counter() - started)
    return anchors, embeddings_array


def _window_anchor_indices(
    anchors: Sequence[AnchorFeature],
    start_sec: float,
    config: NormalPerformanceConfig,
) -> list[int]:
    end_sec = start_sec + config.window_duration_sec
    return [
        row.index
        for row in anchors
        if row.timestamp_sec + 1e-6 >= start_sec and row.timestamp_sec < end_sec - 1e-6
    ]


def score_windows(
    video: VideoMetadata,
    config: NormalPerformanceConfig,
    anchors: Sequence[AnchorFeature],
    embeddings: np.ndarray,
) -> list[WindowScore]:
    labels = deterministic_kmeans(embeddings, config.cluster_count, config.random_seed)
    required_anchors = int(round(config.window_duration_sec / config.anchor_interval_sec))
    starts = build_window_starts(video.duration_sec, config.window_duration_sec, config.window_step_sec)
    index_groups: list[list[int]] = []
    retained_starts: list[float] = []
    for start in starts:
        indices = _window_anchor_indices(anchors, start, config)
        if len(indices) == required_anchors:
            retained_starts.append(start)
            index_groups.append(indices)
    if not index_groups:
        raise RuntimeError(
            f"{video.path.name} has no complete {config.window_duration_sec:.0f}s window with "
            f"{required_anchors} anchors"
        )

    representative_raw = [
        representative_raw_score(embeddings, labels, indices, config.cluster_count)
        for indices in index_groups
    ]
    all_pose_rows = np.asarray(
        [row.primary_pose_size for row in anchors if row.primary_pose_size is not None],
        dtype=np.float64,
    )
    pose_raw: list[float] = []
    motion_raw: list[float] = []
    presence_ratios: list[float] = []
    invalid_counts: list[int] = []
    for indices in index_groups:
        rows = [anchors[index] for index in indices]
        window_pose = np.asarray(
            [row.primary_pose_size for row in rows if row.primary_pose_size is not None],
            dtype=np.float64,
        )
        pose_raw.append(
            robust_variation(window_pose, all_pose_rows)
            if window_pose.ndim == 2 and all_pose_rows.ndim == 2
            else 0.0
        )
        motions = [float(row.motion_raw) for row in rows if row.motion_raw is not None and math.isfinite(row.motion_raw)]
        motion_raw.append(float(np.median(motions)) if len(motions) >= 3 else 0.0)
        presence_ratios.append(sum(row.face_present for row in rows) / len(rows))
        invalid_counts.append(sum(row.invalid for row in rows))

    representative_percentiles = stable_percentiles(representative_raw)
    pose_percentiles = stable_percentiles(pose_raw)
    motion_percentiles = stable_percentiles(motion_raw)
    scores: list[WindowScore] = []
    for index, (start, anchor_indices) in enumerate(zip(retained_starts, index_groups, strict=True)):
        rep_points = representative_points(float(representative_percentiles[index]))
        person_points = presence_points(presence_ratios[index])
        pose_points = normal_band_points(float(pose_percentiles[index]))
        motion_points = normal_band_points(float(motion_percentiles[index]))
        penalty = invalid_penalty(invalid_counts[index])
        scores.append(WindowScore(
            rank=0,
            start_sec=start,
            end_sec=start + config.window_duration_sec,
            anchor_indices=anchor_indices,
            representative_raw=representative_raw[index],
            representative_percentile=float(representative_percentiles[index]),
            representative_points=rep_points,
            presence_ratio=presence_ratios[index],
            presence_points=person_points,
            pose_size_raw=pose_raw[index],
            pose_size_percentile=float(pose_percentiles[index]),
            pose_size_points=pose_points,
            temporal_motion_raw=motion_raw[index],
            temporal_motion_percentile=float(motion_percentiles[index]),
            temporal_motion_points=motion_points,
            invalid_anchor_count=invalid_counts[index],
            invalid_penalty=penalty,
            total_score=rep_points + person_points + pose_points + motion_points + penalty,
        ))
    scores.sort(key=lambda row: (
        -row.total_score,
        -row.representative_raw,
        -row.presence_ratio,
        row.invalid_anchor_count,
        row.start_sec,
    ))
    for rank, row in enumerate(scores, 1):
        row.rank = rank
        row.selected = rank == 1
    return scores


def _draw_detection_overlay(frame: np.ndarray, detections: Sequence[dict[str, Any]]) -> np.ndarray:
    output = frame.copy()
    for row in detections:
        x1, y1, x2, y2 = (int(round(float(value))) for value in row["bbox_xyxy"])
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
        label = f"face {float(row['confidence']):.2f}"
        cv2.putText(output, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        for point in row.get("keypoints", {}).values():
            x, y, confidence = float(point[0]), float(point[1]), float(point[2])
            if confidence >= 0.50:
                cv2.circle(output, (int(round(x)), int(round(y))), 3, (0, 200, 255), -1, cv2.LINE_AA)
    return output


def render_contact_sheet(
    video_dir: Path,
    anchors: Sequence[AnchorFeature],
    score: WindowScore,
    destination: Path,
) -> None:
    cell_width, image_height, header_height = 384, 216, 34
    columns, rows_count = 5, 2
    canvas = np.full((rows_count * (image_height + header_height), columns * cell_width, 3), 24, dtype=np.uint8)
    for position, anchor_index in enumerate(score.anchor_indices):
        anchor = anchors[anchor_index]
        frame = cv2.imread((video_dir / anchor.image_path).as_posix())
        if frame is None:
            raise RuntimeError(f"cannot read anchor image: {anchor.image_path}")
        frame = _draw_detection_overlay(frame, anchor.detections)
        frame = cv2.resize(frame, (cell_width, image_height), interpolation=cv2.INTER_AREA)
        cell = np.full((image_height + header_height, cell_width, 3), 24, dtype=np.uint8)
        cell[header_height:] = frame
        minute = int(round(anchor.timestamp_sec))
        text = f"t={minute // 60:03d}:{minute % 60:02d} face={anchor.face_count} invalid={int(anchor.invalid)}"
        cv2.putText(cell, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        row_index, column_index = divmod(position, columns)
        y1, x1 = row_index * cell.shape[0], column_index * cell_width
        canvas[y1 : y1 + cell.shape[0], x1 : x1 + cell_width] = cell
    title = (
        f"rank={score.rank} total={score.total_score} start={score.start_sec / 60:.1f}m "
        f"rep={score.representative_points} person={score.presence_points} "
        f"pose={score.pose_size_points} motion={score.temporal_motion_points} invalid={score.invalid_penalty}"
    )
    title_bar = np.full((48, canvas.shape[1], 3), 12, dtype=np.uint8)
    cv2.putText(title_bar, title, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(destination.as_posix(), np.vstack([title_bar, canvas]), [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"failed to write contact sheet: {destination}")


def write_selection_outputs(
    video: VideoMetadata,
    config: NormalPerformanceConfig,
    video_dir: Path,
    anchors: Sequence[AnchorFeature],
    scores: Sequence[WindowScore],
) -> None:
    selection_dir = video_dir / "selection"
    json_rows = [score.to_dict() for score in scores]
    csv_rows = []
    for row in json_rows:
        csv_row = dict(row)
        csv_row["anchor_indices"] = ",".join(str(value) for value in row["anchor_indices"])
        csv_rows.append(csv_row)
    _write_csv(selection_dir / "window_scores.csv", csv_rows)
    _write_jsonl(selection_dir / "window_scores.jsonl", json_rows)
    top = list(scores[: config.top_review_count])
    review_dir = selection_dir / "top_review"
    review_rows: list[tuple[WindowScore, Path]] = []
    for score in top:
        path = review_dir / f"rank_{score.rank:02d}_{int(round(score.start_sec)):06d}s.jpg"
        render_contact_sheet(video_dir, anchors, score, path)
        review_rows.append((score, path))
    selected = scores[0]
    html_rows = []
    for score, path in review_rows:
        html_rows.append(
            "<section>"
            f"<h2>Rank {score.rank}{' — selected' if score.selected else ''}</h2>"
            f"<p>{score.start_sec / 60:.1f}–{score.end_sec / 60:.1f} min; total={score.total_score}; "
            f"representative={score.representative_points}, person={score.presence_points}, "
            f"pose/size={score.pose_size_points}, motion={score.temporal_motion_points}, "
            f"invalid={score.invalid_penalty}</p>"
            f"<img src=\"{html.escape(path.relative_to(selection_dir).as_posix())}\" alt=\"rank {score.rank}\">"
            "</section>"
        )
    report = (
        "<!doctype html><meta charset=\"utf-8\"><title>Normal-performance 10m selection</title>"
        "<style>body{font-family:system-ui;margin:28px;max-width:1600px}img{width:100%;height:auto;border:1px solid #bbb}"
        "section{margin:24px 0 42px}code{background:#eee;padding:2px 4px}</style>"
        f"<h1>{html.escape(video.path.name)}</h1>"
        "<p>This is a normal-operation smoke-test selection, not a labeled accuracy evaluation.</p>"
        f"<p>Selected: <strong>{selected.start_sec / 60:.1f}–{selected.end_sec / 60:.1f} min</strong>.</p>"
        + "".join(html_rows)
    )
    report_path = selection_dir / "selection_report.html"
    temporary = report_path.with_suffix(".html.partial")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(report_path)
    _write_json(selection_dir / "selected_window.json", scores[0].to_dict())


def _run_external(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(list(command), check=False, stdout=subprocess.DEVNULL, stderr=log, text=True)
    if result.returncode != 0:
        detail = log_path.read_text(encoding="utf-8", errors="replace")[:4000]
        raise RuntimeError(f"external command failed ({result.returncode}): {' '.join(command[:10])}\n{detail}")


def _probe_output(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration,size,bit_rate",
            "-of", "json", path.as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def validate_video(path: Path, expected_duration: float, *, expected_frames: int | None = None) -> dict[str, Any]:
    data = _probe_output(path)
    stream = data["streams"][0]
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    if abs(duration - expected_duration) > 0.15:
        raise RuntimeError(f"duration mismatch for {path.name}: {duration:.6f} != {expected_duration:.6f}")
    if expected_frames is not None:
        frames_text = str(stream.get("nb_frames", ""))
        if not frames_text.isdigit() or int(frames_text) != expected_frames:
            raise RuntimeError(f"frame count mismatch for {path.name}: {frames_text} != {expected_frames}")
    decode_log = path.with_suffix(path.suffix + ".decode.log")
    _run_external(
        ["ffmpeg", "-v", "error", "-xerror", "-i", path.as_posix(), "-map", "0:v:0", "-f", "null", "-"],
        decode_log,
    )
    decode_log.unlink(missing_ok=True)
    return data


def create_source_clip(
    video: VideoMetadata,
    score: WindowScore,
    config: NormalPerformanceConfig,
    video_dir: Path,
) -> Path:
    output = video_dir / "selected_source_10m.mp4"
    temporary = video_dir / "selected_source_10m.partial.mp4"
    temporary.unlink(missing_ok=True)
    _run_external(
        [
            "ffmpeg", "-y", "-v", "error", "-ss", f"{score.start_sec:.6f}", "-i", video.path.as_posix(),
            "-t", f"{config.window_duration_sec:.6f}", "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-crf", str(config.source_clip_crf), "-preset", config.source_clip_preset,
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            temporary.as_posix(),
        ],
        video_dir / "logs" / "source_clip_ffmpeg.log",
    )
    validate_video(temporary, config.window_duration_sec)
    temporary.replace(output)
    return output


def _greedy_match_count(previous: Sequence[Sequence[float]], current: Sequence[Sequence[float]], threshold: float = 0.30) -> int:
    pairs = sorted(
        [(_bbox_iou(left, right), left_index, right_index) for left_index, left in enumerate(previous) for right_index, right in enumerate(current)],
        reverse=True,
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    count = 0
    for overlap, left_index, right_index in pairs:
        if overlap < threshold:
            break
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        count += 1
    return count


def _percentile_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def run_full_inference(
    video: VideoMetadata,
    score: WindowScore,
    config: NormalPerformanceConfig,
    model: Any,
    video_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    inference_dir = video_dir / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = video_dir / "logs"
    annotated_output = inference_dir / "yolo11s_pose_annotated_10m.mp4"
    prediction_output = inference_dir / "predictions.jsonl"
    silent_temporary = inference_dir / "annotated_silent.partial.mp4"
    annotated_temporary = inference_dir / "yolo11s_pose_annotated_10m.partial.mp4"
    prediction_temporary = inference_dir / "predictions.jsonl.partial"
    for path in (silent_temporary, annotated_temporary, prediction_temporary):
        path.unlink(missing_ok=True)

    output_fps = video.fps
    expected_frames = int(round(config.window_duration_sec * output_fps))
    frame_bytes = video.width * video.height * 3
    decoder = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-ss", f"{score.start_sec:.6f}", "-i", video.path.as_posix(),
            "-t", f"{config.window_duration_sec:.6f}", "-map", "0:v:0", "-vf", f"fps={output_fps:.12g}:start_time=0",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    encoder_log = (logs_dir / "annotated_encoder.log").open("w", encoding="utf-8")
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{video.width}x{video.height}", "-r", f"{output_fps:.12g}", "-i", "-", "-an",
            "-c:v", "libx264", "-crf", str(config.output_video_crf), "-preset", config.output_video_preset,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", silent_temporary.as_posix(),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=encoder_log,
        text=False,
    )
    assert decoder.stdout is not None and encoder.stdin is not None
    frames_processed = 0
    frames_with_face = 0
    total_detections = 0
    face_counts: list[float] = []
    confidences: list[float] = []
    area_ratios: list[float] = []
    visible_keypoints = 0
    total_keypoints = 0
    continuity_matches = 0
    continuity_opportunities = 0
    detection_drop_events = 0
    detection_reappearance_events = 0
    previous_boxes: list[tuple[float, float, float, float]] = []
    batch: list[np.ndarray] = []
    started = time.perf_counter()

    def flush(prediction_file: Any) -> None:
        nonlocal frames_processed, frames_with_face, total_detections
        nonlocal visible_keypoints, total_keypoints, continuity_matches, continuity_opportunities
        nonlocal detection_drop_events, detection_reappearance_events, previous_boxes
        if not batch:
            return
        results = model.predict(
            source=batch,
            imgsz=config.imgsz,
            conf=config.confidence,
            max_det=config.max_detections,
            device=config.device,
            verbose=False,
        )
        for frame, result in zip(batch, results, strict=True):
            detections = _parse_detections(result)
            frame_index = frames_processed
            clip_time = frame_index / output_fps
            source_time = score.start_sec + clip_time
            prediction_file.write(json.dumps({
                "source": video.path.as_posix(),
                "frame_index": frame_index,
                "clip_time_sec": clip_time,
                "source_time_sec": source_time,
                "objects": [row.to_dict() for row in detections],
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
            boxes = [row.bbox for row in detections]
            if boxes:
                frames_with_face += 1
            face_counts.append(float(len(boxes)))
            total_detections += len(boxes)
            for detection in detections:
                confidences.append(detection.confidence)
                x1, y1, x2, y2 = detection.bbox
                area_ratios.append(max(0.0, (x2 - x1) * (y2 - y1) / float(video.width * video.height)))
                for point in detection.keypoints:
                    total_keypoints += 1
                    visible_keypoints += int(point[2] >= config.keypoint_visibility_threshold)
            if frames_processed > 0:
                continuity_matches += _greedy_match_count(previous_boxes, boxes)
                continuity_opportunities += max(len(previous_boxes), len(boxes))
                detection_drop_events += int(bool(previous_boxes) and not boxes)
                detection_reappearance_events += int(not previous_boxes and bool(boxes))
            previous_boxes = boxes
            annotated = result.plot()
            encoder.stdin.write(np.ascontiguousarray(annotated).tobytes())
            frames_processed += 1
            if frames_processed % max(1, int(round(output_fps * 10))) == 0:
                elapsed = time.perf_counter() - started
                LOGGER.info(
                    "[%s] full inference %.1f/%.1fs frames=%d fps=%.2f",
                    video.video_id,
                    frames_processed / output_fps,
                    config.window_duration_sec,
                    frames_processed,
                    frames_processed / max(elapsed, 1e-9),
                )
        batch.clear()

    try:
        with prediction_temporary.open("w", encoding="utf-8") as prediction_file:
            while frames_processed + len(batch) < expected_frames:
                raw = decoder.stdout.read(frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise RuntimeError(f"short inference frame: {len(raw)}/{frame_bytes}")
                batch.append(np.frombuffer(raw, dtype=np.uint8).reshape(video.height, video.width, 3).copy())
                if len(batch) >= config.batch_size:
                    flush(prediction_file)
            flush(prediction_file)
    except BaseException:
        decoder.kill()
        encoder.stdin.close()
        encoder.kill()
        encoder.wait()
        encoder_log.close()
        decoder.wait()
        raise
    encoder.stdin.close()
    encoder_return = encoder.wait()
    encoder_log.close()
    decoder_error = decoder.stderr.read().decode("utf-8", errors="replace") if decoder.stderr else ""
    decoder_return = decoder.wait()
    if decoder_return != 0:
        raise RuntimeError(f"inference decoder failed: {decoder_error[:4000]}")
    if encoder_return != 0:
        detail = (logs_dir / "annotated_encoder.log").read_text(encoding="utf-8", errors="replace")[:4000]
        raise RuntimeError(f"annotated encoder failed: {detail}")
    if frames_processed != expected_frames:
        raise RuntimeError(f"full inference frame mismatch: {frames_processed} != {expected_frames}")

    _run_external(
        [
            "ffmpeg", "-y", "-v", "error", "-ss", f"{score.start_sec:.6f}", "-t", f"{config.window_duration_sec:.6f}",
            "-i", video.path.as_posix(), "-i", silent_temporary.as_posix(), "-map", "1:v:0", "-map", "0:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart",
            annotated_temporary.as_posix(),
        ],
        logs_dir / "audio_mux_ffmpeg.log",
    )
    validate_video(annotated_temporary, config.window_duration_sec, expected_frames=expected_frames)
    silent_temporary.unlink(missing_ok=True)
    annotated_temporary.replace(annotated_output)
    prediction_temporary.replace(prediction_output)
    elapsed = time.perf_counter() - started
    summary = {
        "purpose": "normal-operation qualitative/time-series smoke test; not labeled accuracy evaluation",
        "video_id": video.video_id,
        "selected_start_sec": score.start_sec,
        "selected_end_sec": score.end_sec,
        "fps": output_fps,
        "frames_processed": frames_processed,
        "elapsed_sec": elapsed,
        "effective_inference_fps": frames_processed / max(elapsed, 1e-9),
        "frames_with_face": frames_with_face,
        "face_detection_coverage": frames_with_face / frames_processed,
        "total_detections": total_detections,
        "face_count": _percentile_summary(face_counts),
        "confidence": _percentile_summary(confidences),
        "face_area_ratio": _percentile_summary(area_ratios),
        "keypoint_visible_ratio": visible_keypoints / total_keypoints if total_keypoints else None,
        "continuity_iou_match_rate": continuity_matches / continuity_opportunities if continuity_opportunities else None,
        "detection_drop_events": detection_drop_events,
        "detection_reappearance_events": detection_reappearance_events,
    }
    _write_json(inference_dir / "summary.json", summary)
    lines = [
        "# YOLO11s Pose normal-operation 10-minute inference",
        "",
        "> This is a qualitative/time-series smoke test without ground-truth labels; it is not an accuracy evaluation.",
        "",
        f"- selected interval: `{score.start_sec:.3f}`–`{score.end_sec:.3f}` seconds",
        f"- frames: `{frames_processed}` at `{output_fps:.6g}` fps",
        f"- face detection coverage: `{summary['face_detection_coverage']:.4f}`",
        f"- total detections: `{total_detections}`",
        f"- median confidence: `{summary['confidence']['median']}`",
        f"- keypoint visible ratio: `{summary['keypoint_visible_ratio']}`",
        f"- continuity IoU match rate: `{summary['continuity_iou_match_rate']}`",
        f"- detection drop events: `{detection_drop_events}`",
        f"- effective inference fps: `{summary['effective_inference_fps']:.2f}`",
        "",
    ]
    report_path = inference_dir / "report.md"
    temporary_report = report_path.with_suffix(".md.partial")
    temporary_report.write_text("\n".join(lines), encoding="utf-8")
    temporary_report.replace(report_path)
    return annotated_output, prediction_output, summary


def _load_anchor_scan(video_dir: Path) -> tuple[list[AnchorFeature], np.ndarray]:
    anchors_path = video_dir / "selection" / "anchors.jsonl"
    embedding_path = video_dir / "selection" / "anchor_embeddings.npy"
    anchors: list[AnchorFeature] = []
    with anchors_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("primary_pose_size") is not None:
                row["primary_pose_size"] = tuple(float(value) for value in row["primary_pose_size"])
            anchors.append(AnchorFeature(**row))
    embeddings = np.load(embedding_path)
    if len(anchors) != len(embeddings):
        raise RuntimeError(f"resume anchor/embedding mismatch: {len(anchors)} != {len(embeddings)}")
    return anchors, embeddings


def _source_snapshot(video: VideoMetadata) -> dict[str, Any]:
    return {
        "path": video.path.as_posix(),
        "size": video.file_size,
        "mtime_ns": video.mtime_ns,
        "duration_sec": video.duration_sec,
        "width": video.width,
        "height": video.height,
        "fps": video.fps,
        "frame_count": video.frame_count,
        "codec": video.codec,
    }


def run_video(
    video: VideoMetadata,
    config: NormalPerformanceConfig,
    model: Any,
    embedding_model: Any,
    *,
    force: bool,
    resume: bool,
    weights_sha256: str,
) -> dict[str, Any]:
    video_dir = config.output_dir / video.video_id
    if video_dir.exists() and force:
        shutil.rmtree(video_dir)
    if video_dir.exists() and not resume and any(video_dir.iterdir()):
        raise FileExistsError(
            f"output already exists for {video.video_id}: {video_dir}; use --resume or --force"
        )
    video_dir.mkdir(parents=True, exist_ok=True)
    completed_manifest = video_dir / "manifest.json"
    if resume and completed_manifest.is_file():
        manifest = json.loads(completed_manifest.read_text(encoding="utf-8"))
        required = [
            video_dir / "selected_source_10m.mp4",
            video_dir / "inference" / "yolo11s_pose_annotated_10m.mp4",
            video_dir / "inference" / "predictions.jsonl",
            video_dir / "inference" / "summary.json",
        ]
        if manifest.get("status") == "completed" and all(path.is_file() for path in required):
            LOGGER.info("[%s] completed output reused", video.video_id)
            return manifest

    source_before = _source_snapshot(video)
    selection_dir = video_dir / "selection"
    can_resume_scan = resume and (selection_dir / "anchors.jsonl").is_file() and (selection_dir / "anchor_embeddings.npy").is_file()
    if can_resume_scan:
        LOGGER.info("[%s] reusing anchor scan", video.video_id)
        anchors, embeddings = _load_anchor_scan(video_dir)
    else:
        anchors, embeddings = scan_anchors(video, config, model, embedding_model, video_dir)
    scores = score_windows(video, config, anchors, embeddings)
    write_selection_outputs(video, config, video_dir, anchors, scores)
    selected = scores[0]
    LOGGER.info(
        "[%s] selected %.1f-%.1f min score=%d (rep=%d person=%d pose=%d motion=%d invalid=%d)",
        video.video_id,
        selected.start_sec / 60.0,
        selected.end_sec / 60.0,
        selected.total_score,
        selected.representative_points,
        selected.presence_points,
        selected.pose_size_points,
        selected.temporal_motion_points,
        selected.invalid_penalty,
    )

    source_clip = video_dir / "selected_source_10m.mp4"
    if not (resume and source_clip.is_file()):
        source_clip = create_source_clip(video, selected, config, video_dir)
    annotated = video_dir / "inference" / "yolo11s_pose_annotated_10m.mp4"
    predictions = video_dir / "inference" / "predictions.jsonl"
    summary_path = video_dir / "inference" / "summary.json"
    if resume and annotated.is_file() and predictions.is_file() and summary_path.is_file():
        inference_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        annotated, predictions, inference_summary = run_full_inference(video, selected, config, model, video_dir)

    stat_after = video.path.stat()
    if stat_after.st_size != video.file_size or stat_after.st_mtime_ns != video.mtime_ns:
        raise RuntimeError(f"source video changed during processing: {video.path}")
    manifest = {
        "status": "completed",
        "purpose": "normal-operation qualitative/time-series smoke test; not labeled accuracy evaluation",
        "source": source_before,
        "source_unchanged": True,
        "model": {
            "weights": config.weights.as_posix(),
            "weights_sha256": weights_sha256,
            "task": "pose",
            "class": "face",
            "imgsz": config.imgsz,
            "confidence": config.confidence,
            "device": config.device,
        },
        "selection": scores[0].to_dict(),
        "outputs": {
            "source_clip": source_clip.relative_to(config.output_dir).as_posix(),
            "annotated_video": annotated.relative_to(config.output_dir).as_posix(),
            "predictions": predictions.relative_to(config.output_dir).as_posix(),
            "selection_report": (selection_dir / "selection_report.html").relative_to(config.output_dir).as_posix(),
        },
        "inference_summary": inference_summary,
        "config": config.resolved_dict(),
    }
    _write_json(completed_manifest, manifest)
    return manifest


def run_pipeline(config: NormalPerformanceConfig, *, force: bool = False, resume: bool = False) -> dict[str, Any]:
    if force and resume:
        raise ValueError("--force and --resume are mutually exclusive")
    videos = find_input_videos(config)
    if not videos:
        raise FileNotFoundError(f"no videos found in {config.input_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = config.output_dir / "config_resolved.yaml"
    temporary_config = resolved_path.with_suffix(".yaml.partial")
    temporary_config.write_text(yaml.safe_dump(config.resolved_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary_config.replace(resolved_path)

    from ultralytics import YOLO  # type: ignore

    model = YOLO(config.weights.as_posix())
    embedding_model = YOLO(config.weights.as_posix())
    if model.task != "pose":
        raise RuntimeError(f"configured YOLO model must be pose task, got {model.task!r}")
    if str(model.names.get(0, "")).lower() != "face":
        raise RuntimeError(f"configured YOLO class 0 must be face, got {model.names!r}")
    weights_sha256 = _sha256(config.weights)
    manifests: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, path in enumerate(videos, 1):
        LOGGER.info("video %d/%d: %s", index, len(videos), path.name)
        try:
            metadata = probe_video(path)
            if metadata.duration_sec + 1e-9 < config.window_duration_sec:
                raise RuntimeError(
                    f"video is shorter than {config.window_duration_sec:.0f}s: {metadata.duration_sec:.3f}s"
                )
            manifests.append(run_video(
                metadata,
                config,
                model,
                embedding_model,
                force=force,
                resume=resume,
                weights_sha256=weights_sha256,
            ))
        except Exception as error:
            LOGGER.exception("failed: %s", path)
            errors.append({"video": path.as_posix(), "error": str(error)})
    summary = {
        "status": "completed" if not errors else ("partial_failure" if manifests else "failed"),
        "videos_discovered": len(videos),
        "videos_completed": len(manifests),
        "elapsed_sec": time.perf_counter() - started,
        "video_manifests": [
            (config.output_dir / Path(row["outputs"]["source_clip"]).parts[0] / "manifest.json").as_posix()
            for row in manifests
        ],
        "errors": errors,
    }
    _write_json(config.output_dir / "run_summary.json", summary)
    if errors:
        detail = "\n".join(f"{row['video']}: {row['error']}" for row in errors)
        raise RuntimeError(f"normal-performance pipeline failed for {len(errors)} video(s):\n{detail}")
    return summary
