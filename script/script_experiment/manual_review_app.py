import argparse
import csv
import html
import json
import mimetypes
import os
import sys
import tempfile
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_TEMPLATE = Path("experiments/scrfd_mediapipe_baseline/reports/manual_review_template.csv")
DEFAULT_RESULTS_JSON = Path("experiments/scrfd_mediapipe_baseline/reports/manual_review_results.json")
DEFAULT_RESULTS_CSV = Path("experiments/scrfd_mediapipe_baseline/reports/manual_review_results.csv")

EDITABLE_FIELDS = [
    "review_status",
    "failure_type",
    "reason_primary",
    "reason_secondary",
    "reason_multi",
    "confidence_note",
    "review_note",
]
FAILURE_TYPES = [
    "",
    "ok",
    "head_miss",
    "face_miss",
    "head_and_face_miss",
    "fp",
    "duplicate",
    "duplicate_annotation",
    "eval_mismatch",
    "head_miss;fp",
]
REASONS = ["", "small", "large", "mask", "partial", "dark", "angle", "blur", "occlusion", "head_box", "unclear"]


def normalize_failure_type(value: str) -> str:
    mapping = {
        "miss": "head_miss",
        "miss;fp": "head_miss;fp",
    }
    return mapping.get(value, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local web app for frame-level manual review.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def row_key(row: dict[str, Any]) -> str:
    parts = [
        row.get("annotation_kind", ""),
        row.get("annotation_mode", ""),
        row.get("image_stem", ""),
        row.get("image_name", ""),
    ]
    return "\u241f".join(parts)


def load_template(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = [dict(row) for row in reader]
        for row in rows:
            row["suggested_failure_type"] = normalize_failure_type(row.get("suggested_failure_type", ""))
            row["failure_type"] = normalize_failure_type(row.get("failure_type", ""))
        return list(reader.fieldnames or []), rows


def load_results(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and "reviews" in data and isinstance(data["reviews"], dict):
        reviews = {str(key): dict(value) for key, value in data["reviews"].items()}
        for review in reviews.values():
            review["failure_type"] = normalize_failure_type(str(review.get("failure_type", "")))
        return reviews
    return {}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        file.write(text)
        temp_name = file.name
    os.replace(temp_name, path)


def merge_rows(rows: list[dict[str, str]], reviews: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    merged = []
    for row in rows:
        copy = dict(row)
        review = reviews.get(row_key(row), {})
        for field in EDITABLE_FIELDS:
            if field in review:
                copy[field] = str(review[field])
        merged.append(copy)
    return merged


def progress(rows: list[dict[str, str]]) -> dict[str, int]:
    done = sum(1 for row in rows if row.get("review_status") == "done")
    skipped = sum(1 for row in rows if row.get("review_status") == "skipped")
    counts = {
        "total": len(rows),
        "done": done,
        "skipped": skipped,
        "remaining": len(rows) - done - skipped,
        "ok": 0,
        "head_miss": 0,
        "face_miss": 0,
        "head_and_face_miss": 0,
        "fp": 0,
        "duplicate": 0,
        "duplicate_annotation": 0,
        "head_miss_fp": 0,
        "eval_mismatch": 0,
    }
    for row in rows:
        failure_type = row.get("failure_type") or row.get("suggested_failure_type", "")
        if failure_type == "head_miss;fp":
            counts["head_miss_fp"] += 1
        elif failure_type in counts:
            counts[failure_type] += 1
    return counts


def write_results_json(path: Path, reviews: dict[str, dict[str, str]]) -> None:
    payload = {
        "schema_version": "1.0",
        "updated_at": int(time.time()),
        "reviews": reviews,
    }
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_results_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]], reviews: dict[str, dict[str, str]]) -> None:
    merged = merge_rows(rows, reviews)
    out_fields = list(fieldnames)
    for field in EDITABLE_FIELDS:
        if field not in out_fields:
            out_fields.insert(0, field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)


def app_html() -> str:
    failure_options = "".join(f"<option value='{html.escape(value)}'>{html.escape(value or '-')}</option>" for value in FAILURE_TYPES)
    reason_options = "".join(f"<option value='{html.escape(value)}'>{html.escape(value or '-')}</option>" for value in REASONS)
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Manual Review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #15171a;
      --muted: #667085;
      --line: #d8dde6;
      --accent: #0f766e;
      --accent-soft: #d9f4ef;
      --danger: #b42318;
      --warn: #b54708;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 10px 16px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
    }}
    .title {{
      display: flex;
      gap: 10px;
      align-items: baseline;
      min-width: 0;
    }}
    h1 {{
      font-size: 18px;
      line-height: 1.2;
      margin: 0;
    }}
    .status {{
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    .toolbar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    button, select, input, textarea {{
      font: inherit;
    }}
    button, .filter {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
    }}
    button:hover, .filter.active {{
      border-color: var(--accent);
      background: var(--accent-soft);
    }}
    button.primary {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 390px;
      gap: 12px;
      padding: 12px;
      height: calc(100vh - 58px);
    }}
    .image-pane, .side-pane {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 0;
    }}
    .image-pane {{
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 10px;
      overflow: auto;
    }}
    #overlay {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border-radius: 4px;
    }}
    .side-pane {{
      overflow: auto;
      padding: 14px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin: 12px 0;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfe;
    }}
    .metric b {{
      display: block;
      font-size: 18px;
    }}
    .metric span {{
      color: var(--muted);
      font-size: 12px;
    }}
    label {{
      display: block;
      margin-top: 11px;
      color: #344054;
      font-size: 13px;
      font-weight: 600;
    }}
    select, input, textarea {{
      width: 100%;
      margin-top: 5px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fff;
      color: var(--ink);
    }}
    textarea {{
      resize: vertical;
      min-height: 66px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      word-break: break-all;
    }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      margin: 2px 4px 2px 0;
      background: #eef2f6;
      color: #344054;
      font-size: 12px;
    }}
    .badge.warn {{ background: #fff3e6; color: var(--warn); }}
    .badge.danger {{ background: #fee4e2; color: var(--danger); }}
    .nav {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 12px;
    }}
    .hint {{
      margin-top: 12px;
      padding: 10px;
      border-radius: 6px;
      background: #f2f4f7;
      color: #475467;
      font-size: 13px;
      line-height: 1.45;
    }}
    @media (max-width: 900px) {{
      header {{ grid-template-columns: 1fr; }}
      .toolbar {{ justify-content: flex-start; }}
      main {{
        grid-template-columns: 1fr;
        height: auto;
      }}
      .image-pane {{ min-height: 55vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="title">
      <h1>Manual Review</h1>
      <div id="progress" class="status"></div>
    </div>
    <div class="toolbar">
      <button class="filter active" data-filter="all">all</button>
      <button class="filter" data-filter="todo">todo</button>
      <button class="filter" data-filter="skipped">skipped</button>
      <button class="filter" data-filter="head_miss">head_miss</button>
      <button class="filter" data-filter="face_miss">face_miss</button>
      <button class="filter" data-filter="fp">fp</button>
      <button class="filter" data-filter="duplicate">duplicate</button>
      <button class="filter" data-filter="duplicate_annotation">duplicate_annotation</button>
      <button class="filter" data-filter="head_miss;fp">head_miss;fp</button>
      <button class="filter" data-filter="ok">ok</button>
      <button class="filter" data-filter="simple">simple</button>
      <button class="filter" data-filter="skeleton">skeleton</button>
      <button id="exportCsv" class="primary">CSV Export</button>
    </div>
  </header>
  <main>
    <section class="image-pane">
      <img id="overlay" alt="overlay">
    </section>
    <aside class="side-pane">
      <div class="meta">
        <div><b id="position"></b></div>
        <div id="imageName"></div>
        <div id="stem"></div>
      </div>
      <div id="badges"></div>
      <div class="metrics">
        <div class="metric"><b id="tp">0</b><span>TP</span></div>
        <div class="metric"><b id="fp">0</b><span>FP</span></div>
        <div class="metric"><b id="fn">0</b><span>FN</span></div>
      </div>
      <div class="metrics">
        <div class="metric"><b id="gt">0</b><span>GT Head</span></div>
        <div class="metric"><b id="det">0</b><span>Detection</span></div>
        <div class="metric"><b id="iou">0</b><span>mean IoU</span></div>
      </div>
      <form id="reviewForm">
        <label>review_status</label>
        <select id="review_status">
          <option value=""></option>
          <option value="todo">todo</option>
          <option value="done">done</option>
          <option value="skipped">skipped</option>
        </select>
        <label>failure_type</label>
        <select id="failure_type">{failure_options}</select>
        <label>reason_primary</label>
        <select id="reason_primary">{reason_options}</select>
        <label>reason_secondary</label>
        <select id="reason_secondary">{reason_options}</select>
        <label>reason_multi</label>
        <input id="reason_multi" placeholder="dark;angle">
        <label>confidence_note</label>
        <input id="confidence_note" placeholder="原因があいまいな場合のメモ">
        <label>review_note</label>
        <textarea id="review_note"></textarea>
      </form>
      <div class="nav">
        <button id="prev">Prev</button>
        <button id="next">Next</button>
      </div>
      <div id="saveState" class="hint">Ready</div>
      <div class="hint">
        failure_type は head_miss / face_miss / fp を分けます。
        mask, angle, dark などは face_miss などの原因として reason に入れてください。
        head_box はHead矩形の広さ・狭さ・ズレが原因と思われる場合のreasonです。
      </div>
    </aside>
  </main>
  <script>
    const editableFields = {json.dumps(EDITABLE_FIELDS)};
    let rows = [];
    let filtered = [];
    let index = 0;
    let filter = "all";
    let saveTimer = null;

    const $ = (id) => document.getElementById(id);

    async function api(path, options) {{
      const response = await fetch(path, options);
      if (!response.ok) {{
        throw new Error(await response.text());
      }}
      return response.json();
    }}

    function rowId(row) {{
      return row.id;
    }}

    function effectiveFailureType(row) {{
      return row.failure_type || row.suggested_failure_type || "";
    }}

    function applyFilter() {{
      filtered = rows.filter((row) => {{
        const type = effectiveFailureType(row);
        if (filter === "all") return true;
        if (filter === "todo") return row.review_status !== "done";
        if (filter === "skipped") return row.review_status === "skipped";
        if (filter === "simple" || filter === "skeleton") return row.annotation_mode === filter;
        return type === filter;
      }});
      if (index >= filtered.length) index = Math.max(0, filtered.length - 1);
      render();
    }}

    function renderProgress(progress) {{
      $("progress").textContent =
        `total ${{progress.total}} / done ${{progress.done}} / skipped ${{progress.skipped}} / remaining ${{progress.remaining}} ` +
        `| head_miss ${{progress.head_miss}} / face_miss ${{progress.face_miss}} ` +
        `/ fp ${{progress.fp}} / duplicate ${{progress.duplicate}} / duplicate_annotation ${{progress.duplicate_annotation}} / ok ${{progress.ok}}`;
    }}

    function render() {{
      if (!filtered.length) {{
        $("overlay").removeAttribute("src");
        $("position").textContent = "No records";
        return;
      }}
      const row = filtered[index];
      $("position").textContent = `${{index + 1}} / ${{filtered.length}}`;
      $("imageName").textContent = row.image_name;
      $("stem").textContent = `${{row.annotation_mode}} | ${{row.image_stem}}`;
      $("overlay").src = `/image?path=${{encodeURIComponent(row.overlay_path)}}`;
      $("tp").textContent = row.true_positive || "0";
      $("fp").textContent = row.false_positive || "0";
      $("fn").textContent = row.false_negative || "0";
      $("gt").textContent = row.ground_truth_count || "0";
      $("det").textContent = row.detection_count || "0";
      $("iou").textContent = Number(row.mean_iou || 0).toFixed(3);
      $("badges").innerHTML = [
        `<span class="badge">${{escapeHtml(row.annotation_mode || "")}}</span>`,
        `<span class="badge warn">suggested: ${{escapeHtml(row.suggested_failure_type || "-")}}</span>`,
        Number(row.false_negative || 0) > 0 ? `<span class="badge danger">FN ${{escapeHtml(row.false_negative)}}</span>` : "",
        Number(row.false_positive || 0) > 0 ? `<span class="badge warn">FP ${{escapeHtml(row.false_positive)}}</span>` : "",
      ].join("");
      for (const field of editableFields) {{
        $(field).value = row[field] || "";
      }}
      $("saveState").textContent = "Ready";
    }}

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      }}[char]));
    }}

    function currentRow() {{
      return filtered[index];
    }}

    function collectForm() {{
      const patch = {{}};
      for (const field of editableFields) {{
        patch[field] = $(field).value;
      }}
      return patch;
    }}

    async function saveNow() {{
      const row = currentRow();
      if (!row) return;
      Object.assign(row, collectForm());
      $("saveState").textContent = "Saving...";
      const result = await api("/api/save", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ id: rowId(row), review: collectForm() }}),
      }});
      renderProgress(result.progress);
      $("saveState").textContent = "Saved";
    }}

    function scheduleSave() {{
      const row = currentRow();
      if (!row) return;
      Object.assign(row, collectForm());
      $("saveState").textContent = "Unsaved changes";
      clearTimeout(saveTimer);
      saveTimer = setTimeout(() => {{
        saveNow().catch((error) => $("saveState").textContent = `Save failed: ${{error.message}}`);
      }}, 350);
    }}

    function move(delta) {{
      if (!filtered.length) return;
      saveNow().catch(() => {{}});
      index = Math.max(0, Math.min(filtered.length - 1, index + delta));
      render();
    }}

    async function init() {{
      const state = await api("/api/state");
      rows = state.rows;
      renderProgress(state.progress);
      applyFilter();
    }}

    document.querySelectorAll(".filter").forEach((button) => {{
      button.addEventListener("click", () => {{
        document.querySelectorAll(".filter").forEach((b) => b.classList.remove("active"));
        button.classList.add("active");
        filter = button.dataset.filter;
        index = 0;
        applyFilter();
      }});
    }});
    for (const field of editableFields) {{
      document.addEventListener("change", (event) => {{
        if (event.target && event.target.id === field) scheduleSave();
      }});
      document.addEventListener("input", (event) => {{
        if (event.target && event.target.id === field && ["reason_multi", "confidence_note", "review_note"].includes(field)) scheduleSave();
      }});
    }}
    $("prev").addEventListener("click", () => move(-1));
    $("next").addEventListener("click", () => move(1));
    $("exportCsv").addEventListener("click", async () => {{
      await saveNow();
      const result = await api("/api/export", {{ method: "POST" }});
      $("saveState").textContent = `CSV exported: ${{result.path}}`;
      window.location.href = result.download_url || "/download/results.csv";
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
      if (event.key === "ArrowLeft") move(-1);
      if (event.key === "ArrowRight") move(1);
    }});
    init().catch((error) => {{
      $("saveState").textContent = `Load failed: ${{error.message}}`;
    }});
  </script>
</body>
</html>
"""


class ReviewApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = Path.cwd().resolve()
        self.template_path = args.template
        self.results_json_path = args.results_json
        self.results_csv_path = args.results_csv
        self.fieldnames, self.template_rows = load_template(self.template_path)
        self.reviews = load_results(self.results_json_path)

    def rows_for_client(self) -> list[dict[str, str]]:
        rows = []
        for row in merge_rows(self.template_rows, self.reviews):
            copy = dict(row)
            copy["id"] = row_key(row)
            rows.append(copy)
        return rows

    def save_review(self, key: str, review: dict[str, Any]) -> dict[str, int]:
        clean = {field: str(review.get(field, "")) for field in EDITABLE_FIELDS}
        self.reviews[key] = clean
        write_results_json(self.results_json_path, self.reviews)
        return progress(self.rows_for_client())

    def export_csv(self) -> Path:
        write_results_csv(self.results_csv_path, self.fieldnames, self.template_rows, self.reviews)
        return self.results_csv_path

    def image_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.root / path
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(str(raw_path))
        if self.root not in resolved.parents and resolved != self.root:
            raise PermissionError(str(raw_path))
        return resolved


class Handler(BaseHTTPRequestHandler):
    app: ReviewApp

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, status: HTTPStatus = HTTPStatus.OK, content_type: str = "text/html; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_text(app_html())
            return
        if parsed.path == "/api/state":
            rows = self.app.rows_for_client()
            self.send_json({"rows": rows, "progress": progress(rows)})
            return
        if parsed.path == "/image":
            params = urllib.parse.parse_qs(parsed.query)
            raw_path = params.get("path", [""])[0]
            try:
                path = self.app.image_path(raw_path)
                content = path.read_bytes()
            except Exception as error:
                self.send_text(str(error), HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8")
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if parsed.path == "/download/results.csv":
            try:
                path = self.app.export_csv()
                content = path.read_bytes()
            except Exception as error:
                self.send_text(str(error), HTTPStatus.INTERNAL_SERVER_ERROR, "text/plain; charset=utf-8")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="manual_review_results.csv"')
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        self.send_text("not found", HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/save":
            payload = self.read_json()
            key = str(payload.get("id", ""))
            review = payload.get("review", {})
            if not key or not isinstance(review, dict):
                self.send_json({"error": "invalid payload"}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"progress": self.app.save_review(key, review)})
            return
        if parsed.path == "/api/export":
            path = self.app.export_csv()
            self.send_json({"path": str(path), "download_url": "/download/results.csv"})
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def main() -> int:
    args = parse_args()
    if not args.template.is_file():
        raise FileNotFoundError(f"template not found: {args.template}")

    Handler.app = ReviewApp(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"manual review app: {url}")
    print(f"template: {args.template}")
    print(f"results_json: {args.results_json}")
    print(f"results_csv: {args.results_csv}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
