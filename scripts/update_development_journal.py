#!/usr/bin/env python3
"""Add idempotent, structured entries to the project development journal."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOURNAL = PROJECT_ROOT / "docs" / "development_journal.md"
ENTRY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ENTRIES_MARKER = "<!-- development-journal-entries -->"
PENDING_REFLECTION = "未記入（結果確認後、ユーザー本人の言葉を追記する）"


@dataclass(frozen=True)
class JournalEntry:
    entry_id: str
    date: str
    time: str
    title: str
    status: str
    source: str
    design: tuple[str, ...]
    implementation: tuple[str, ...]
    training_conditions: tuple[str, ...]
    training_results: tuple[str, ...]
    artifacts: tuple[str, ...]
    verification: tuple[str, ...]
    user_reflection: str | None
    git_revision: str
    git_dirty: str


def _run_git(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_context() -> tuple[str, str]:
    revision = _run_git("rev-parse", "--short=12", "HEAD") or "unavailable"
    status = _run_git("status", "--porcelain")
    if status is None:
        dirty = "unavailable"
    else:
        dirty = "dirty" if status else "clean"
    return revision, dirty


def _format_bullets(values: Sequence[str], empty: str) -> str:
    normalized = [value.strip() for value in values if value.strip()]
    if not normalized:
        normalized = [empty]
    lines: list[str] = []
    for value in normalized:
        value_lines = value.splitlines() or [value]
        lines.append(f"- {value_lines[0]}")
        lines.extend(f"  {line}" for line in value_lines[1:])
    return "\n".join(lines)


def _format_reflection(value: str | None) -> str:
    if value is None or not value.strip():
        return f"> {PENDING_REFLECTION}"
    lines = value.strip().splitlines()
    return "\n".join(["> **ユーザー原文:**", ">", *(f"> {line}" for line in lines)])


def render_entry(entry: JournalEntry) -> str:
    design = _format_bullets(entry.design, "記録対象となる追加の設計判断なし")
    implementation = _format_bullets(
        entry.implementation,
        "実装変更なし",
    )
    conditions = _format_bullets(
        entry.training_conditions,
        "not run（このエントリでは学習を実行していない）",
    )
    results = _format_bullets(
        entry.training_results,
        "not run（学習結果なし）",
    )
    artifacts = _format_bullets(entry.artifacts, "新規出力なし")
    verification = _format_bullets(entry.verification, "検証未実施")
    reflection = _format_reflection(entry.user_reflection)
    return f"""<!-- dev-journal-entry:start id={entry.entry_id} -->
### {entry.time} — {entry.title}

- ID: `{entry.entry_id}`
- 状態: `{entry.status}`
- 記録元: `{entry.source}`
- Git: `{entry.git_revision}` (`{entry.git_dirty}`)

#### 1. 設計意図・判断根拠

{design}

実装:

{implementation}

#### 2. 学習条件・学習結果

条件:

{conditions}

結果:

{results}

#### 3. 出力・検証・可視化

出力:

{artifacts}

検証と意図:

{verification}

#### 4. 結果に対するユーザーの考え・認識

{reflection}
<!-- dev-journal-entry:end id={entry.entry_id} -->"""


def _entry_block_pattern(entry_id: str) -> re.Pattern[str]:
    escaped = re.escape(entry_id)
    return re.compile(
        rf"<!-- dev-journal-entry:start id={escaped} -->.*?"
        rf"<!-- dev-journal-entry:end id={escaped} -->",
        re.DOTALL,
    )


def _locked_rewrite(journal_path: Path, transform: Callable[[str], str]) -> None:
    """Serialize journal edits so concurrent agents cannot lose one another's entry."""
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.touch(exist_ok=True)
    with journal_path.open("r+", encoding="utf-8") as journal_file:
        fcntl.flock(journal_file.fileno(), fcntl.LOCK_EX)
        journal_file.seek(0)
        text = journal_file.read()
        updated = transform(text)
        journal_file.seek(0)
        journal_file.write(updated.rstrip() + "\n")
        journal_file.truncate()
        journal_file.flush()
        os.fsync(journal_file.fileno())
        fcntl.flock(journal_file.fileno(), fcntl.LOCK_UN)


def add_entry(journal_path: Path, entry: JournalEntry) -> str:
    block = render_entry(entry)
    existing = _entry_block_pattern(entry.entry_id)

    def transform(text: str) -> str:
        if existing.search(text):
            return existing.sub(lambda _match: block, text, count=1)
        date_heading = f"## {entry.date}"
        if date_heading in text:
            insert_at = text.index(date_heading) + len(date_heading)
            return text[:insert_at] + "\n\n" + block + text[insert_at:]
        if ENTRIES_MARKER not in text:
            raise ValueError(
                f"journal is missing required marker: {ENTRIES_MARKER}"
            )
        insertion = f"{ENTRIES_MARKER}\n\n## {entry.date}\n\n{block}"
        return text.replace(ENTRIES_MARKER, insertion, 1)

    _locked_rewrite(journal_path, transform)
    return entry.entry_id


def update_reflection(journal_path: Path, entry_id: str, reflection: str) -> None:
    def transform(text: str) -> str:
        entry_match = _entry_block_pattern(entry_id).search(text)
        if entry_match is None:
            raise ValueError(f"journal entry not found: {entry_id}")
        block = entry_match.group(0)
        heading = "#### 4. 結果に対するユーザーの考え・認識"
        section_pattern = re.compile(
            rf"({re.escape(heading)}\n\n).*?(\n<!-- dev-journal-entry:end id=)",
            re.DOTALL,
        )

        def replace_section(match: re.Match[str]) -> str:
            return match.group(1) + _format_reflection(reflection) + match.group(2)

        updated_block, count = section_pattern.subn(replace_section, block, count=1)
        if count != 1:
            raise ValueError(f"reflection section is malformed for entry: {entry_id}")
        return text[: entry_match.start()] + updated_block + text[entry_match.end() :]

    _locked_rewrite(journal_path, transform)


def _valid_entry_id(value: str) -> str:
    if not ENTRY_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "entry id must contain only letters, numbers, dot, underscore, or hyphen"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add", help="add or replace one logical entry")
    add.add_argument("--entry-id", required=True, type=_valid_entry_id)
    add.add_argument("--title", required=True)
    add.add_argument(
        "--status",
        choices=("in_progress", "completed", "failed", "blocked"),
        default="completed",
    )
    add.add_argument("--source", default="Codex")
    add.add_argument("--date", help="YYYY-MM-DD; defaults to Asia/Tokyo today")
    add.add_argument("--time", help="HH:MM with zone label; defaults to current JST")
    add.add_argument("--design", action="append", default=[])
    add.add_argument("--implementation", action="append", default=[])
    add.add_argument("--training-condition", action="append", default=[])
    add.add_argument("--training-result", action="append", default=[])
    add.add_argument("--artifact", action="append", default=[])
    add.add_argument("--verification", action="append", default=[])
    add.add_argument("--user-reflection")

    reflect = subparsers.add_parser(
        "reflect", help="replace section 4 with the user's verbatim reflection"
    )
    reflect.add_argument("--entry-id", required=True, type=_valid_entry_id)
    reflect.add_argument("--user-reflection", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    journal_path = args.journal.resolve()
    if args.command == "reflect":
        update_reflection(journal_path, args.entry_id, args.user_reflection)
        print(f"updated reflection: {args.entry_id}")
        return 0

    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    revision, dirty = git_context()
    entry = JournalEntry(
        entry_id=args.entry_id,
        date=args.date or now.date().isoformat(),
        time=args.time or now.strftime("%H:%M JST"),
        title=args.title.strip(),
        status=args.status,
        source=args.source.strip(),
        design=tuple(args.design),
        implementation=tuple(args.implementation),
        training_conditions=tuple(args.training_condition),
        training_results=tuple(args.training_result),
        artifacts=tuple(args.artifact),
        verification=tuple(args.verification),
        user_reflection=args.user_reflection,
        git_revision=revision,
        git_dirty=dirty,
    )
    add_entry(journal_path, entry)
    print(f"updated journal entry: {entry.entry_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
