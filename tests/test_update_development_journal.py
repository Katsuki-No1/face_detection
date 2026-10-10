from __future__ import annotations

from pathlib import Path

from scripts.update_development_journal import (
    ENTRIES_MARKER,
    JournalEntry,
    PENDING_REFLECTION,
    add_entry,
    update_reflection,
)


def make_entry(**overrides: object) -> JournalEntry:
    values: dict[str, object] = {
        "entry_id": "20260914-journal",
        "date": "2026-09-14",
        "time": "15:30 JST",
        "title": "開発日誌を導入",
        "status": "completed",
        "source": "test",
        "design": ("日付単位で一元化する",),
        "implementation": ("追記スクリプトを追加",),
        "training_conditions": (),
        "training_results": (),
        "artifacts": ("docs/development_journal.md",),
        "verification": ("単体テストを実行",),
        "user_reflection": None,
        "git_revision": "abc123",
        "git_dirty": "dirty",
    }
    values.update(overrides)
    return JournalEntry(**values)  # type: ignore[arg-type]


def write_empty_journal(path: Path) -> None:
    path.write_text(f"# 開発日誌\n\n{ENTRIES_MARKER}\n", encoding="utf-8")


def test_add_entry_creates_date_and_all_four_sections(tmp_path: Path) -> None:
    journal = tmp_path / "journal.md"
    write_empty_journal(journal)

    add_entry(journal, make_entry())

    text = journal.read_text(encoding="utf-8")
    assert "## 2026-09-14" in text
    assert "#### 1. 設計意図・判断根拠" in text
    assert "#### 2. 学習条件・学習結果" in text
    assert "#### 3. 出力・検証・可視化" in text
    assert "#### 4. 結果に対するユーザーの考え・認識" in text
    assert PENDING_REFLECTION in text
    assert "not run（このエントリでは学習を実行していない）" in text


def test_same_entry_id_replaces_instead_of_duplicating(tmp_path: Path) -> None:
    journal = tmp_path / "journal.md"
    write_empty_journal(journal)
    add_entry(journal, make_entry())

    add_entry(journal, make_entry(title="更新後のタイトル"))

    text = journal.read_text(encoding="utf-8")
    assert text.count("dev-journal-entry:start id=20260914-journal") == 1
    assert "更新後のタイトル" in text
    assert "開発日誌を導入" not in text


def test_new_entries_are_inserted_first_within_same_date(tmp_path: Path) -> None:
    journal = tmp_path / "journal.md"
    write_empty_journal(journal)
    add_entry(journal, make_entry(entry_id="first", title="先の作業"))
    add_entry(journal, make_entry(entry_id="second", title="後の作業"))

    text = journal.read_text(encoding="utf-8")
    assert text.index("後の作業") < text.index("先の作業")


def test_reflection_is_updated_verbatim_without_changing_other_sections(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "journal.md"
    write_empty_journal(journal)
    add_entry(journal, make_entry())

    reflection = "精度は改善した。\nただし小さい顔はまだ弱い。\\1"
    update_reflection(journal, "20260914-journal", reflection)

    text = journal.read_text(encoding="utf-8")
    assert "> 精度は改善した。" in text
    assert "> ただし小さい顔はまだ弱い。\\1" in text
    assert PENDING_REFLECTION not in text
    assert "日付単位で一元化する" in text
