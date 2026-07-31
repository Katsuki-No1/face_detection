from __future__ import annotations

import io

from tools.train_dinov3_deimv2 import Tee


def test_tee_exposes_terminal_compatibility_for_wandb() -> None:
    first = io.StringIO()
    second = io.StringIO()
    tee = Tee(first, second)

    assert tee.isatty() is False
    assert tee.encoding is None
    assert tee.write("metric\n") == 7
    assert first.getvalue() == "metric\n"
    assert second.getvalue() == "metric\n"
