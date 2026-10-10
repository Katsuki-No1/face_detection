from pathlib import Path
import os
import subprocess
import sys


def test_messenger_evaluation_script_imports_without_pythonpath(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, str(root / "scripts/evaluate_messenger_recovery.py"), "--help"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--success-marker" in result.stdout
