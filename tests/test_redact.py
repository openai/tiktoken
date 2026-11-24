import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

BLOCK = "before\n# ===== redact-beg =====\nsecret\n# ===== redact-end =====\nafter\n"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_redact_processes_non_ascii_filenames(tmp_path: Path) -> None:
    # scripts/redact.py asserts its repo root is named "tiktoken"
    root = tmp_path / "tiktoken"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts" / "redact.py", root / "scripts" / "redact.py")
    (root / "pyproject.toml").touch()

    names = ["café.txt", "plain.txt"]
    for name in names:
        (root / name).write_text(BLOCK)

    subprocess.run(["git", "init", "-q", "."], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)

    # core.quotePath defaults to true; redact.py must still find every file
    subprocess.run(
        [sys.executable, "scripts/redact.py", "--dry-run=false"],
        cwd=root,
        check=True,
        capture_output=True,
    )

    for name in names:
        assert (root / name).read_text() == "before\nafter\n"
