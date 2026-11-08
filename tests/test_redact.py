import shutil
import subprocess
import sys
from pathlib import Path


def test_redact_uses_repository_paths_when_called_from_elsewhere(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "tiktoken"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).parents[1] / "scripts" / "redact.py", scripts / "redact.py"
    )
    (repo / "pyproject.toml").write_text("[project]\nname = 'tiktoken'\n")

    original = (
        "before\n# ===== redact-beg =====\nsecret\n# ===== redact-end =====\nafter\n"
    )
    tracked = repo / "tracked.txt"
    tracked.write_text(original)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)

    caller = tmp_path / "caller"
    caller.mkdir()
    same_named = caller / "tracked.txt"
    same_named.write_text(original)

    subprocess.run(
        [sys.executable, str(scripts / "redact.py"), "--dry-run=false"],
        cwd=caller,
        check=True,
        capture_output=True,
        text=True,
    )

    assert tracked.read_text() == "before\nafter\n"
    assert same_named.read_text() == original
