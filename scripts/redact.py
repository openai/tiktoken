import argparse
import re
import subprocess
from pathlib import Path


def redact_file(path: Path, dry_run: bool) -> None:
    if not path.exists() or path.is_dir():
        return

    text = path.read_text()
    if not text:
        return

    first_line = text.splitlines()[0]
    if "redact" in first_line:
        if not dry_run:
            path.unlink()
        print(f"Deleted {path}")
        return

    pattern = "|".join(
        r" *" + re.escape(x)
        for x in [
            "# ===== redact-beg =====\n",
            "# ===== redact-end =====\n",
            "<!--- redact-beg -->\n",
            "<!--- redact-end -->\n",
        ]
    )

    if re.search(pattern, text):
        redacted_text = "".join(re.split(pattern, text)[::2])
        if not dry_run:
            path.write_text(redacted_text)
        print(f"Redacted {path}")
        return

    print(f"Skipped {path}")


def redact(dry_run: bool) -> None:
    tiktoken_root = Path(__file__).parent.parent
    assert tiktoken_root.name == "tiktoken"
    assert (tiktoken_root / "pyproject.toml").exists()

    try:
        output = subprocess.check_output(["git", "ls-files"], cwd=tiktoken_root, text=True)
        paths = [Path(p) for p in output.splitlines()]
    except subprocess.CalledProcessError:
        paths = list(tiktoken_root.glob("**/*"))

    for path in paths:
        redact_file(path, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=lambda x: not x or x[0].lower() != "f", default=True)
    args = parser.parse_args()
    redact(args.dry_run)
    if args.dry_run:
        print("Dry run, use --dry-run=false to actually redact files")


if __name__ == "__main__":
    main()
