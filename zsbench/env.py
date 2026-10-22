"""The repo-root .env: settings and API keys. Never print its values."""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Copy KEY=value lines into os.environ. Variables already set win; blank values are skipped."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if value and key not in os.environ:
            os.environ[key] = value
