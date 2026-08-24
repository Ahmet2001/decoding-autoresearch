from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import EvaluationReport, ResearchState


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_report(path: Path, report: EvaluationReport) -> None:
    atomic_write_json(path, report.to_dict())


def load_report(path: Path) -> EvaluationReport:
    return EvaluationReport.from_dict(read_json(path))


def load_state(path: Path) -> ResearchState:
    if not path.exists():
        return ResearchState()
    return ResearchState.from_dict(read_json(path))


def save_state(path: Path, state: ResearchState) -> None:
    atomic_write_json(path, state.to_dict())

