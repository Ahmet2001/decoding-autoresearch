from __future__ import annotations

import json
import re
from pathlib import Path

from .metrics import normalize_text
from .models import BenchmarkCase


VALID_SPLITS = {"screen", "validation"}
VALID_LANGUAGES = {"en", "tr"}


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                data = json.loads(raw_line)
                case = BenchmarkCase(**data)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid benchmark line {line_number}: {exc}") from exc
            if case.id in ids:
                raise ValueError(f"Duplicate benchmark id: {case.id}")
            if case.split not in VALID_SPLITS:
                raise ValueError(f"Invalid split for {case.id}: {case.split}")
            if case.language not in VALID_LANGUAGES:
                raise ValueError(f"Invalid language for {case.id}: {case.language}")
            if not case.messages or any(
                message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in case.messages
            ):
                raise ValueError(f"Invalid messages for {case.id}")
            ids.add(case.id)
            cases.append(case)
    screen = [case for case in cases if case.split == "screen"]
    validation = [case for case in cases if case.split == "validation"]
    if len(screen) != 12 or len(validation) != 24:
        raise ValueError("Benchmark must contain exactly 12 screen and 24 validation cases")
    for split, rows in (("screen", screen), ("validation", validation)):
        counts = {language: sum(row.language == language for row in rows) for language in VALID_LANGUAGES}
        if counts["en"] != counts["tr"]:
            raise ValueError(f"{split} cases must be evenly split between English and Turkish")
    return cases


def validate_answer(case: BenchmarkCase, answer: str) -> tuple[bool, str]:
    validator = case.validator
    kind = validator.get("type")
    normalized = normalize_text(answer).strip()
    if not normalized:
        return False, "empty answer"
    min_chars = int(validator.get("min_chars", 1))
    max_chars = int(validator.get("max_chars", 10000))
    if len(answer.strip()) < min_chars:
        return False, f"answer shorter than {min_chars} characters"
    if len(answer.strip()) > max_chars:
        return False, f"answer longer than {max_chars} characters"
    if kind == "judge":
        return True, "requires pairwise judge"
    if kind == "exact":
        expected = normalize_text(str(validator["value"])).strip()
        passed = normalized == expected
        return passed, "exact match" if passed else f"expected exact value {validator['value']!r}"
    if kind in {"contains_all", "contains_any"}:
        values = [normalize_text(str(value)) for value in validator.get("values", [])]
        matches = [value in normalized for value in values]
        passed = all(matches) if kind == "contains_all" else any(matches)
        return passed, kind if passed else f"missing expected content: {values}"
    if kind == "contains_groups":
        groups = [
            [normalize_text(str(value)) for value in group]
            for group in validator.get("groups", [])
        ]
        if not groups or any(not group for group in groups):
            raise ValueError(f"contains_groups requires non-empty groups for {case.id}")
        missing = [group for group in groups if not any(value in normalized for value in group)]
        passed = not missing
        return passed, "contains_groups" if passed else f"missing expected groups: {missing}"
    if kind == "regex":
        flags = re.IGNORECASE if not validator.get("case_sensitive", False) else 0
        passed = re.search(str(validator["pattern"]), answer.strip(), flags) is not None
        return passed, "regex match" if passed else "regex did not match"
    raise ValueError(f"Unknown validator type for {case.id}: {kind}")


def cases_for_split(cases: list[BenchmarkCase], split: str) -> list[BenchmarkCase]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return [case for case in cases if case.split == split]
