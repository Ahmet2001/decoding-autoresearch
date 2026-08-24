from __future__ import annotations

import hashlib
import json
import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import DecodingConfig


DECODING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
)
INTEGER_DECODING_FIELDS = {"top_k"}


@dataclass(frozen=True)
class AppConfig:
    root: Path
    raw: dict[str, Any]

    @property
    def ollama(self) -> dict[str, Any]:
        return self.raw["ollama"]

    @property
    def deepseek(self) -> dict[str, Any]:
        return self.raw["deepseek"]

    @property
    def benchmark(self) -> dict[str, Any]:
        return self.raw.get("benchmark", {})

    @property
    def metrics(self) -> dict[str, Any]:
        return self.raw["metrics"]

    @property
    def quality(self) -> dict[str, Any]:
        return self.raw["quality"]

    @property
    def run(self) -> dict[str, Any]:
        return self.raw["run"]

    @property
    def decoding_search(self) -> dict[str, Any]:
        return self.raw["decoding_search"]

    @property
    def algorithm(self) -> dict[str, Any]:
        return self.raw["algorithm"]

    def load_decoding(self) -> DecodingConfig:
        return load_decoding_config(self.root / "decoding_config.toml", self.decoding_search)


def load_config(path: Path) -> AppConfig:
    path = path.resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    required = {
        "ollama",
        "deepseek",
        "metrics",
        "quality",
        "run",
        "algorithm",
        "decoding_search",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(sorted(missing))}")
    bounds = raw["decoding_search"]
    if set(bounds) != set(DECODING_FIELDS):
        raise ValueError("decoding_search must define exactly the supported decoding fields")
    for field, limits in bounds.items():
        if not isinstance(limits, dict) or set(limits) != {"min", "max"}:
            raise ValueError(f"decoding_search.{field} must contain exactly min and max")
        if limits["min"] > limits["max"]:
            raise ValueError(f"decoding_search.{field} has min greater than max")
    config = AppConfig(path.parent, raw)
    metric_weights = [
        float(raw["metrics"][name])
        for name in (
            "loop_fraction_weight",
            "loop_incidence_weight",
            "truncation_weight",
            "reasoning_efficiency_weight",
        )
    ]
    if any(weight < 0 for weight in metric_weights) or not math.isclose(
        sum(metric_weights), 1.0, abs_tol=1e-9
    ):
        raise ValueError("metric weights must be non-negative and sum to 1.0")
    config.load_decoding()
    return config


def validate_decoding_values(
    values: dict[str, Any], bounds: dict[str, Any]
) -> DecodingConfig:
    if not isinstance(values, dict) or set(values) != set(DECODING_FIELDS):
        raise ValueError(
            "decoding config must contain exactly: " + ", ".join(DECODING_FIELDS)
        )
    normalized: dict[str, float | int] = {}
    for field in DECODING_FIELDS:
        value = values[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric")
        if field in INTEGER_DECODING_FIELDS:
            if not isinstance(value, int):
                raise ValueError(f"{field} must be an integer")
            normalized[field] = value
        else:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
            normalized[field] = value
        limits = bounds[field]
        if not float(limits["min"]) <= float(normalized[field]) <= float(limits["max"]):
            raise ValueError(
                f"{field}={normalized[field]} is outside "
                f"[{limits['min']}, {limits['max']}]"
            )
    return DecodingConfig(**normalized)


def load_decoding_config(path: Path, bounds: dict[str, Any]) -> DecodingConfig:
    if not path.exists():
        raise ValueError(f"Missing mutable decoding file: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) != {"decoding"} or not isinstance(raw["decoding"], dict):
        raise ValueError("decoding_config.toml must contain only a [decoding] table")
    return validate_decoding_values(raw["decoding"], bounds)


def render_decoding_config(config: DecodingConfig) -> str:
    values = config.to_dict()
    lines = ["# The only file changed by autonomous research.", "[decoding]"]
    for field in DECODING_FIELDS:
        value = values[field]
        rendered = str(value).lower() if isinstance(value, float) else str(value)
        lines.append(f"{field} = {rendered}")
    return "\n".join(lines) + "\n"


def decoding_hash(config: DecodingConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_dotenv(path: Path, *, override: bool = False) -> None:
    """Load the small KEY=VALUE subset needed by this project without another dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = value


def require_deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key or key == "replace-with-a-rotated-key":
        raise RuntimeError(
            "A rotated DEEPSEEK_API_KEY is required. Copy .env.example to .env and add it."
        )
    return key
