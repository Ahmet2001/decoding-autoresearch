from pathlib import Path

import pytest

from prompt_research.config import (
    decoding_hash,
    load_config,
    load_decoding_config,
    render_decoding_config,
    validate_decoding_values,
)


ROOT = Path(__file__).parents[1]


def test_decoding_round_trip_and_stable_hash(tmp_path: Path) -> None:
    app = load_config(ROOT / "config.toml")
    original = app.load_decoding()
    path = tmp_path / "decoding_config.toml"
    path.write_text(render_decoding_config(original), encoding="utf-8")
    loaded = load_decoding_config(path, app.decoding_search)
    assert loaded == original
    assert decoding_hash(loaded) == decoding_hash(original)


def test_decoding_rejects_unknown_non_integer_and_out_of_range_values() -> None:
    app = load_config(ROOT / "config.toml")
    values = app.load_decoding().to_dict()
    with pytest.raises(ValueError, match="exactly"):
        validate_decoding_values({**values, "unknown": 1}, app.decoding_search)
    with pytest.raises(ValueError, match="integer"):
        validate_decoding_values({**values, "top_k": 20.5}, app.decoding_search)
    with pytest.raises(ValueError, match="outside"):
        validate_decoding_values({**values, "temperature": 9.0}, app.decoding_search)
