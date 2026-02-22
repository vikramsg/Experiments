
import numpy as np
import pytest

from lora_data.recorder import (
    _init_session,
    _is_silent,
    parse_prompts,
)


def test_parse_prompts_toml():
    """Test that TOML array is correctly parsed into individual prompts."""
    toml_content = """
    prompts = [
        ".",
        "{",
        "Read @docs/training_report.md\\nAnd review the code base.",
        "Fix the linting errors."
    ]
    """

    prompts = parse_prompts(toml_content)

    assert len(prompts) == 4

    assert prompts[0] == "."
    assert prompts[1] == "{"

    expected_multiline = "Read @docs/training_report.md\nAnd review the code base."
    assert prompts[2] == expected_multiline

    assert prompts[3] == "Fix the linting errors."


def test_parse_prompts_empty_or_invalid():
    """Test that empty blocks or blocks with invalid data types fail fast."""
    assert parse_prompts("prompts = []") == []

    with pytest.raises(ValueError, match="The 'prompts' key must be a list of strings"):
        parse_prompts('prompts = "not a list"')


def test_is_silent():
    # True zeros
    assert bool(_is_silent(np.zeros(1000, dtype=np.int16))) is True

    # Low noise
    noise = np.random.randint(-100, 100, 1000).astype(np.int16)
    assert bool(_is_silent(noise, threshold=500)) is True

    # Human voice (spikes)
    voice = np.zeros(1000, dtype=np.int16)
    voice[500] = 5000
    assert bool(_is_silent(voice, threshold=500)) is False

    # Empty
    assert bool(_is_silent(np.array([], dtype=np.int16))) is True


def test_init_session():
    all_prompts = ["a", "b", "c", "d"]
    completed = {"b", "d"}

    session = _init_session(all_prompts, completed)

    assert session.total_prompts == 4
    assert session.completed_count == 2
    assert len(session.pending_prompts) == 2
    assert set(session.pending_prompts) == {"a", "c"}
