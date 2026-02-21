import json
from pathlib import Path

import numpy as np
import pytest

from lora_data.recorder import (
    _append_to_manifest,
    _init_session,
    _is_silent,
    _load_completed_prompts,
    _remove_last_from_manifest,
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
    assert _is_silent(np.zeros(1000, dtype=np.int16)) == True
    
    # Low noise
    noise = np.random.randint(-100, 100, 1000).astype(np.int16)
    assert _is_silent(noise, threshold=500) == True
    
    # Human voice (spikes)
    voice = np.zeros(1000, dtype=np.int16)
    voice[500] = 5000
    assert _is_silent(voice, threshold=500) == False
    
    # Empty
    assert _is_silent(np.array([], dtype=np.int16)) == True

def test_load_completed_prompts(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    
    # Doesn't exist
    assert _load_completed_prompts(manifest) == set()
    
    # Create valid manifest
    with open(manifest, "w") as f:
        f.write(json.dumps({"audio": "1.wav", "text": "foo"}) + "\n")
        f.write("\n") # empty line
        f.write("not json\n") # invalid
        f.write(json.dumps({"audio": "2.wav", "text": "bar"}) + "\n")

    completed = _load_completed_prompts(manifest)
    assert completed == {"foo", "bar"}

def test_init_session():
    all_prompts = ["a", "b", "c", "d"]
    completed = {"b", "d"}
    
    session = _init_session(all_prompts, completed)
    
    assert session.total_prompts == 4
    assert session.completed_count == 2
    assert len(session.pending_prompts) == 2
    assert set(session.pending_prompts) == {"a", "c"}

def test_append_remove_manifest(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    
    _append_to_manifest(manifest, Path("1.wav"), "first")
    _append_to_manifest(manifest, Path("2.wav"), "second")
    
    lines = manifest.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "first"
    assert json.loads(lines[1])["text"] == "second"
    
    _remove_last_from_manifest(manifest)
    lines = manifest.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "first"
