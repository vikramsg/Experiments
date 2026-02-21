import pytest

from lora_data.recorder import parse_prompts


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
    
    # Assert multi-line strings work natively or through explicit \n
    expected_multiline = "Read @docs/training_report.md\nAnd review the code base."
    assert prompts[2] == expected_multiline
    
    assert prompts[3] == "Fix the linting errors."

def test_parse_prompts_empty_or_invalid():
    """Test that empty blocks or blocks with invalid data types fail fast."""
    assert parse_prompts("prompts = []") == []
    
    with pytest.raises(ValueError, match="The 'prompts' key must be a list of strings"):
        parse_prompts('prompts = "not a list"')
