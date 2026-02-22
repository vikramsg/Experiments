"""Test the prompt generation logic."""

from lora_data._jargon_prompt_generator import generate_prompts, spell_out_for_tts

def test_generate_prompts():
    """Ensure prompts are generated deterministically and uniquely."""
    prompts1 = generate_prompts(num=50, seed=123)
    prompts2 = generate_prompts(num=50, seed=123)
    prompts3 = generate_prompts(num=50, seed=456)

    assert len(prompts1) == 50
    assert len(set(prompts1)) == 50
    assert prompts1 == prompts2
    assert prompts1 != prompts3
    assert any("justfile" in p for p in prompts1)

def test_spell_out_for_tts():
    """Ensure symbols and jargon are spelled out properly for TTS."""
    assert spell_out_for_tts("@refactor.md") == "at refactor dot m d"
    assert spell_out_for_tts("WER /tmp") == "w e r slash temp"
    assert spell_out_for_tts("_posts f5-tts-mlx") == "underscore posts f 5 t t s m l x"
    assert spell_out_for_tts("justfile") == "just file"
