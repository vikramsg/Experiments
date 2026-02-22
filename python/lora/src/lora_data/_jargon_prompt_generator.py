"""Prompt generation logic for synthetic ASR data."""

import random

JARGON = [
    "@refactor.md",
    "justfile",
    "uv run",
    "WER",
    "_posts",
    "@app",
    "/tmp",
    "nohup.out",
    "data/mixed_train.jsonl",
    "f5-tts-mlx",
    "lora_adapter",
    "moonshine-tiny",
]

TEMPLATES = [
    "Please check the {} file.",
    "Did you run {} yet?",
    "The {} is showing a high error rate.",
    "Look at the {} for more details.",
    "Update the {} with the new parameters.",
    "Why is {} failing?",
    "We need to decrease the {} significantly.",
    "Add {} to the configuration.",
    "The output is saved in {}.",
    "Ensure {} is properly formatted.",
]


def generate_prompts(num: int = 500, seed: int = 42) -> list[str]:
    """Deterministically generate training prompts containing jargon."""
    random.seed(seed)
    prompts = set()
    while len(prompts) < num:
        jargon1 = random.choice(JARGON)
        jargon2 = random.choice(JARGON)
        if random.random() < 0.3 and jargon1 != jargon2:
            sentence = random.choice(TEMPLATES).format(f"{jargon1} and {jargon2}")
        else:
            sentence = random.choice(TEMPLATES).format(jargon1)
        prompts.add(sentence)

    # Sort to ensure absolute determinism after set conversion
    return sorted(list(prompts))


def spell_out_for_tts(text: str) -> str:
    """Expand symbols into pronounceable text for the TTS engine."""
    replacements = {
        "@": " at ",
        "_": " underscore ",
        "/": " slash ",
        ".md": " dot m d ",
        ".jsonl": " dot json l ",
        ".out": " dot out ",
        "uv": " u v ",
        "WER": " w e r ",
        "f5-tts-mlx": " f 5 t t s m l x ",
        "tmp": " temp ",
        "nohup": " no hup ",
        "justfile": " just file ",
    }
    spoken_text = text
    for symbol, spoken in replacements.items():
        spoken_text = spoken_text.replace(symbol, spoken)
    return " ".join(spoken_text.split())
