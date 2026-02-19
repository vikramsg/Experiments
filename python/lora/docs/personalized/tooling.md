# Data Collection Tooling Guide

To effectively fine-tune the Moonshine ASR model for your personalized voice and coding commands, you need a highly efficient workflow to generate a large volume of precise `<audio, text>` pairs.

This document outlines the Python-based tooling required to rapidly build this dataset.

## Core Objective
Generate a robust JSONL manifest (`data/my_voice_train.jsonl` and `data/my_voice_eval.jsonl`) without manual audio slicing or transcription typing overhead.

## The Tooling Stack

You will need a unified Python script (e.g., `scripts/record_dataset.py`) that handles prompting, recording, and manifest generation in a rapid-fire loop.

### 1. The Prompt Generator
The script needs a list of target phrases to train the model on. It should randomly select phrases from a predefined list and present them to you on the screen to read aloud.

**Features:**
- Load prompts from a raw text file (`data/coding_prompts.txt`).
- Prompts should include technical jargon, punctuation commands, and standard agent workflows:
  - `"revert that last commit"`
  - `"open slash src slash main dot py"`
  - `"change the variable x to snake case"`
  - `"run grep search for normalize audio"`
  - `"wrap the function in a try catch block"`

### 2. The Audio Recorder
The script must handle microphone input directly.

**Requirements:**
- Use `pyaudio` or `sounddevice` to capture microphone input.
- **Critical:** Record at exactly **16,000 Hz** (16kHz), **Mono** channel, **16-bit PCM**. This perfectly matches Moonshine's expected input, eliminating runtime resampling issues during training.
- Save files using the `wave` module directly to a dedicated directory (e.g., `data/raw_audio/my_voice/`).
- Use unique filenames (e.g., UUIDs or timestamps: `clip_1708365021.wav`).

### 3. The Interactive Loop (CLI UX)
The process must be frictionless. You shouldn't have to touch the mouse.

**Workflow:**
1. The script prints a prompt: `[PROMPT]: "revert that last commit"`
2. It waits for a keypress (e.g., `Spacebar` or `Enter`) to start recording.
3. It prints: `🔴 RECORDING... (Press Spacebar to stop)`
4. You read the prompt.
5. You press `Spacebar` to stop.
6. The script immediately saves the `.wav` file, appends the entry to the manifest, and presents the next prompt.
7. Allow a keypress (e.g., `r`) to discard and re-record the current prompt if you stumbled.

### 4. Manifest Generator
As each recording is saved, the script must immediately append the metadata to a JSONL file to prevent data loss.

**Format (JSONL):**
```json
{"audio": "data/raw_audio/my_voice/clip_123.wav", "text": "revert that last commit"}
```

### 5. Utilities
- **Audio Normalization:** Integrate `librosa` or `pydub` to automatically trim silence from the beginning and end of the recorded clips before saving.
- **Train/Eval Splitter:** A final utility script (`scripts/split_manifest.py`) to take your master `my_voice_all.jsonl` and randomly partition it into an 80% `my_voice_train.jsonl` and a 20% `my_voice_eval.jsonl`. Ensure the test split has adequate representation of complex symbols or commands.