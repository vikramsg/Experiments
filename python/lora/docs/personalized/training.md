# Personalized Voice Command Fine-Tuning Guide

## Overview
This document outlines the strategy, learnings, and step-by-step plan for fine-tuning the Moonshine v2 architecture to accurately transcribe a specific user's voice dictating technical coding commands (e.g., "revert", "v2", "slash", "git status"). 

This is required because pre-trained ASR models (including Moonshine) are heavily biased toward standard conversational English or audiobook literature and frequently mistranscribe technical jargon, punctuation commands, and specific developer workflows.

---

## 1. Crucial Architectural Learnings (Feb 2026)

Extensive baseline testing on the `UsefulSensors/moonshine-streaming-tiny` (v2) model yielded specific constraints for successful fine-tuning:

*   **Universal LoRA is Required:** Attempting to isolate LoRA strictly to the Decoder (to preserve the position-free encoder) or using DoRA (Weight-Decomposed LoRA) causes catastrophic unlearning and immediate regression on the v2 architecture at standard learning rates (`1e-5`).
*   **The Stable Configuration:** To successfully adapt the v2 model without degrading its baseline phonetic understanding, you **must** use standard LoRA applied universally across all attention projections: `q_proj,k_proj,v_proj,o_proj`.
*   **Guardrails Matter:** Always evaluate against a standard, non-technical dataset (e.g., `data/heldout_manifest.jsonl`) during training to ensure the model doesn't overfit so heavily to your voice/code that it forgets how to transcribe regular English. We tolerate a max regression of `0.2%` on this heldout set.

---

## 2. The Custom Training Plan

### Phase 1: Data Collection & Preparation
To teach the model your specific terminology, you must provide it with examples of your voice saying those exact words.

1.  **Record Audio Samples:**
    *   Record yourself speaking technical commands exactly as you would to a coding agent.
    *   *Examples:* "revert that last commit", "open slash src slash main dot py", "switch to the v2 architecture", "run grep search for normalize audio".
    *   *Format:* 16kHz, Mono, WAV or FLAC format. Keep clips under 20 seconds.
2.  **Transcribe Exactly:**
    *   Create exact text transcripts of what you said. If you want the model to output symbols, spell them out how you speak them, or vice versa depending on your agent's parsing logic (e.g., if you say "slash", transcribe it as "slash" or "/" based on what the coding agent expects).
3.  **Build Manifests:**
    *   Use the repository's tooling to generate standard JSONL manifests.
    *   **Training Manifest (`data/my_voice_train.jsonl`):** ~80% of your recordings (aim for at least 100-200 samples to start).
    *   **Domain Evaluation Manifest (`data/my_voice_eval.jsonl`):** ~20% of your recordings (used to verify the model is actually learning your terminology).
    *   *Tooling:* Use `src/lora_data/recorder.py` to simultaneously record and compile your audio and transcripts into the required `{"audio": "path/to.wav", "text": "YOUR TRANSCRIPT"}` JSONL format, using `data/coding_prompts.toml` for the text.

### Phase 2: Execution Recipe
Run the exact configuration proven to be stable on v2. Ensure you use the newly exposed `--lora-targets` argument.

```bash
uv run python src/lora_training/runners/experiment.py 
    --output-dir outputs/my_voice_v2_lora 
    --model-id UsefulSensors/moonshine-streaming-tiny 
    --max-steps 1000 
    --learning-rate 1e-5 
    --eval-interval 200 
    --dataset-path data/my_voice_train.jsonl 
    --manifest-path data/my_voice_eval.jsonl 
    --safety-manifest-path data/heldout_manifest.jsonl 
    --max-seconds 20 
    --seed 42 
    --lora-targets "q_proj,k_proj,v_proj,o_proj"
```

### Phase 3: Evaluation Criteria
Monitor the `outputs/my_voice_v2_lora/experiment.log` for the `Interval WER` evaluations every 200 steps.

*   **Primary Goal (Domain WER):** The WER on `data/my_voice_eval.jsonl` should drop significantly from its baseline. Because the baseline Moonshine model likely fails heavily on your specific technical jargon, there will be high headroom. Look for an absolute reduction of >= 5.0%.
*   **Safety Goal (Heldout WER):** The WER on `data/heldout_manifest.jsonl` must not exceed its baseline by more than `0.2%`. If it does, you are overfitting: reduce `--max-steps` or lower the `--learning-rate`.

### Phase 4: Inference & Deployment
Once training completes and the metrics look good, the best adapter will be saved to `outputs/my_voice_v2_lora/lora_adapter_best`.

You can test it directly:
```bash
uv run python src/lora_training/transcribe.py 
    --model-id UsefulSensors/moonshine-streaming-tiny 
    --adapter-dir outputs/my_voice_v2_lora/lora_adapter_best 
    --processor-dir outputs/my_voice_v2_lora/processor 
    --audio-list data/my_voice_eval.jsonl 
    --output outputs/my_voice_results.json 
    --device mps
```

*For deployment via Apple Silicon Neural Engine (ANE), refer to `docs/inference.md` to merge this adapter into the base model and convert it to CoreML.*