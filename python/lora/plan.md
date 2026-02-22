# Model Training & Evaluation Plan: Synthetic Vocabulary Expansion (End-to-End)

**Goal**: Overcome the vocabulary generalization plateau by generating a massive synthetic dataset using local voice cloning (F5-TTS MLX), mixing it with real recordings, training a high-capacity DoRA adapter on `Moonshine-Tiny`, and rigorously verifying the improvements on real held-out data.

**Execution Guidance (CRITICAL ZERO-STOP MANDATE):**
The agent MUST NOT stop or ask for permission between any of the steps outlined below. Once the user triggers the execution, the agent must autonomously orchestrate the entire two-phase pipeline—from the initial proof-of-concept through the full-scale generation, training, and final evaluation—in a single continuous block of work. It will only return control to the user after the final Word Error Rate (WER) and qualitative transcriptions of the FULL run are printed to the console. If Phase 1 fails, it will halt and report the error.

---

### Phase 1: Proof of Concept (Mini-Run)
Before spending hours on the full generation and training, prove the pipeline works end-to-end.
1. **Tooling & Setup**: Autonomously ensure `f5-tts-mlx` and dependencies are installed.
2. **Mini Generation**: Generate just 5 synthetic audio files heavily saturated with the target jargon.
3. **Mini Mix**: Combine these 5 synthetic files with the real training data (`data/mixed_train_mini.jsonl`).
4. **Mini Training**: Launch a short background training experiment (`just run-experiment my_voice_tune_v5_mini` for 50 steps).
5. **Mini Verification**: Monitor logs until completion. Run a quick transcription verify to ensure the model produces text without crashing and logs exist.

### Phase 2: Full Scale Run
Immediately upon successful completion of Phase 1, seamlessly transition to Phase 2.
1. **Full Generation**: Generate 500+ synthetic `.wav` files mimicking the user's voice speaking jargon.
2. **Full Mix**: Output to `data/synthetic_train.jsonl` and combine with `data/my_voice_train.jsonl` into `data/mixed_train.jsonl`.
3. **High-Capacity Experiment**: Launch `just run-experiment my_voice_tune_v5` with:
   * **Max Steps**: `1000`
   * **Eval Interval**: `100`
   * **Learning Rate**: `1e-4`
   * **Adapter Config**: `--lora-r 32`, `--lora-alpha 64`, `--lora-dropout 0.1`, `--use-dora`
4. **Continuous Autonomous Monitoring**: Poll the experiment logs (`just poll my_voice_tune_v5`) in a loop.
5. **Post-Training Verification**: Transcribe the evaluation split using the best v5 adapter.

**Acceptance Criteria:**
1. The pipeline transitions from Phase 1 to Phase 2 autonomously.
2. The final primary WER on `data/my_voice_eval.jsonl` drops significantly.
3. Qualitative verification shows correct transcription of symbols like `@` and jargon like `WER`.
4. Safety WER remains stable.

## Reference
- docs/personalized/exp_04_vocabulary_generalization.md