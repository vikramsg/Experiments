# Model Training & Evaluation Plan (End-to-End)

**Goal**: Split the recorded dataset, execute a background LoRA fine-tuning experiment on `Moonshine-Tiny`, and rigorously verify the resulting adapter against the baseline. 

**Execution Guidance (CRITICAL):**
The agent MUST NOT stop at any point during this phase. Once the training job is launched, the agent must continuously poll the experiment logs until the training is 100% complete. Once complete, it must immediately run the evaluation and report the results. The agent is strictly mandated to execute this end-to-end autonomously.

### Step 1: Pre-Training Fixes & Data Splitting
1.  **Fix Entrypoint**: Update `main.py` to point to the correct refactored runner path (`lora_training.runners.experiment`).
2.  **Dataset Split**: Execute `just split-dataset` to randomly partition the verified recordings (`data/my_voice_all.jsonl`) into `data/my_voice_train.jsonl` (80%) and `data/my_voice_eval.jsonl` (20%).

### Step 2: Experiment Launch
Launch the background experiment via `just run-experiment my_voice_tune` using the following hyperparameters:
*   **Model**: `UsefulSensors/moonshine-tiny`
*   **Train Data**: `data/my_voice_train.jsonl`
*   **Eval Data**: `data/my_voice_eval.jsonl`
*   **Steps**: `500` (sufficient for a small dataset of ~100 items to converge without catastrophic overfitting).
*   **Eval Interval**: `50`
*   **Learning Rate**: `3e-4`

### Step 3: Continuous Monitoring
Use the `just poll my_voice_tune` and `just status my_voice_tune` commands to stream the background logs. The agent must loop this monitoring process autonomously until the training script gracefully exits and the adapter is saved to `outputs/my_voice_tune/adapter`.

### Step 4: Post-Training Verification
Execute the STT transcribe script against the evaluation split using the newly trained adapter:
`just transcribe "--model-id UsefulSensors/moonshine-tiny --adapter-dir outputs/my_voice_tune/adapter --manifest data/my_voice_eval.jsonl --output outputs/my_voice_tune/eval_results.json"`

**Acceptance Criteria:**
*   The training successfully runs to completion without crashing.
*   The final transcribed predictions show marked improvement on coding jargon compared to the baseline.
*   The final WER and sample comparisons are printed to the console for user review.

## Reference

- docs/personalized/ - For all documentation related to personalized training.
- [docs/personalized/exp_04_vocabulary_generalization.md](docs/personalized/exp_04_vocabulary_generalization.md) - Revised strategy for High-Capacity and Vocabulary Generalization to solve vocabulary plateaus.
- docs/training_report.md - Template for training report
