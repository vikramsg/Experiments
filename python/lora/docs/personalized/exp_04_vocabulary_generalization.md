# Experiment v4: High-Capacity Vocabulary Generalization

## Objective
Overcome the WER improvement plateau observed in earlier experiments (v1-v3). The model needs to learn specific technical jargon (e.g., `@refactor.md`, `WER`, `justfile`) from a very small dataset (~66 training samples) without memorizing the acoustic specifics of the training set.

## Hypotheses
1. **The Seesaw of Data Scarcity vs. Complexity:** With only ~66 training examples, the model easily memorizes acoustics instead of learning new vocabulary. It needs architectural changes rather than just hyperparameter tuning to break out of this plateau.
2. **LoRA Capacity Bottleneck:** The default LoRA rank (`r=8`) provides too narrow a parameter bottleneck to simultaneously preserve the model's base knowledge while encoding entirely new phonetic-to-text mapping combinations for complex coding jargon.
3. **Weight-Decomposed LoRA (DoRA):** Standard LoRA struggles with magnitude vs. direction updates when data is scarce. DoRA effectively decouples these, an approach that has been empirically shown to drastically improve generalization on small datasets compared to standard LoRA.
4. **Regularization through Dropout:** To prevent the tiny base model (`moonshine-tiny`) from collapsing and perfectly memorizing the training audio clips, a higher dropout rate forces the network to rely on broader phonetic representations instead of memorizing specific acoustic artifacts.

## Proposed Hyperparameters
- **Model:** `UsefulSensors/moonshine-tiny`
- **Train Data:** `data/my_voice_train.jsonl`
- **Eval Data:** `data/my_voice_eval.jsonl`
- **Safety Data:** `data/heldout_manifest.jsonl`
- **Max Steps:** `200`
- **Learning Rate:** `1e-4`
- **LoRA Rank (`--lora-r`):** `32` (increased from 8)
- **LoRA Alpha (`--lora-alpha`):** `64`
- **LoRA Dropout (`--lora-dropout`):** `0.1` (increased from 0.05)
- **DoRA (`--use-dora`):** `True`

## Evaluation & Tests
1. **Domain WER Test:** Evaluate the trained adapter on `data/my_voice_eval.jsonl` to ensure WER drops significantly below the v2/v3 baseline of ~16.8%.
2. **Safety WER Test:** Evaluate on `data/heldout_manifest.jsonl` to ensure the safety WER remains stable around ~4.5%, proving no catastrophic forgetting occurred.
3. **Qualitative Jargon Test:** Manually inspect the transcription logs for specific punctuation and jargon retention (e.g., `@`, `_`, `WER`, `refactor.md`).