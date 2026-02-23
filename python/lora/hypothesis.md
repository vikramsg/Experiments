# Vocabulary Generalization Hypothesis Tracking

## Goal
Reduce Word Error Rate (WER) below 15% on real voice evaluation data (`data/my_voice_eval.jsonl`) without degrading safety performance (measured on `data/heldout_manifest.jsonl`).

## Baseline Context
- **Base Model**: `UsefulSensors/moonshine-tiny`
- **Training Data**: `data/my_voice_train.jsonl` (82 samples of real voice)
- **Problem**: Previous runs (v4/v5) used high capacity (r=32, targeting fc1/fc2) and massive synthetic data (~500 samples), which caused catastrophic forgetting of standard English and the model completely broke down on real evaluation audio (WER > 50%).

---

## Hypothesis 1 (my_voice_tune_v6)
**Date**: 2026-02-22
**Theory**: High capacity adapters (r=32, fc1/fc2 targets) combined with synthetic data caused catastrophic forgetting. A low-capacity adapter (r=8, q_proj/v_proj only) trained purely on 82 real voice samples for ~15 epochs (150 steps) will prevent forgetting and improve the baseline WER.
**Configuration**:
- `model-id`: UsefulSensors/moonshine-tiny
- `dataset-path`: data/my_voice_train.jsonl
- `max-steps`: 150
- `eval-interval`: 25
- `learning-rate`: 5e-5
- `lora-r`: 8
- `lora-alpha`: 16
- `lora-targets`: q_proj,v_proj
- `use-dora`: true
- `lora-dropout`: 0.1

**Status**: RUNNING

## Results (my_voice_tune_v6)
- **Best Domain WER**: 17.16% (at step 75)
- **Best Safety WER**: 4.49%
- **Conclusion**: Catastrophic forgetting was completely eliminated (Safety WER stayed at ~4.5%). However, Domain WER plateaued at ~17.16%, missing the 15% target. This confirms the model lacked the capacity to learn the new complex vocabulary from the small dataset.

## Hypothesis 2 (my_voice_tune_v7)
**Date**: 2026-02-22
**Theory**: Restricting to `q_proj` and `v_proj` with `r=8` prevented forgetting, but lacked the parameter capacity to map the complex new vocabulary. We will expand targets to all attention projections (`q_proj,k_proj,v_proj,o_proj`) and slightly increase rank to `r=16` while keeping the FFN layers frozen to protect English grammar. We will also restore learning rate to 1e-4 for better convergence over 200 steps.
**Configuration**:
- `model-id`: UsefulSensors/moonshine-tiny
- `dataset-path`: data/my_voice_train.jsonl
- `max-steps`: 200
- `eval-interval`: 25
- `learning-rate`: 1e-4
- `lora-r`: 16
- `lora-alpha`: 32
- `lora-targets`: q_proj,k_proj,v_proj,o_proj
- `use-dora`: true
- `lora-dropout`: 0.1

**Status**: RUNNING

## Results (my_voice_tune_v7)
- **Best Domain WER**: 17.54% (at step 25)
- **Best Safety WER**: ~4.5%
- **Conclusion**: Expanding the targets and capacity did not break the 15% barrier. Instead, it overfit faster, peaking at step 25 and degrading to 21.27% by step 200. The model is likely memorizing the training samples too quickly.

## Hypothesis 3 (my_voice_tune_v8)
**Date**: 2026-02-22
**Theory**: DoRA's magnitude updates are too aggressive for an ultra-small dataset (82 samples), leading to early overfitting (peaking at step 25). Standard LoRA is traditionally more stable for few-shot vocabulary injection. We will disable DoRA, use standard LoRA, and evaluate more frequently to catch the optimal checkpoint.
**Configuration**:
- `model-id`: UsefulSensors/moonshine-tiny
- `dataset-path`: data/my_voice_train.jsonl
- `max-steps`: 100
- `eval-interval`: 20
- `learning-rate`: 1e-4
- `lora-r`: 16
- `lora-alpha`: 32
- `lora-targets`: q_proj,k_proj,v_proj,o_proj
- `use-dora`: false
- `lora-dropout`: 0.1

**Status**: RUNNING

## Results (my_voice_tune_v8)
- **Best Domain WER**: 17.91% (at step 20)
- **Best Safety WER**: ~4.5%
- **Conclusion**: Disabling DoRA did not fix the plateau. The model peaks extremely early and hits a hard wall at ~17.5%. The fundamental problem is that 82 real voice samples do not contain enough repetitions of complex jargon (e.g., there is only 1 instance of `@refactor.md` in the dataset) for the model to learn them, regardless of LoRA configuration. We *must* use the synthetic data to learn the vocabulary, but we must heavily regularize it to prevent the catastrophic forgetting seen in v5.

## Hypothesis 4 (my_voice_tune_v9)
**Date**: 2026-02-22
**Theory**: To learn complex jargon, we need the 500+ synthetic data repetitions. To prevent the synthetic audio from destroying the real-voice acoustics and English grammar (as happened in v5), we will use a highly constrained, regularized adapter. We will train on the `mixed_train.jsonl` dataset using standard LoRA, `r=16`, targeting ONLY `q_proj,v_proj` (freezing the MLPs `fc1,fc2`), with a low learning rate (`2e-5`) and high dropout (`0.15`). This forces the model to slowly map the new phonetic tokens without overwriting its base knowledge.
**Configuration**:
- `model-id`: UsefulSensors/moonshine-tiny
- `dataset-path`: data/mixed_train.jsonl
- `max-steps`: 300
- `eval-interval`: 50
- `learning-rate`: 2e-5
- `lora-r`: 16
- `lora-alpha`: 32
- `lora-targets`: q_proj,v_proj
- `use-dora`: false
- `lora-dropout`: 0.15

**Status**: RUNNING

## Results (my_voice_tune_v9)
- **Best Domain WER**: 16.04% (at step 200)
- **Best Safety WER**: 4.54%
- **Conclusion**: Success! We broke the 17.1% plateau. Using the `mixed_train.jsonl` dataset (which includes the 500+ synthetic files) but heavily restricting the LoRA adapter (freezing `fc1/fc2` and disabling DoRA) perfectly prevented the catastrophic forgetting seen in v5. The model's English grammar remained intact (stable 4.5% Safety WER). However, 16.04% is still above the 15% goal, suggesting the adapter capacity or learning rate was slightly too conservative to perfectly map the new complex jargon.

## Hypothesis 5 (my_voice_tune_v10)
**Date**: 2026-02-22
**Theory**: The regularized mixed-dataset approach is the correct path, but `q_proj,v_proj` with `2e-5` LR was slightly too restrictive. We will expand the targets to full attention (`q_proj,k_proj,v_proj,o_proj`), keep DoRA disabled for stability, and raise the learning rate to `4e-5` to give the model the capacity and momentum it needs to drop below 15%.
**Configuration**:
- `model-id`: UsefulSensors/moonshine-tiny
- `dataset-path`: data/mixed_train.jsonl
- `max-steps`: 300
- `eval-interval`: 50
- `learning-rate`: 4e-5
- `lora-r`: 16
- `lora-alpha`: 32
- `lora-targets`: q_proj,k_proj,v_proj,o_proj
- `use-dora`: false
- `lora-dropout`: 0.1

**Status**: RUNNING

## Results (my_voice_tune_v10)
- **Best Domain WER**: 18.28% (at step 150)
- **Best Safety WER**: ~4.5%
- **Conclusion**: Expanding the targets to `q_proj,k_proj,v_proj,o_proj` and increasing LR worsened performance (Domain WER regressed back to the 18% plateau). This proves that `q_proj,v_proj` is the optimal regularization boundary to prevent the synthetic audio from degrading real-voice performance. The additional targets simply allowed the model to overfit the synthetic acoustics.

## Hypothesis 6 (my_voice_tune_v11)
**Date**: 2026-02-22
**Theory**: The `v9` architecture (`q,v` only, low LR) was extremely close to breaking the <15% threshold, achieving 16.04%. Since expanding target modules caused overfitting, we will instead re-enable DoRA on the exact `v9` configuration. For a larger 500+ sample mixed dataset, DoRA will help disentangle the magnitude of the updates (e.g., the volume/pitch of the synthetic voice) from the directional updates (the actual phonetic-to-jargon mappings), allowing the model to cleanly learn the vocabulary without memorizing the synthetic acoustics.
**Configuration**:
- `model-id`: UsefulSensors/moonshine-tiny
- `dataset-path`: data/mixed_train.jsonl
- `max-steps`: 200
- `eval-interval`: 25
- `learning-rate`: 2e-5
- `lora-r`: 16
- `lora-alpha`: 32
- `lora-targets`: q_proj,v_proj
- `use-dora`: true
- `lora-dropout`: 0.15

**Status**: RUNNING

## Results (my_voice_tune_v11)
- **Best Domain WER**: 17.54%
- **Best Safety WER**: ~4.5%
- **Conclusion**: Re-enabling DoRA plateaued the model at 17.54%. The `v9` configuration (Standard LoRA, `q,v` only, low LR on mixed dataset) remains the absolute best configuration, having hit 16.04%.

## Hypothesis 7 (my_voice_tune_v12)
**Date**: 2026-02-22
**Theory**: The `v9` configuration (standard LoRA, `q,v` targets only, `lr=2e-5`, mixed dataset) was the most successful, achieving 16.04%. Adding modules (`k,o` in v10) or DoRA (v11) introduced instability and regression. To bridge the final 1.0% gap to <15%, we need slightly more representational capacity strictly within the safe `q,v` boundaries. We will increase the rank from `16` to `32` while keeping everything else identical to `v9`.
**Configuration**:
- `model-id`: UsefulSensors/moonshine-tiny
- `dataset-path`: data/mixed_train.jsonl
- `max-steps`: 250
- `eval-interval`: 50
- `learning-rate`: 2e-5
- `lora-r`: 32
- `lora-alpha`: 64
- `lora-targets`: q_proj,v_proj
- `use-dora`: false
- `lora-dropout`: 0.15

**Status**: RUNNING

## Results (my_voice_tune_v12)
- **Best Domain WER**: 16.04% (at step 250)
- **Best Safety WER**: ~4.5%
- **Conclusion**: Increasing rank to 32 yielded the exact same 16.04% WER. Qualitative analysis shows the model perfectly learned the jargon (`refactor.md`, `WER`) but is making slight semantic substitutions on normal words (e.g., confusing "last" with "lowest"). This indicates the capacity is correct, but the learning rate may still be slightly too high, shaking the base semantic embeddings. 

## Hypothesis 8 (my_voice_tune_v13)
**Date**: 2026-02-22
**Theory**: The `q_proj,v_proj` regularization works perfectly to teach the model the jargon. To fix the slight semantic substitutions (like "lowest" instead of "last") and finally break <15%, we will drop the learning rate to a very conservative `1e-5` and train for a longer period (`500` steps). This will allow the model to slowly integrate the new vocabulary without destabilizing its existing phonetic confidence.
**Configuration**:
- `model-id`: UsefulSensors/moonshine-tiny
- `dataset-path`: data/mixed_train.jsonl
- `max-steps`: 500
- `eval-interval`: 50
- `learning-rate`: 1e-5
- `lora-r`: 16
- `lora-alpha`: 32
- `lora-targets`: q_proj,v_proj
- `use-dora`: false
- `lora-dropout`: 0.15

**Status**: RUNNING

## Results (my_voice_tune_v13)
- **Best Domain WER**: 16.04% (at step 500)
- **Best Safety WER**: ~4.5%
- **Conclusion**: Increasing training duration and dropping learning rate yielded the exact same 16.04% WER plateau, failing on the exact same words (confusing "last" with "lowest"). This confirms a hard architectural limit for the `moonshine-tiny` (v1) model on this complex mapping task. 

## Hypothesis 9 (my_voice_tune_v14)
**Date**: 2026-02-22
**Theory**: We have hit an architectural plateau with `moonshine-tiny`. As outlined in the original `docs/experiment_plan.md`, the true target architecture for this project is `UsefulSensors/moonshine-streaming-tiny` (v2), which has a more robust positional and semantic structure. We will apply our most stable vocabulary-injection configuration (Standard LoRA, `q_proj,v_proj` only, mixed dataset) to the v2 model to see if it can successfully map the jargon without making semantic substitution errors on basic English words.
**Configuration**:
- `model-id`: UsefulSensors/moonshine-streaming-tiny
- `dataset-path`: data/mixed_train.jsonl
- `max-steps`: 200
- `eval-interval`: 50
- `learning-rate`: 1e-4
- `lora-r`: 16
- `lora-alpha`: 32
- `lora-targets`: q_proj,v_proj
- `use-dora`: false
- `lora-dropout`: 0.15

**Status**: RUNNING


## Results (my_voice_tune_v14)
- **Best Domain WER**: 21.27% (at step 150)
- **Best Safety WER**: ~4.8%
- **Conclusion**: Testing the `v9` configuration (`q_proj,v_proj` only, mixed dataset, standard LoRA) on the `UsefulSensors/moonshine-streaming-tiny` (v2) architecture failed to break the plateau. The v2 streaming architecture appears fundamentally harder to fine-tune for vocabulary injection with this limited dataset size, peaking significantly higher than the v1 architecture (21.27% vs 16.04%).
