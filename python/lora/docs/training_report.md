# Training Report

## Run Summary
- Date: 2026-02-18
- Model: `UsefulSensors/moonshine-tiny`
- Device: `mps`
- Dataset: `data/train_plus_heldout_manifest.jsonl` (240 samples)
- Additional manifest: `data/train_manifest_short_10s.jsonl` (17 samples, <=10s from train manifest)
- Command: `LORA_TARGETS=q_proj,k_proj,v_proj,o_proj uv run python -m lora.runners.real_small --manifest-path data/train_plus_heldout_manifest.jsonl --max-steps 1600 --learning-rate 1e-5 --gradient-accumulation-steps 4 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --batch-size 1 --output-dir outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run1 --device mps`
- Repro command: same as above with output `outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run3`

## Configuration
- LoRA targets: `q_proj,k_proj,v_proj,o_proj`
- LoRA rank/alpha: `r=8`, `alpha=16`
- Dropout: `0.05`
- Learning rate: `1e-5`
- Steps: `1600`
- Gradient accumulation: `4`
- Batch size: `1`
- Seed: `42` (default)

## Results
- Held-out WER (`data/heldout_manifest.jsonl`)
  - Run1 baseline: `0.044583333333333336`
  - Run1 tuned: `0.04416666666666667`
  - Run3 baseline: `0.044583333333333336`
  - Run3 tuned: `0.04416666666666667`
- Short manifest WER (`data/train_manifest_short_10s.jsonl`)
  - Run1 baseline: `0.05084745762711865`
  - Run1 tuned: `0.05084745762711865`
  - Run3 baseline: `0.05084745762711865`
  - Run3 tuned: `0.05084745762711865`
- Runtime
  - Run1 elapsed: `882.06s`
  - Run3 elapsed: `895.01s`

## Sample Predictions (held-out, run1)
- index `0`
  - reference: "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"
  - baseline: "Concord returned to its place amidst the tents."
  - tuned: "Concord returned to its place amidst the tents."
- index `2`
  - reference: "CONGRATULATIONS WERE POURED IN UPON THE PRINCESS EVERYWHERE DURING HER JOURNEY"
  - baseline: "Congratulations were poured in upon the Princess everywhere during her journey."
  - tuned: "Congratulations were poured in upon the Princess everywhere during her journey,"
- index `7`
  - reference: "YOU WILL BE FRANK WITH ME I ALWAYS AM"
  - baseline: "You will be frank with me, I always am."
  - tuned: "You will be frank with me, I always am."

## Artifacts
- Run1 adapter: `outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run1/lora_adapter`
- Run1 processor: `outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run1/processor`
- Run1 reports: `outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run1/heldout_baseline.json`, `outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run1/heldout_tuned.json`, `outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run1/train_short10_baseline.json`, `outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run1/train_short10_tuned.json`
- Run3 adapter: `outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run3/lora_adapter`
- Run3 processor: `outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run3/processor`
- Run3 reports: `outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run3/heldout_baseline.json`, `outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run3/heldout_tuned.json`, `outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run3/train_short10_baseline.json`, `outputs/real_manifest_combined_lr1e5_r8_qkvo_20260218_run3/train_short10_tuned.json`

## Notes
- `data/train_manifest_short_10s.jsonl` is derived by filtering `data/train_manifest.jsonl` to samples with duration <=10s.
- Held-out WER improvement is reproducible across two runs with identical configs.
