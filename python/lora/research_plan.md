# Synthetic Audio Quality Verification Plan

**Goal**: Verify that the F5-TTS local voice cloning setup produces intelligible, accurate English audio that sounds like the user *before* generating a massive dataset and running a training job. 

**Motivation**: Previous attempts resulted in non-English or hallucinated audio because the reference clip was too short and the TTS model couldn't handle raw symbols (like `@` or `_`). We need to prove our fixes (longer reference audio + phonetic spelling of symbols) actually work.

### Verification Methodology
The verification relies on a "round-trip" test: Generate audio from text -> Transcribe the audio back to text -> Compare.

1. **Target Selection**: Select 3-5 representative sentences containing our target jargon.
   * *Example*: "Please check the @refactor.md file."
2. **Text Normalization**: Convert the target text into a phonetically pronounceable format for the TTS model.
   * *Example*: "Please check the at refactor dot m d file."
3. **Audio Generation**: Use `f5-tts-mlx` with a long, clean reference audio of the user's voice (`data/raw_audio/my_voice/clip_ec16024b.wav`) to generate synthetic `.wav` files for these normalized sentences.
4. **Transcription (STT)**: Pass the generated `.wav` files through the **base** `UsefulSensors/moonshine-tiny` model (without any adapters) to transcribe the audio.
5. **Agent Evaluation (The Judgment)**: The agent will compare the Moonshine transcriptions against the *normalized* spoken text. 
   * **Success Criteria**: The transcription must be in English and phonetically consistent with the intended spoken text. (e.g., If Moonshine outputs "Please check the at refactor dot md file", it means the audio clearly articulated those words). It does not need to perfectly reconstruct the raw symbols (`@`), as teaching the model to map "at" to `@` is the job of the downstream training phase.
   * **Failure Criteria**: If Moonshine outputs gibberish, non-English text, or completely unrelated words, the audio generation is failing and must be fixed.

### Next Steps
* **If Verified**: Proceed to the full end-to-end generation and training pipeline (Phase 1 & Phase 2 from `plan.md`).
* **If Failed**: Halt and iterate on the F5-TTS generation parameters, reference audio, or text normalization strategy until the output is consistently intelligible.