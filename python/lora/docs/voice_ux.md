# Voice UX Design

## Goal
Create a "Live Mode" CLI that allows natural voice interaction with the Moonshine model on macOS.

## Core Experience: "The Walkie-Talkie"
To avoid the complexity of Voice Activity Detection (VAD) tuning, we will use a **Push-to-Talk (PTT)** mechanism. This offers the highest accuracy and best user control.

### Interaction Loop
1.  **Idle State:**
    *   Display: A clean status bar saying `[ HOLD SPACE TO SPEAK ]`.
    *   System: Mic is off. Model is loaded in memory (CoreML/MPS).

2.  **Recording State (User holds Spacebar):**
    *   Display: Changes to `[ LISTENING... ]`.
    *   Visual: A dynamic ASCII volume bar or simple waveform updates in real-time (e.g., `|||||||....`).
    *   System: Audio is captured into a buffer.

3.  **Processing State (User releases Spacebar):**
    *   Display: `[ THINKING... ]` with a spinner.
    *   System: Audio buffer is sent to the Moonshine model (Inference).

4.  **Result State:**
    *   Display: The transcribed text appears in the chat history.
    *   System: Returns to Idle.

## Technical Stack

| Component | Library | Reason |
| :--- | :--- | :--- |
| **UI/TUI** | `rich` | Beautiful, modern terminal formatting, live updating panels, and progress bars. |
| **Audio Input** | `sounddevice` | Simple, modern, and works great on macOS (unlike PyAudio which can be finicky). |
| **Key Detection** | `pynput` | Reliable detection of key press/release events on macOS. |
| **Processing** | `numpy` | Efficient audio buffer management. |

## Mockup (Rich TUI)

```text
╭────────────────────────── Moonshine Live ───────────────────────────╮
│                                                                     │
│ > User: Hello, how are you?                                         │
│ > Model: I am ready to transcribe.                                  │
│                                                                     │
│                                                                     │
│                                                                     │
╰─────────────────────────────────────────────────────────────────────╯
[ ● REC ] |||||||||||||||||||||........................... -20dB
```

## Implementation Plan
1.  **Prototype:** Create `scripts/live_mic.py`.
2.  **Audio Loop:** Implement `sounddevice` InputStream with a callback.
3.  **Trigger:** Connect `pynput` listener to start/stop the audio stream.
4.  **Integration:** Load the optimized Moonshine model and pass the audio buffer on release.
