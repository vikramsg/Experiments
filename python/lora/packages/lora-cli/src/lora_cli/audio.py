import sounddevice as sd
import numpy as np

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.frames = []
        self.stream = None

    def _callback(self, indata, frames, time, status):
        if self.is_recording:
            self.frames.append(indata.copy())

    def start(self):
        self.is_recording = True
        self.frames = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._callback
        )
        self.stream.start()

    def stop(self) -> np.ndarray:
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if not self.frames:
            return np.array([], dtype=np.float32)
            
        return np.concatenate(self.frames, axis=0).flatten()
