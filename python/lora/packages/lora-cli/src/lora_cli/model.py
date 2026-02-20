import numpy as np
import torch
from lora.model_utils import normalize_audio_rms
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


class SpeechRecognizer:
    def __init__(self, model_id: str, mock: bool = False):
        self.mock = mock
        self.model_id = model_id
        self.processor = None
        self.model = None
        
        if not self.mock:
            self._load_model()

    def _load_model(self):
        # Lazy loading to keep startup fast
        print("Loading model...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id)
        self.model.eval()

    def transcribe(self, audio_data: np.ndarray) -> str:
        if self.mock:
            return "This is a simulated transcription."

        if len(audio_data) == 0:
            raise ValueError("Cannot transcribe empty audio data. Audio recording may have failed.")

        # Normalize audio to match training preprocessing
        normalized_audio = normalize_audio_rms(audio_data.tolist(), target_rms=0.075)
        
        input_values = self.processor(
            normalized_audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_values

        with torch.no_grad():
            generated_ids = self.model.generate(input_values)

        transcription = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription
