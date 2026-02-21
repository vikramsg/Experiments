from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lora_training.transcribe import transcribe


@pytest.fixture
def mock_dependencies():
    with patch("lora_training.transcribe.load_processor") as mock_lp, \
         patch("lora_training.transcribe.AutoConfig.from_pretrained") as mock_ac, \
         patch("lora_training.transcribe.MoonshineForConditionalGeneration.from_pretrained") as mock_mf, \
         patch("lora_training.transcribe.PeftModel.from_pretrained") as mock_pm, \
         patch("lora_training.transcribe.normalize_audio_rms") as mock_rms, \
         patch("lora_training.transcribe.choose_device", return_value="cpu"):
        
        # Setup mocks
        mock_processor = MagicMock()
        mock_processor.feature_extractor.sampling_rate = 16000
        mock_processor.tokenizer.batch_decode.return_value = ["mocked prediction"]
        
        mock_inputs = MagicMock()
        mock_inputs.input_values = MagicMock()
        mock_inputs.attention_mask = MagicMock()
        mock_processor.return_value = mock_inputs
        mock_lp.return_value = mock_processor
        
        mock_config = MagicMock()
        mock_config.model_type = "moonshine"
        mock_ac.return_value = mock_config
        
        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_pm.return_value = mock_model
        mock_mf.return_value = mock_model
        
        yield mock_pm, mock_mf


def test_transcribe_audio_baseline(mock_dependencies, tmp_path: Path):
    mock_pm, mock_mf = mock_dependencies
    
    # Create fake audio file (won't actually be read fully since we mock)
    audio_path = tmp_path / "test.wav"
    audio_path.touch()
    
    # Create mock args
    args = argparse.Namespace(
        model_id="test/model",
        adapter_dir=None,
        processor_dir=None,
        audio=str(audio_path),
        manifest=None,
        output=None,
        device="cpu"
    )
    
    report = transcribe(args)
    
    assert report.model_id == "test/model"
    assert report.adapter_dir is None
    assert report.wer is None  # no text
    assert len(report.samples) == 1
    assert report.samples[0].prediction == "mocked prediction"
    mock_pm.assert_not_called()  # No peft model since adapter_dir is None


def test_transcribe_manifest_with_adapter_and_wer(mock_dependencies, tmp_path: Path):
    mock_pm, mock_mf = mock_dependencies
    
    manifest_path = tmp_path / "test.jsonl"
    manifest_path.write_text(json.dumps({"audio": [0.0], "text": "mocked prediction"}) + "\n")
    output_path = tmp_path / "out.json"
    
    args = argparse.Namespace(
        model_id="test/model",
        adapter_dir="test/adapter",
        processor_dir="test/processor",
        audio=None,
        manifest=str(manifest_path),
        output=str(output_path),
        device="cpu"
    )
    
    report = transcribe(args)
    
    assert report.adapter_dir == "test/adapter"
    assert report.wer == 0.0  # reference matches prediction perfectly!
    assert len(report.samples) == 1
    assert report.samples[0].reference == "mocked prediction"
    assert output_path.exists()
    mock_pm.assert_called_once()
