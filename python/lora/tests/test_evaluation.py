from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from lora_training.evaluation import eval_wer, normalize_text, summarize_losses


def test_summarize_losses_handles_empty() -> None:
    with pytest.raises(ValueError, match="Cannot summarize empty list of losses"):
        summarize_losses([])


def test_summarize_losses_returns_mean() -> None:
    assert summarize_losses([1.0, 2.0, 3.0]) == 2.0


def test_normalize_text() -> None:
    assert normalize_text("Hello, World!") == "hello world"
    assert normalize_text("  Testing... 123  ") == "testing 123"
    assert normalize_text("UPPERCASE") == "uppercase"
    assert normalize_text("Hyphen-ated word.") == "hyphenated word"


@patch("lora_training.evaluation.load")
def test_eval_wer_respects_max_batches(mock_load) -> None:
    # Setup mock metric
    mock_metric = MagicMock()
    mock_metric.compute.return_value = 0.5
    mock_load.return_value = mock_metric

    # Setup mock model
    model = MagicMock()
    model.config.model_type = "moonshine"
    model.parameters.return_value = iter([torch.randn(1, dtype=torch.float32)])
    model.generate.return_value = [[1, 2, 3]]

    # Setup mock processor
    processor = MagicMock()
    processor.feature_extractor.sampling_rate = 16000
    processor.decode.return_value = "pred"
    processor.tokenizer.pad_token_id = 0
    processor.tokenizer.batch_decode.return_value = ["ref"]

    # Setup dataloader with 5 batches
    batch = {
        "input_values": torch.randn(1, 10),
        "attention_mask": torch.ones(1, 10),
        "labels": torch.zeros(1, 1, dtype=torch.long)
    }
    dataloader = [batch] * 5

    device = torch.device("cpu")
    
    # Test max_batches=2
    wer = eval_wer(model, processor, dataloader, device, max_batches=2)
    
    assert wer == 0.5
    # verify that it only processed 2 batches
    assert processor.decode.call_count == 2
