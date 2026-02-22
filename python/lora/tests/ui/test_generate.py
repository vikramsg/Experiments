import multiprocessing


def test_generate_page_renders(test_client):
    response = test_client.get("/generate")
    assert response.status_code == 200
    assert "Synthetic Data Generation" in response.text
    assert "num_samples" in response.text


def test_api_generate_triggers_background_process(test_client, monkeypatch):
    called = False

    class MockProcess:
        def __init__(self, target, args):
            nonlocal called
            called = True
            self.pid = 12345

        def start(self):
            pass

    monkeypatch.setattr(multiprocessing, "Process", MockProcess)

    response = test_client.post(
        "/api/generate",
        data={
            "num_samples": 5,
            "dataset_name": "test_synth",
            "audio_prefix": "synth",
            "mix_with_real": "false",
        },
    )

    assert response.status_code == 200
    assert "Generation Started!" in response.text
    assert "12345" in response.text
    assert called is True
