import subprocess
from pathlib import Path

from db.models import Run
from ui.core import db_client


def test_train_page_renders(test_client):
    response = test_client.get("/train")
    assert response.status_code == 200
    assert "Start Training Experiment" in response.text
    assert "max_steps" in response.text


def test_api_train_starts_process_and_polls_logs(test_client, monkeypatch, tmp_path):
    # Mock subprocess.Popen
    called_cmd = []

    class MockPopen:
        def __init__(self, cmd, stdout, stderr):
            nonlocal called_cmd
            called_cmd = cmd
            self.pid = 999

            # Extract output dir from cmd
            out_dir = cmd[cmd.index("--output-dir") + 1]
            log_file = Path(out_dir) / "experiment.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("DUMMY LOG OUTPUT\nLine 2")

    monkeypatch.setattr(subprocess, "Popen", MockPopen)

    # 1. Trigger the training
    response = test_client.post(
        "/api/train",
        data={
            "run_name": "test_tune",
            "max_steps": "50",
            "learning_rate": "1e-4",
            "adapter_args": "--lora-r 32",
        },
    )

    assert response.status_code == 200
    assert "Process PID: 999" in response.text
    assert "every 2s" in response.text

    # Check if the right arguments were passed
    assert "main.py" in called_cmd
    assert "--max-steps" in called_cmd
    assert "50" in called_cmd
    assert "--lora-r" in called_cmd
    assert "32" in called_cmd

    # Get the run ID from the DB
    with db_client.session_scope() as session:
        run = session.query(Run).filter_by(name="test_tune").first()
        run_id = run.id

    # 2. Test the HTMX Polling Endpoint
    poll_response = test_client.get(f"/api/train/status/{run_id}")
    assert poll_response.status_code == 200
    assert "DUMMY LOG OUTPUT" in poll_response.text
    assert "Live Logs" in poll_response.text
