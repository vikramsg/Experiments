def test_dashboard_renders(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert "Dashboard" in response.text
    assert "LoRA Studio" in response.text
