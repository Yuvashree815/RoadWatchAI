from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    """
    Test the /health endpoint to ensure the backend is running.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["message"] == "RoadWatch AI backend is running."
    assert "version" in data
