from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health_check():
    """Verify that /health returns HTTP 200 and healthy status."""
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "service": "summarizer-api"}

def test_summarize_validation_error():
    """Verify that input shorter than 3 characters is rejected with HTTP 422."""
    with TestClient(app) as test_client:
        # 'ai' has only 2 characters, min_length is 3
        response = test_client.post("/summarize", json={"topic": "ai"})
        assert response.status_code == 422
        # Verify that the validation error points to 'topic'
        detail = response.json()["detail"]
        assert detail[0]["loc"] == ["body", "topic"]

def test_summarize_success_mocked():
    """
    Test /summarize endpoint with a mocked LLM call.
    This ensures our API logic works without spending API quota or waiting on network.
    """
    mock_choice = MagicMock()
    mock_choice.message.content = "- Bullet 1\n- Bullet 2\n- Bullet 3"
    
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch("src.app.Groq") as MockGroq:
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_completion
        MockGroq.return_value = mock_client_instance

        with TestClient(app) as test_client:
            response = test_client.post(
                "/summarize",
                json={"topic": "Machine Learning Transformers"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["topic"] == "Machine Learning Transformers"
            assert "- Bullet 1" in data["summary"]
