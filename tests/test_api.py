import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

class TestAPI:
    
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        # Vérifie que le message contient "Prédiction" ou "employés"
        assert "prédiction" in response.json()["message"].lower() or "employés" in response.json()["message"].lower()
    
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_predict_endpoint_valid_text(self):
        response = client.post("/predict", json={"text": "Employé avec 5 ans ancienneté"})
        assert response.status_code == 200
        assert "prediction" in response.json()
    
    def test_predict_endpoint_empty_text(self):
        response = client.post("/predict", json={"text": ""})
        # L'API accepte le texte vide ou retourne 200 avec une prédiction
        assert response.status_code in [200, 422]

if __name__ == "__main__":
    pytest.main(["-v", "tests/"])