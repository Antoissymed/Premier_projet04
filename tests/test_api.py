import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

class TestAPI:
    
    def test_root_endpoint(self):
        """Test GET / endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "status" in response.json()
    
    def test_health_endpoint(self):
        """Test GET /health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_predict_endpoint_valid_data(self):
        """Test POST /predict with valid employee data (31 parameters)"""
        data = {
            "satisfaction_employee_environnement": 3,
            "note_evaluation_precedente": 3,
            "niveau_hierarchique_poste": 2,
            "satisfaction_employee_nature_travail": 4,
            "satisfaction_employee_equipe": 3,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "eval_number": "E_1",
            "note_evaluation_actuelle": 3,
            "heure_supplementaires": "Non",
            "augementation_salaire_precedente": "5%",
            "age": 35,
            "genre": "M",
            "revenu_mensuel": 4500,
            "statut_marital": "Célibataire",
            "departement": "Consulting",
            "poste": "Cadre Commercial",
            "nombre_experiences_precedentes": 3,
            "nombre_heures_travailless": 160,
            "annee_experience_totale": 8,
            "annees_dans_l_entreprise": 5,
            "annees_dans_le_poste_actuel": 2,
            "nombre_participation_pee": 1,
            "nb_formations_suivies": 2,
            "nombre_employee_sous_responsabilite": 0,
            "distance_domicile_travail": 10,
            "niveau_education": 4,
            "domaine_etude": "Infra & Cloud",
            "ayant_enfants": "Non",
            "frequence_deplacement": "Occasionnel",
            "annees_depuis_la_derniere_promotion": 2,
            "annes_sous_responsable_actuel": 3
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 200
        assert "prediction" in response.json()
        assert "prediction_label" in response.json()
        assert "confidence" in response.json()
    
    def test_predict_endpoint_missing_field(self):
        """Test POST /predict with missing required field (should fail)"""
        data = {
            "satisfaction_employee_environnement": 3,
            "note_evaluation_precedente": 3
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 422  # Validation error

if __name__ == "__main__":
    pytest.main(["-v", "tests/"])


