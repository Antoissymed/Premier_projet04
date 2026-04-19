from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import joblib
import os
import pandas as pd
import numpy as np

# Créer l'application FastAPI
app = FastAPI(
    title="API de Prédiction - Départ des employés",
    description="API pour prédire si un employé va quitter l'entreprise",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Modèle pour la requête
class EmployeeInput(BaseModel):
    anciennete: Optional[float] = Field(None, description="Ancienneté en années")
    satisfaction: Optional[float] = Field(None, description="Satisfaction (0-10)")
    note_evaluation: Optional[float] = Field(None, description="Note d'évaluation (0-10)")
    projet_moyen: Optional[float] = Field(None, description="Nombre moyen de projets")
    heures_mensuelles: Optional[float] = Field(None, description="Heures travaillées par mois")
    accident_travail: Optional[int] = Field(None, description="Nombre d'accidents")
    promotion_5ans: Optional[int] = Field(None, description="Nombre de promotions en 5 ans")
    departement: Optional[str] = Field(None, description="Département")
    salaire: Optional[float] = Field(None, description="Salaire")

# Modèle simplifié pour la prédiction (si vous voulez garder le format texte)
class TextInput(BaseModel):
    text: str = Field(..., description="Description de l'employé")

# Modèle pour la réponse
class PredictionOutput(BaseModel):
    input_data: Dict[str, Any]
    prediction: int
    prediction_label: str
    confidence: float

# Variables globales
model = None

def load_model():
    global model
    model_path = "models/meilleur_modele_rf.pkl"
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"✅ Modèle chargé depuis {model_path}")
        print(f"   Type: {type(model).__name__}")
    else:
        print(f"❌ Modèle non trouvé: {model_path}")
        print("   L'API fonctionnera en mode démo")

@app.on_event("startup")
async def startup_event():
    load_model()
    print("🚀 API de prédiction démarrée!")

@app.get("/")
def root():
    return {
        "message": "API de prédiction - Départ des employés",
        "status": "online",
        "model_loaded": model is not None,
        "endpoints": ["/health", "/predict", "/docs"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": str(type(model).__name__) if model else None
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TextInput):
    try:
        # Mode démo si le modèle n'est pas chargé
        if model is None:
            return PredictionOutput(
                input_data={"text": input_data.text},
                prediction=0,
                prediction_label="Démo - Ne partira pas",
                confidence=0.95
            )
        
        # TODO: Transformer le texte en features numériques
        # Pour l'instant, prédiction factice
        # À remplacer par votre vrai preprocessing
        
        # Exemple de prédiction avec le modèle
        # features = preprocess_text_to_features(input_data.text)
        # prediction = model.predict([features])[0]
        # probabilities = model.predict_proba([features])[0]
        # confidence = float(max(probabilities))
        
        # Version temporaire
        prediction = 0
        confidence = 0.90
        prediction_label = "Ne partira pas" if prediction == 0 else "Va partir"
        
        return PredictionOutput(
            input_data={"text": input_data.text},
            prediction=int(prediction),
            prediction_label=prediction_label,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)