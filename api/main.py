from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import os
import pandas as pd

app = FastAPI(
    title="API de Prédiction - Départ des employés",
    description="API pour prédire si un employé va quitter l'entreprise",
    version="1.0.0"
)

MODEL_PATH = "models/modele_final_optimise.pkl"

FEATURES = [
    "satisfaction_employee_environnement",
    "note_evaluation_precedente",
    "niveau_hierarchique_poste",
    "satisfaction_employee_nature_travail",
    "satisfaction_employee_equipe",
    "satisfaction_employee_equilibre_pro_perso",
    "eval_number",
    "note_evaluation_actuelle",
    "heure_supplementaires",
    "augementation_salaire_precedente",
    "age",
    "genre",
    "revenu_mensuel",
    "statut_marital",
    "departement",
    "poste",
    "nombre_experiences_precedentes",
    "nombre_heures_travailless",
    "annee_experience_totale",
    "annees_dans_l_entreprise",
    "annees_dans_le_poste_actuel",
    "nombre_participation_pee",
    "nb_formations_suivies",
    "nombre_employee_sous_responsabilite",
    "distance_domicile_travail",
    "niveau_education",
    "domaine_etude",
    "ayant_enfants",
    "frequence_deplacement",
    "annees_depuis_la_derniere_promotion",
    "annes_sous_responsable_actuel"
]

class EmployeeInput(BaseModel):
    satisfaction_employee_environnement: int
    note_evaluation_precedente: int
    niveau_hierarchique_poste: int
    satisfaction_employee_nature_travail: int
    satisfaction_employee_equipe: int
    satisfaction_employee_equilibre_pro_perso: int
    eval_number: str
    note_evaluation_actuelle: int
    heure_supplementaires: str
    augementation_salaire_precedente: str
    age: int
    genre: str
    revenu_mensuel: float
    statut_marital: str
    departement: str
    poste: str
    nombre_experiences_precedentes: int
    nombre_heures_travailless: int
    annee_experience_totale: int
    annees_dans_l_entreprise: int
    annees_dans_le_poste_actuel: int
    nombre_participation_pee: int
    nb_formations_suivies: int
    nombre_employee_sous_responsabilite: int
    distance_domicile_travail: int
    niveau_education: int
    domaine_etude: str
    ayant_enfants: str
    frequence_deplacement: str
    annees_depuis_la_derniere_promotion: int
    annes_sous_responsable_actuel: int

class PredictionOutput(BaseModel):
    input_data: Dict[str, Any]
    prediction: int
    prediction_label: str
    confidence: float

model = None

def preprocess_input(data):
    data["eval_number"] = int(data["eval_number"].replace("E_", ""))
    data["heure_supplementaires"] = 1 if data["heure_supplementaires"] == "Oui" else 0
    data["ayant_enfants"] = 1 if data["ayant_enfants"] in ["Y", "Oui"] else 0
    data["augementation_salaire_precedente"] = float(
        data["augementation_salaire_precedente"].replace("%", "").strip()
    )
    data["genre"] = 1 if data["genre"] == "M" else 0
    statut_map = {"Célibataire": 0, "Marié(e)": 1, "Divorcé(e)": 2}
    data["statut_marital"] = statut_map.get(data["statut_marital"], 0)
    departement_map = {"Commercial": 0, "Consulting": 1, "Ressources Humaines": 2}
    data["departement"] = departement_map.get(data["departement"], 0)
    freq_map = {"Rare": 0, "Occasionnel": 1, "Frequent": 2}
    data["frequence_deplacement"] = freq_map.get(data["frequence_deplacement"], 0)
    poste_map = {"Cadre Commercial": 0, "Assistant de Direction": 1}
    data["poste"] = poste_map.get(data["poste"], 0)
    domaine_map = {"Infra & Cloud": 0}
    data["domaine_etude"] = domaine_map.get(data["domaine_etude"], 0)
    return data

@app.on_event("startup")
async def startup_event():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Modèle chargé")
    else:
        print("Modèle introuvable")

@app.get("/")
def root():
    return {
        "message": "API de prédiction - Départ des employés",
        "status": "online",
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: EmployeeInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    try:
        data = preprocess_input(input_data.dict())
        df = pd.DataFrame([data], columns=FEATURES)
        pred = model.predict(df)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            conf = float(max(proba))
        else:
            conf = 0.0
        label = "Va partir" if int(pred) == 1 else "Ne partira pas"
        return PredictionOutput(
            input_data=data,
            prediction=int(pred),
            prediction_label=label,
            confidence=conf
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)