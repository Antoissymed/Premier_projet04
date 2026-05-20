import pandas as pd
from preprocessing import preprocess_input
from model import get_model

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


def make_prediction(input_data: dict) -> dict:
    """
    Prend les données brutes d'un employé, applique le prétraitement
    et retourne la prédiction du modèle.

    Args:
        input_data: dictionnaire avec les champs bruts de l'employé

    Returns:
        dictionnaire avec prediction, prediction_label et confidence
    """
    model = get_model()

    # Prétraitement
    processed = preprocess_input(input_data)

    # Création du DataFrame dans le bon ordre de colonnes
    df = pd.DataFrame([processed], columns=FEATURES)

    # Prédiction
    pred = int(model.predict(df)[0])

    # Confiance
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        confidence = float(max(proba))
    else:
        confidence = 0.0

    label = "Va partir" if pred == 1 else "Ne partira pas"

    return {
        "input_data": processed,
        "prediction": pred,
        "prediction_label": label,
        "confidence": confidence
    }