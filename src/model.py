import joblib
import os
from typing import Optional

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modele_final_optimise.pkl")

_model = None


def load_model(path: str = MODEL_PATH):
    """
    Charge le modèle depuis le fichier .pkl et le met en cache.
    """
    global _model
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    _model = joblib.load(path)
    print(f"✅ Modèle chargé depuis : {path}")
    return _model


def get_model():
    """
    Retourne le modèle en cache. Le charge si ce n'est pas déjà fait.
    """
    global _model
    if _model is None:
        _model = load_model()
    return _model


def get_model_info() -> dict:
    """
    Retourne des informations sur le modèle chargé.
    """
    model = get_model()
    return {
        "type": type(model).__name__,
        "loaded": model is not None,
        "path": MODEL_PATH,
        "features_count": len(model.feature_names_in_) if hasattr(model, "feature_names_in_") else "N/A"
    }