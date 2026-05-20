import pandas as pd

STATUT_MAP = {"Célibataire": 0, "Marié(e)": 1, "Divorcé(e)": 2}
DEPARTEMENT_MAP = {"Commercial": 0, "Consulting": 1, "Ressources Humaines": 2}
FREQ_MAP = {"Rare": 0, "Occasionnel": 1, "Frequent": 2}
POSTE_MAP = {"Cadre Commercial": 0, "Assistant de Direction": 1}
DOMAINE_MAP = {"Infra & Cloud": 0}

def preprocess_input(data: dict) -> dict:
    """
    Transforme les données brutes d'entrée en valeurs numériques
    attendues par le modèle de machine learning.
    """
    data = data.copy()

    data["eval_number"] = int(str(data["eval_number"]).replace("E_", ""))
    data["heure_supplementaires"] = 1 if data["heure_supplementaires"] == "Oui" else 0
    data["ayant_enfants"] = 1 if data["ayant_enfants"] in ["Y", "Oui"] else 0
    data["augementation_salaire_precedente"] = float(
        str(data["augementation_salaire_precedente"]).replace("%", "").strip()
    )
    data["genre"] = 1 if data["genre"] == "M" else 0
    data["statut_marital"] = STATUT_MAP.get(data["statut_marital"], 0)
    data["departement"] = DEPARTEMENT_MAP.get(data["departement"], 0)
    data["frequence_deplacement"] = FREQ_MAP.get(data["frequence_deplacement"], 0)
    data["poste"] = POSTE_MAP.get(data["poste"], 0)
    data["domaine_etude"] = DOMAINE_MAP.get(data["domaine_etude"], 0)

    return data


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le prétraitement sur un DataFrame complet (ex: dataset CSV).
    """
    df = df.copy()

    if "eval_number" in df.columns:
        df["eval_number"] = df["eval_number"].astype(str).str.replace("E_", "").astype(int)
    if "heure_supplementaires" in df.columns:
        df["heure_supplementaires"] = df["heure_supplementaires"].apply(lambda x: 1 if x == "Oui" else 0)
    if "ayant_enfants" in df.columns:
        df["ayant_enfants"] = df["ayant_enfants"].apply(lambda x: 1 if x in ["Y", "Oui"] else 0)
    if "augementation_salaire_precedente" in df.columns:
        df["augementation_salaire_precedente"] = df["augementation_salaire_precedente"].astype(str).str.replace("%", "").str.strip().astype(float)
    if "genre" in df.columns:
        df["genre"] = df["genre"].apply(lambda x: 1 if x == "M" else 0)
    if "statut_marital" in df.columns:
        df["statut_marital"] = df["statut_marital"].map(STATUT_MAP).fillna(0).astype(int)
    if "departement" in df.columns:
        df["departement"] = df["departement"].map(DEPARTEMENT_MAP).fillna(0).astype(int)
    if "frequence_deplacement" in df.columns:
        df["frequence_deplacement"] = df["frequence_deplacement"].map(FREQ_MAP).fillna(0).astype(int)
    if "poste" in df.columns:
        df["poste"] = df["poste"].map(POSTE_MAP).fillna(0).astype(int)
    if "domaine_etude" in df.columns:
        df["domaine_etude"] = df["domaine_etude"].map(DOMAINE_MAP).fillna(0).astype(int)

    return df