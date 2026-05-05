#  Projet Machine Learning - Prédiction du départ des employés

##  Objectif
Prédire si un employé va quitter l’entreprise à partir de données RH via une API FastAPI.

---

##  Données
Les données proviennent de plusieurs sources RH :
- dataset_fusionne_complet.csv
- extrait_eval.csv
- extrait_sirh.csv
- extrait_sondage.csv

---

##  Modèle
- Algorithme : Random Forest
- Optimisation : GridSearchCV
- Fichier du modèle : models/modele_final_optimise.pkl

---

##  API FastAPI

### ▶️ Lancer l’API
```bash
uvicorn api.main:app --reload --port 8001


http://127.0.0.1:8001/docs

{
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

{
  "prediction": 0,
  "prediction_label": "Ne partira pas",
  "confidence": 0.55
}


Premier_projet04/
├── api/          # API FastAPI
├── data/         # Données
├── models/       # Modèle entraîné
├── notebooks/    # Analyse et modélisation
├── src/          # Code
├── tests/        # Tests pytest

git clone https://github.com/Antoissymed/Premier_projet04.git
cd Premier_projet04

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

pytest -v

Technologies utilisées

Python	Langage principal
FastAPI	Framework API
Scikit-learn	Modèle Random Forest
Pandas	Manipulation des données
PostgreSQL	Base de données
SQLAlchemy	ORM
Pytest	Tests unitaires



Antoissymed - Projet de déploiement Machine Learning

Formation OpenClassrooms

