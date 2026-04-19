# Projet Machine Learning - Prédiction du départ des employés

## Objectif
Prédire si un employé va quitter l’entreprise à partir de données RH via une API FastAPI.

## Données
- dataset_fusionne_complet.csv
- extrait_eval.csv
- extrait_sirh.csv
- extrait_sondage.csv

## Modèle
Random Forest optimisé (fichier .pkl dans models/)

## API
Lancer l’API :
uvicorn api.main:app --reload


Accès :
http://127.0.0.1:8000/docs

## Structure
Premier_projet/
├── api/
├── data/
├── models/
├── notebooks/
├── src/
├── tests/

## Installation
git clone https://github.com/Antoissymed/Premier_projet04.git
cd Premier_projet04
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Tests
pytest