import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os

# Charger le .env
load_dotenv()

# Lire les variables depuis .env
DB_PASSWORD = quote_plus(os.getenv('MOTDEPASSE', ''))
DB_USER = os.getenv('DB_USER', 'postgres')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'ml_project')

# Connexion PostgreSQL
engine = create_engine(
    f"postgresql+pg8000://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Lire le CSV
df = pd.read_csv("data/dataset_fusionne_complet.csv")
print(f"Dataset chargé : {len(df)} lignes")

# Insérer dans PostgreSQL
df.to_sql("employee_attrition", engine, if_exists="replace", index=False)
print(f"Lignes inserees dans employee_attrition : {len(df)}")