import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# Connexion PostgreSQL
password = quote_plus("postgres123")
engine = create_engine(
    f"postgresql+pg8000://postgres:{password}@localhost:5432/ml_project"
)

# Lire le CSV
df = pd.read_csv("data/dataset_fusionne_complet.csv")
print(f"Dataset chargé : {len(df)} lignes")

# Insérer dans PostgreSQL
df.to_sql("employee_attrition", engine, if_exists="replace", index=False)
print(f"Lignes inserees dans employee_attrition : {len(df)}")