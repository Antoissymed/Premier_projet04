from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import os

# Charger le .env EN PREMIER
load_dotenv()

# Lire les variables depuis .env
DB_PASSWORD = os.getenv('MOTDEPASSE')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'ml_project')

# Construire l'URL avec les variables
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String, nullable=False)
    prediction = Column(Integer, nullable=False)
    prediction_label = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Initialise la base de données PostgreSQL"""
    Base.metadata.create_all(bind=engine)
    print("✅ Base de données PostgreSQL créée avec succès!")


def save_prediction(input_text: str, prediction: int, label: str, confidence: float):
    """Enregistre une prédiction dans PostgreSQL"""
    db = SessionLocal()
    try:
        record = Prediction(
            input_text=input_text,
            prediction=prediction,
            prediction_label=label,
            confidence=confidence
        )
        db.add(record)
        db.commit()
        print(f"✅ Prédiction enregistrée dans PostgreSQL (id: {record.id})")
        return record.id
    except Exception as e:
        print(f"❌ Erreur PostgreSQL: {e}")
        db.rollback()
        return None
    finally:
        db.close()


if __name__ == "__main__":
    init_db()