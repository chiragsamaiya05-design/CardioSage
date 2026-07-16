from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "model.pkl"

DATA_PATH = BASE_DIR / "data" / "heart.csv"