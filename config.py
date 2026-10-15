from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "phishy.joblib"

SEED = 42
DEFAULT_THRESHOLD = 0.5
