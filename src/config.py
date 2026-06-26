from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "data.csv"
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.joblib"

# Data
TARGET_LANGUAGE = "nl"

# Model hyperparameters
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
CLASS_WEIGHT = "balanced"
MAX_ITER = 1000
RANDOM_STATE = 42
TEST_SIZE = 0.2
