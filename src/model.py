import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import MAX_FEATURES, NGRAM_RANGE, CLASS_WEIGHT, MAX_ITER, RANDOM_STATE


class SentimentModel:
    """Sentiment classifier wrapping a TF-IDF + Logistic Regression pipeline."""

    def __init__(self) -> None:
        """Build the sklearn pipeline."""
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)),
            ("clf", LogisticRegression(class_weight=CLASS_WEIGHT, max_iter=MAX_ITER, random_state=RANDOM_STATE)),
        ])

    def train(self, X, y) -> None:
        """Fit the pipeline on X and y."""
        self.pipeline.fit(X, y)

    def predict(self, review: str) -> str:
        """Return the predicted label for a single review string."""
        return self.pipeline.predict([review])[0]

    def save(self, path) -> None:
        """Serialize the pipeline to path, creating parent directories as needed."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path) -> "SentimentModel":
        """Load a pipeline from path and return a ready SentimentModel."""
        model = cls()
        model.pipeline = joblib.load(path)
        return model
