import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import MAX_FEATURES, NGRAM_RANGE, CLASS_WEIGHT, MAX_ITER, RANDOM_STATE, STOP_WORDS


class SentimentModel:
    """Sentiment classifier wrapping a TF-IDF + Logistic Regression pipeline."""

    def __init__(self) -> None:
        """Build the sklearn pipeline."""
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, stop_words=STOP_WORDS)),
            ("clf", LogisticRegression(class_weight=CLASS_WEIGHT, max_iter=MAX_ITER, random_state=RANDOM_STATE)),
        ])

    def train(self, X, y) -> None:
        """Fit the pipeline on X and y."""
        self.pipeline.fit(X, y)

    def predict(self, review: str) -> str:
        """Return the predicted label for a single review string."""
        return str(self.pipeline.predict([review])[0])

    def save(self, path) -> None:
        """Serialize the pipeline to path, creating parent directories as needed."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    def explain(self, review: str, top_n: int = 10) -> list[tuple[str, float]]:
        """Return the top_n (word, contribution) tuples for words present in the review, sorted by contribution descending."""
        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]

        tfidf_matrix = tfidf.transform([review])
        predicted_label = clf.predict(tfidf_matrix)[0]
        class_index = list(clf.classes_).index(predicted_label)

        contributions = tfidf_matrix.multiply(clf.coef_[class_index]).toarray()[0]

        feature_names = np.array(tfidf.get_feature_names_out())
        present = tfidf_matrix.nonzero()[1]

        ranked = sorted(
            ((feature_names[i], contributions[i]) for i in present),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_n]

    @classmethod
    def load(cls, path) -> "SentimentModel":
        """Load a pipeline from path and return a ready SentimentModel."""
        model = cls()
        model.pipeline = joblib.load(path)
        return model
