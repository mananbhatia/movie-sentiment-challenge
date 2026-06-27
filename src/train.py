import mlflow
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.data_loader import get_data
from src.model import SentimentModel

try:
    stopwords.words("dutch")
except LookupError:
    nltk.download("stopwords")

_DUTCH_STOPWORDS = stopwords.words("dutch")


def run_experiment(
    name: str,
    ngram_range: tuple,
    stop_words,
    X_train,
    X_test,
    y_train,
    y_test,
) -> None:
    """Train a TF-IDF + LR pipeline, log params/metrics to MLflow, and print a classification report."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=ngram_range, stop_words=stop_words)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    per_class_f1 = f1_score(y_test, y_pred, average=None, labels=pipeline.classes_)

    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "ngram_range": str(ngram_range),
            "stop_words": "dutch" if stop_words is not None else "none",
        })
        mlflow.log_metrics({
            "macro_f1": macro_f1,
            "accuracy": acc,
            **{f"f1_{cls}": score for cls, score in zip(pipeline.classes_, per_class_f1)},
        })

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))


def main() -> None:
    """Run three experiments, then train and save the final SentimentModel."""
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    run_experiment("baseline_unigrams", ngram_range=(1, 1), stop_words=None,
                   X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    run_experiment("bigrams", ngram_range=(1, 2), stop_words=None,
                   X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    run_experiment("dutch_stopwords", ngram_range=(1, 1), stop_words=_DUTCH_STOPWORDS,
                   X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    model = SentimentModel()
    model.train(X_train, y_train)
    model.save(MODEL_PATH)
    print(f"\nFinal model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
