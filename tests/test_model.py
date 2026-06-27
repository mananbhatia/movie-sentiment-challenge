import pytest

from src.model import SentimentModel

_LABELS = {"Positive", "Average", "Negative"}

_REVIEWS = [
    ("geweldig prachtig fantastisch geweldig film", "Positive"),
    ("uitstekend briljant geweldig meesterwerk schitterend", "Positive"),
    ("mooi aangenaam positief boeiend vermakelijk", "Positive"),
    ("redelijk gemiddeld acceptabel oké middelmatig", "Average"),
    ("gewoon normaal matig onopvallend middelmatig", "Average"),
    ("doorsnee redelijk zo-zo neutraal middelmatig", "Average"),
    ("verschrikkelijk slecht afschuwelijk teleurstellend saai", "Negative"),
    ("vreselijk slecht waardeloos verschrikkelijk afgrijselijk", "Negative"),
    ("boring vervelend slecht zwak teleurstellend", "Negative"),
]


@pytest.fixture
def trained_model() -> SentimentModel:
    """Return a SentimentModel trained on a small in-memory dataset."""
    reviews = [r for r, _ in _REVIEWS]
    labels = [l for _, l in _REVIEWS]
    model = SentimentModel()
    model.train(reviews, labels)
    return model


def test_predict_returns_valid_label(trained_model: SentimentModel) -> None:
    """Prediction output must be one of the three known class labels."""
    result = trained_model.predict("geweldig film")
    assert result in _LABELS


def test_save_load_roundtrip(trained_model: SentimentModel, tmp_path) -> None:
    """A model saved to disk and reloaded must produce identical predictions."""
    path = tmp_path / "models" / "model.joblib"
    review = "verschrikkelijk slechte film"

    trained_model.save(path)
    loaded = SentimentModel.load(path)

    assert loaded.predict(review) == trained_model.predict(review)
