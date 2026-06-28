# Dutch Movie Review Sentiment Classifier

Classifies Dutch-language movie reviews as **Positive**, **Average**, or **Negative**. The solution is a TF-IDF + Logistic Regression model wrapped in a scikit-learn Pipeline, trained with tracked experiments, and served behind a FastAPI `/classify` endpoint. The API can optionally explain its predictions, and the whole service runs in a Docker container.

The dataset is a CSV of movie reviews with two columns, `Reviews` and `Label`. Only Dutch reviews are considered; non-Dutch rows are filtered out during preprocessing, and the API rejects non-Dutch input at request time.

## Setup

A trained model is included, so to run the API you only need the serving dependencies:

​```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
​```

For the full environment — training, experiments, tests, and the notebook — install the dev dependencies instead:

​```bash
pip install -r requirements-dev.txt
​```

There are two requirements files on purpose. `requirements.txt` is the lean set needed to *serve* the model, which keeps the Docker image small. `requirements-dev.txt` is the full environment, including MLflow, NLTK, pytest, and the notebook tooling. The Docker image installs only the lean file.

The dataset is only needed to retrain the model or run the notebook; serving uses the included model. To retrain, place the provided CSV at `data/data.csv` (rename it, or update `DATA_PATH` in `src/config.py`, if your file has a different name). The data folder is gitignored, so the CSV is not committed.

## Usage

All commands are run from the project root.

**Train the model and log experiments (optional)**

A trained model is already included, so you can serve the API or run Docker directly without this step. To retrain:

```bash
python -m src.train
```

This loads the data, runs three tracked experiments, saves the chosen model to `models/sentiment_model.joblib`, and prints a classification report for each run. The first run also downloads the NLTK Dutch stopwords corpus, which needs internet access.

**Serve the API**

```bash
uvicorn src.api:app --reload
```

Then open `http://127.0.0.1:8000/docs` for the interactive Swagger UI, where you can send a review and see the predicted label. The `--reload` flag is for development; drop it to run normally.

**Run with Docker**

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

The container serves the same API. Open `http://127.0.0.1:8000/docs` to use it. The `-p 8000:8000` flag maps the container's port to your host so the service is reachable.

**Run the tests**

```bash
python -m pytest
```

**Benchmark inference latency**

```bash
python benchmark.py
```

**Browse the experiment runs**

The experiments are tracked in a local SQLite database (`mlflow.db`) created by the training run. To view them:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the URL it prints (usually `http://127.0.0.1:5000`) and switch to the "Model training" view to compare runs. If the UI blocks the connection, start the server explicitly:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --allowed-hosts "127.0.0.1:5000,localhost:5000"
```

## Calling the API

`POST /classify` takes a single review and returns a label. Set `explain` to `true` to also receive the words that most influenced the prediction, each with a contribution score. A higher score means the word pushed the prediction more strongly toward the returned label.

Request:

```json
{
  "review": "Deze film was verschrikkelijk saai en teleurstellend.",
  "explain": false
}
```

Response:

```json
{
  "label": "Negative",
  "explanation": null
}
```

If the review is not Dutch, the endpoint returns `400 Bad Request` with a message asking for a Dutch review. There is also a `GET /health` endpoint that returns a simple liveness check.

## Project structure

```
.
├── data/                     # input CSV (gitignored)
├── models/                   # saved model artifact (committed)
├── notebooks/
│   └── exploration.ipynb     # data exploration and modelling experiments
├── src/
│   ├── config.py             # paths and hyperparameters in one place
│   ├── data_loader.py        # load, deduplicate, filter to Dutch, return X and y
│   ├── model.py              # SentimentModel: train, predict, save, load, explain
│   ├── api.py                # FastAPI app exposing /classify and /health
│   └── train.py              # training + experiment tracking + model saving
├── tests/
│   └── test_model.py         # unit tests for the model class
├── benchmark.py              # measures inference latency
├── Dockerfile
├── requirements.txt          # serving dependencies
└── requirements-dev.txt      # full dependencies
```

## Approach and design decisions

**The data and the labels.** The reviews are mixed-language, so the first preprocessing step detects each review's language with `langdetect` and keeps only the Dutch ones. After deduplication and filtering, the three classes are imbalanced: Positive and Average have roughly two thousand examples each, while Negative has under three hundred. Negative is the rare, hard class, and that shaped most of the modelling choices below. Language detection is seeded so the filtering step is reproducible across runs.

**Model.** Reviews are turned into features with TF-IDF and classified with Logistic Regression, bundled together in a single scikit-learn Pipeline. TF-IDF is a sensible, well-understood representation for text classification, and Logistic Regression is a strong linear baseline: it is fast to train, fast at inference, and interpretable, which is what makes the optional explanation feature cheap to add. Bundling the vectorizer and classifier in one Pipeline means the exact same transformation is applied at training and at inference, which avoids train/serve skew and lets the vectorizer and classifier be saved and loaded together as one artifact.

**Metric.** Because the classes are imbalanced, raw accuracy is misleading: a model could score well by doing nothing useful on the rare Negative class. The headline metric is therefore macro-averaged F1, which weights each class equally regardless of size, alongside the full per-class classification report. Logistic Regression is trained with balanced class weights so the rare Negative class is not drowned out during training.

**Experiment tracking.** Training runs three experiments and logs each to MLflow: a unigram baseline, a version with bigrams, and a version with Dutch stop words removed. Bigrams gave a small, consistent improvement (they capture phrases like "zo slecht" and concession constructions that single words miss), so the final model uses them. Removing stop words actually hurt, because Dutch stop word lists include words like "geen", "maar", and "toch" that carry real sentiment signal here. MLflow stores the parameters and metrics for each run so they can be compared side by side. The final model is fully determined by the hyperparameters in `config.py`, so the deployed model is exactly the configuration that won the comparison.

**Serving.** The API is built with FastAPI, which gives request validation through Pydantic and an interactive docs page for free. The trained model is loaded once when the server starts and reused for every request, rather than being loaded per request; loading is the expensive part, so paying that cost once at startup keeps per-request latency low. Input is validated for language before prediction: a non-Dutch review is rejected rather than given a confident but meaningless label, since the model only understands Dutch.

**Latency.** Inference is sub-millisecond at the median, because a linear model over sparse TF-IDF features is just a sparse matrix-vector product. The benchmark reports median, 95th percentile, and mean over a few hundred reviews. This measures the model's inference time, not the full HTTP round trip, which would add a small overhead.

**Explainability.** Because the model is linear, a prediction can be explained directly from the model's own coefficients: each word's contribution is its TF-IDF value multiplied by its coefficient for the predicted class. The API exposes this as an optional field, so a caller who wants to understand *why* a review was classified a certain way can ask for the contributing words, while the default response stays minimal. No additional explainability library is needed.

**Testing.** Two unit tests cover the model class. The first trains the model and checks that `predict` returns one of the three valid labels. The second saves a trained model to disk, reloads it, and checks that the reloaded model gives the same prediction, which verifies that persistence preserves behaviour and protects the train/serve boundary. The tests use a small in-memory fixture, so they run fast and do not depend on the dataset.

## Results

On a stratified held-out test set, the bigram model reaches a macro-F1 of around 0.63. Positive and Average are classified reasonably well; Negative is harder, both because it is rare and because lukewarm and negative reviews share vocabulary. The test set contains only a few dozen Negative examples, so that class's metrics carry more variance than the others.

The experiment comparison and the explanation output both line up with intuition: the words the model leans on for Negative are clearly negative ("slecht", "saai", "verschrikkelijk"), the Positive words are clearly positive ("geweldig", "prachtig", "fantastisch"), and the Average class is characterised by hedging and concession words ("hoewel", "echter", "maar"), which is exactly what makes it the ambiguous middle class.

Removing Dutch stop words raised the Negative class's precision (from 0.53 to 0.60) but lowered its recall more (from 0.67 to 0.59), so the overall F1 dropped. Stop-word lists strip out negation words like "niet" and "geen", which are exactly the words that mark a review as negative. With them gone, the model only flags a review as Negative when a strongly negative word like "slecht" survives, so the Negatives it predicts are more often correct (higher precision), but it misses the ones whose negative signal lived in the negation (lower recall). Since recall fell more than precision rose, removing stop words was a net loss, so the final model keeps them.

## Future work

- **Data and model versioning.** The trained model is committed to the repo so the service runs out of the box, which is fine for a small artifact but not how I would handle it at scale. I would version the dataset and model with DVC (or a model registry) so each trained model is tied to an exact data version, rather than relying on git for binaries.
- **More on the Negative class.** Gather more Negative examples or use stratified cross-validation for a more stable estimate, since the current Negative metrics are based on a small test slice.
- **Faster language validation.** The API's language check adds a few milliseconds per request; a faster detector (such as fastText) or an optional toggle would reduce that cost in a high-throughput setting.
