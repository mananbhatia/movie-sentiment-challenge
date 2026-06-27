from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel

from src.config import MODEL_PATH
from src.model import SentimentModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and clean up on shutdown."""
    app.state.model = SentimentModel.load(MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


class ClassifyRequest(BaseModel):
    review: str


class ClassifyResponse(BaseModel):
    label: str


@app.post("/classify", response_model=ClassifyResponse)
async def classify(body: ClassifyRequest, request: Request) -> ClassifyResponse:
    """Predict the sentiment label for the given review."""
    label = request.app.state.model.predict(body.review)
    return ClassifyResponse(label=label)


@app.get("/health")
async def health() -> dict:
    """Return a simple liveness check."""
    return {"status": "ok"}
