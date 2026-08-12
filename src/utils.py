"""Heuristic stub — NOT used by FastAPI/Gradio demos.

Kept only as a tiny reference for keyword-based scoring experiments.
Production demos require ./fine_tuned_model from Day 5.
"""

POSITIVE_WORDS = {
    "good",
    "great",
    "amazing",
    "love",
    "excellent",
    "wonderful",
    "best",
    "happy",
    "awesome",
    "perfect",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "hate",
    "worst",
    "horrible",
    "poor",
    "sad",
    "disappointing",
    "waste",
}


def predict_sentiment(text: str) -> tuple[str, float]:
    tokens = {t.strip(".,!?;:\"'").lower() for t in text.split()}
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    if pos == 0 and neg == 0:
        return "positive", 0.5
    total = pos + neg
    if pos >= neg:
        return "positive", pos / total
    return "negative", neg / total
