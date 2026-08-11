# Sentiment Analysis с DistilBERT

Проект на неделю: от токенизации до fine-tuning, сравнения моделей, анализа ошибок и демо.

## Описание

Бинарный sentiment analysis на **IMDB Dataset of 50K Movie Reviews** (используем **25%** подвыборку для ускорения обучения на CPU).

- **Baseline:** frozen DistilBERT CLS-эмбеддинги + LogisticRegression  
- **Fine-tuned:** `DistilBERTForSequenceClassification` (3 эпохи)

## Структура

```
README.md
├── requirements.txt
├── app.py                    # Gradio демо
├── data/                     # IMDB Dataset.csv
├── fine_tuned_model/         # дообученная модель
├── baseline_model.pkl
├── baseline_results.txt
├── fine_tuned_results.txt
├── comparison_results.txt
├── error_analysis.txt
├── confusion_matrix_*.png
├── src/
│   ├── embeddings.py         # tokenize + CLS embeddings
│   ├── baseline.py           # Day 4 baseline
│   ├── dataset.py            # SentimentDataset
│   ├── finetune.py           # Day 5 fine-tuning
│   ├── predict.py            # inference helpers
│   ├── compare.py            # Day 6 сравнение
│   ├── error_analysis.py     # Day 7 анализ ошибок
│   ├── main.py               # FastAPI: / и /predict
│   ├── models.py
│   └── utils.py              # heuristic fallback
└── tests/
```

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Пайплайн по дням

```bash
python -m src.baseline --max-samples 2000   # baseline (или полный кэш)
python -m src.finetune                      # fine-tune на 25% IMDB
python -m src.compare                       # сравнение + confusion matrices
python -m src.error_analysis                # FP/FN анализ
```

## Запуск демо

### Gradio

```bash
pip install gradio
python app.py
```

Откройте http://127.0.0.1:7860

### FastAPI

```bash
uvicorn src.main:app --reload
```

- `GET /` — healthcheck  
- `POST /predict` — `{"text": "..."}` → `{"label": "positive"|"negative", "score": 0..1}`  
- Docs: http://127.0.0.1:8000/docs

## Результаты

### Fine-tuned модель
- F1 (macro): **0.8636**
- Accuracy: **0.8636**

### Baseline модель
- F1 (macro): **0.8224**
- Accuracy: **0.8224**

**Улучшение F1: +5.01%**

Подробности: `comparison_results.txt`, `error_analysis.txt`.

## Использование в коде

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

inputs = tokenizer("Your text here", return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    pred = torch.argmax(model(**inputs).logits, dim=1).item()
# 0 = negative, 1 = positive
```

## Требования

- Python 3.8+
- transformers, torch, scikit-learn, pandas
- gradio (для демо)
- fastapi, uvicorn (для API)
