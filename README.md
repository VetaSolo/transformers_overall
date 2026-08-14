# Sentiment Analysis с DistilBERT

Недельный проект: токенизация → эмбеддинги → attention → baseline → fine-tuning → сравнение → анализ ошибок + демо.

## Формат данных (как в задании)

CSV минимум с колонками:

```csv
text,label
"This movie was absolutely amazing!",1
"Terrible experience, would not recommend.",0
```

Также поддерживается IMDB Kaggle: `review,sentiment` (positive/negative).

Положите файл в `data/dataset.csv` **или** `data/IMDB Dataset.csv`, либо передайте `--data path/to.csv`.

Пример официального формата: `data/sample_assignment.csv`.

## Структура

```
app.py                      # Gradio демо (требует fine_tuned_model/)
baseline_model.pkl          # Day 4 — сохраняется baseline.py
baseline_results.txt
fine_tuned_results.txt
comparison_results.txt
error_analysis.txt
fine_tuned_model/           # Day 5
data/
  sample_assignment.csv     # text,label
  IMDB Dataset.csv          # опционально
src/
  data_loading.py           # единый loader text/label (+ IMDB aliases)
  embeddings.py             # Days 1–2 helpers
  baseline.py               # Day 4 (+ сохраняет baseline_model.pkl)
  dataset.py / finetune.py  # Day 5
  predict.py / compare.py   # Day 6 (грузит baseline ТОЛЬКО из Day 4)
  error_analysis.py         # Day 7
  main.py                   # FastAPI на fine-tuned (без heuristic fallback)
tests/
```

## Запуск по дням

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Day 4 — baseline (пишет baseline_results.txt + baseline_model.pkl)
python -m src.baseline --data data/sample_assignment.csv
# или на IMDB (25% по умолчанию):
python -m src.baseline

# Day 5 — fine-tuning
python -m src.finetune

# Day 6 — сравнение (нужны fine_tuned_model/ и baseline_model.pkl)
python -m src.compare

# Day 7 — ошибки
python -m src.error_analysis
```

## Демо

### Gradio
```bash
python app.py
```
http://127.0.0.1:7860

### FastAPI
```bash
uvicorn src.main:app --reload
```
`GET /`, `POST /predict` — **только** fine-tuned модель (если её нет → HTTP 503, не stub).

## Результаты (текущий прогон на 25% IMDB)

### Fine-tuned
- F1 (macro): **0.8636**
- Accuracy: **0.8636**

### Baseline
- F1 (macro): **0.8224**
- Accuracy: **0.8224**

Улучшение F1: **+5.01%**

Веса модели: `fine_tuned_model/model.safetensors` (в git через **Git LFS**).  
Confusion matrices: `confusion_matrix_finetuned.png`, `confusion_matrix_baseline.png`.

После клона:
```bash
git lfs install
git lfs pull
```

## Использование в коде

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

inputs = tokenizer("Your text here", return_tensors="pt", truncation=True, max_length=128)
pred = torch.argmax(model(**inputs).logits, dim=1).item()  # 0=neg, 1=pos
```
