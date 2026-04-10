# SMS Spam Classification Pipeline

An ensemble ML pipeline for SMS spam detection using **3 calibrated scikit-learn models** with soft voting, served via **FastAPI** with **PostgreSQL** persistence.

## Architecture

```
SMS Text → TF-IDF (1000 features) → Calibrated Ensemble → ham/spam
                                        ├── GaussianNB
                                        ├── LogisticRegression
                                        └── SVC (RBF kernel)
```

Each base model is wrapped with `CalibratedClassifierCV(cv=5)` for reliable probability estimates, then combined via `VotingClassifier(voting='soft')`.

## Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 98.21% |
| Precision | 98.51% |
| Recall    | 88.00% |
| F1-score  | 92.96% |

## Dataset

**SMS Spam Collection** — 5,572 SMS messages (4,825 ham / 747 spam) located in `Dataset/spam.csv`.

## Quick Start

### 1. Set up environment

```bash
uv venv .venv
source .venv/bin/activate
uv pip install scikit-learn pandas numpy joblib fastapi uvicorn sqlalchemy psycopg2-binary python-dotenv
```

### 2. Train the model

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Extract TF-IDF features (max 1000)
- Train 3 calibrated models + ensemble
- Evaluate on 20% test split
- Save `ensemble.pkl` and `vectorizer.pkl`
- Run verification predictions

### 3. Evaluate the model

```bash
python evaluate.py
```

This will show full test-set metrics (accuracy, precision, recall, F1, ROC-AUC), confusion matrix, confidence stats, misclassified samples, batch predictions on 8 example texts, and an **interactive mode** where you can type any message to classify it.

### 4. Configure PostgreSQL

Edit `.env` with your database credentials:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/spam_db
```

### 5. Run the API

```bash
uvicorn app:app --reload
```

API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify a message → saved to Postgres |
| `/predictions` | GET | List past predictions (filter by `?label=spam`) |
| `/predictions/stats` | GET | Spam/ham counts |
| `/health` | GET | Health check |

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "You won a free iPhone!"}'
```

### 6. Use the saved model directly

```python
import re
import joblib

model = joblib.load("ensemble.pkl")
vectorizer = joblib.load("vectorizer.pkl")

text = "You won a free prize! Click here now!"
cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
features = vectorizer.transform([cleaned]).toarray()

prediction = model.predict(features)[0]        # 0=ham, 1=spam
confidence = model.predict_proba(features)[0].max()

print(f"{'spam' if prediction else 'ham'} ({confidence:.1%})")
```

## Project Structure

```
mlops/
├── Dataset/
│   └── spam.csv          # SMS Spam Collection
├── train.py              # Training pipeline
├── evaluate.py           # Evaluation & interactive inference
├── app.py                # FastAPI application
├── database.py           # SQLAlchemy models & DB config
├── .env                  # PostgreSQL credentials (not committed)
├── ensemble.pkl          # Saved ensemble model (generated)
├── vectorizer.pkl        # Saved TF-IDF vectorizer (generated)
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10+
- scikit-learn, pandas, numpy, joblib
- CPU only — no GPU required
