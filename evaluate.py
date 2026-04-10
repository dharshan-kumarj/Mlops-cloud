"""
Spam Classification — Evaluation & Inference Script
=====================================================
Loads the saved ensemble model and vectorizer, then:
1. Evaluates on the full test set (re-split from dataset)
2. Shows detailed metrics, confusion matrix, and per-class stats
3. Runs interactive inference on custom text inputs
"""

import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATASET_PATH = os.path.join("Dataset", "spam.csv")
VECTORIZER_PATH = "./vectorizer.pkl"
ENSEMBLE_PATH = "./ensemble.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.20
LABELS = {0: "ham", 1: "spam"}


def clean_text(text: str) -> str:
    """Apply the same preprocessing used during training."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def load_artifacts():
    """Load the saved model and vectorizer."""
    for path, name in [(ENSEMBLE_PATH, "Ensemble model"), (VECTORIZER_PATH, "Vectorizer")]:
        if not os.path.isfile(path):
            sys.exit(f"ERROR: {name} not found at '{path}'. Run train.py first.")

    model = joblib.load(ENSEMBLE_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"Loaded model    : {ENSEMBLE_PATH} ({os.path.getsize(ENSEMBLE_PATH) / 1e6:.1f} MB)")
    print(f"Loaded vectorizer: {VECTORIZER_PATH}")
    return model, vectorizer


def load_test_data():
    """Reload and split the dataset identically to training."""
    if not os.path.isfile(DATASET_PATH):
        sys.exit(f"ERROR: Dataset not found at '{DATASET_PATH}'")

    df = pd.read_csv(DATASET_PATH, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df.dropna(subset=["message"], inplace=True)
    df["message"] = df["message"].apply(clean_text)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    _, X_test_text, _, y_test = train_test_split(
        df["message"], df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return X_test_text, y_test


def evaluate_model(model, vectorizer, X_test_text, y_test):
    """Run full evaluation on the test set."""
    X_test = vectorizer.transform(X_test_text).toarray()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)

    # Core metrics
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1-score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
    }
    print("\n  Metrics:")
    for name, val in metrics.items():
        print(f"    {name:<10}: {val:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"                ham   spam")
    print(f"    Actual ham  [{tn:>4}  {fp:>4}]")
    print(f"          spam  [{fn:>4}  {tp:>4}]")

    # Derived stats
    print(f"\n  Derived Stats:")
    print(f"    True Positives  (spam→spam): {tp}")
    print(f"    True Negatives  (ham→ham) : {tn}")
    print(f"    False Positives (ham→spam): {fp}")
    print(f"    False Negatives (spam→ham): {fn}")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"    Specificity     : {specificity:.4f}")

    # Full classification report
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"])
    for line in report.split("\n"):
        print(f"    {line}")

    # Confidence distribution
    print(f"\n  Confidence Distribution:")
    confidences = np.max(model.predict_proba(X_test), axis=1) * 100
    print(f"    Mean : {confidences.mean():.1f}%")
    print(f"    Min  : {confidences.min():.1f}%")
    print(f"    Max  : {confidences.max():.1f}%")
    print(f"    Std  : {confidences.std():.1f}%")

    # Misclassified samples
    misclassified = np.where(y_pred != y_test.values)[0]
    print(f"\n  Misclassified: {len(misclassified)} / {len(y_test)} samples")
    if len(misclassified) > 0:
        print(f"  First 5 misclassified:")
        for i, idx in enumerate(misclassified[:5]):
            true_label = LABELS[y_test.iloc[idx]]
            pred_label = LABELS[y_pred[idx]]
            conf = np.max(model.predict_proba(X_test[idx].reshape(1, -1))) * 100
            text_preview = X_test_text.iloc[idx][:60]
            print(f"    [{i+1}] True={true_label:<4} Pred={pred_label:<4} "
                  f"Conf={conf:.1f}%  \"{text_preview}...\"")

    print("\n" + "=" * 55)
    return metrics


def predict_text(model, vectorizer, text: str):
    """Predict a single text and return label + confidence."""
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned]).toarray()
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = proba.max() * 100
    return LABELS[pred], confidence, proba


def run_sample_predictions(model, vectorizer):
    """Run predictions on a set of example texts."""
    samples = [
        "Congratulations! You won a free prize. Click here!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT! Your account has been compromised. Call now!",
        "Can you pick up some milk on your way home?",
        "Win a brand new iPhone! Text WIN to 12345 now!",
        "Meeting postponed to 3pm. See you then.",
        "FREE entry to our weekly competition! Text GO to 80888",
        "Thanks for your help yesterday, really appreciate it.",
    ]

    print("\n" + "=" * 55)
    print("  SAMPLE PREDICTIONS")
    print("=" * 55)

    for text in samples:
        label, conf, proba = predict_text(model, vectorizer, text)
        icon = "🚫" if label == "spam" else "✉️ "
        print(f"\n  {icon} [{label:>4}] {conf:5.1f}%  \"{text}\"")
        print(f"       ham={proba[0]:.3f}  spam={proba[1]:.3f}")

    print("\n" + "=" * 55)


def interactive_mode(model, vectorizer):
    """Interactive REPL for testing custom messages."""
    print("\n" + "=" * 55)
    print("  INTERACTIVE MODE")
    print("  Type a message to classify. Enter 'q' to quit.")
    print("=" * 55)

    while True:
        try:
            text = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text or text.lower() == "q":
            break

        label, conf, proba = predict_text(model, vectorizer, text)
        icon = "🚫" if label == "spam" else "✉️ "
        print(f"  {icon} {label} (confidence: {conf:.1f}%)")
        print(f"     ham={proba[0]:.3f}  spam={proba[1]:.3f}")

    print("\n  Goodbye!")


def main():
    print("=" * 55)
    print("  SPAM CLASSIFIER — EVALUATION & INFERENCE")
    print("=" * 55)

    # Load saved artifacts
    model, vectorizer = load_artifacts()

    # Evaluate on test set
    X_test_text, y_test = load_test_data()
    evaluate_model(model, vectorizer, X_test_text, y_test)

    # Sample predictions
    run_sample_predictions(model, vectorizer)

    # Interactive mode
    interactive_mode(model, vectorizer)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
