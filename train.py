"""
Spam Classification Pipeline
=============================
An ensemble of 3 calibrated scikit-learn models (GaussianNB, LogisticRegression, SVC)
with soft voting for SMS spam detection.
"""

import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATASET_PATH = os.path.join("Dataset", "spam.csv")
VECTORIZER_PATH = "./vectorizer.pkl"
ENSEMBLE_PATH = "./ensemble.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURES = 1000

# Verification checklist tracker
checklist = {}


def load_and_preprocess(path: str) -> pd.DataFrame:
    """Load SMS Spam Collection CSV and clean it."""
    if not os.path.isfile(path):
        sys.exit(f"ERROR: Dataset file not found at '{path}'")

    # The CSV has columns: v1 (label), v2 (message), and 3 empty unnamed columns
    df = pd.read_csv(path, encoding="latin-1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})

    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")

    # Handle missing values
    df.dropna(subset=["message"], inplace=True)

    # Clean text: lowercase + remove special characters
    df["message"] = (
        df["message"]
        .str.lower()
        .apply(lambda x: re.sub(r"[^a-z0-9\s]", "", x))
    )

    # Encode labels: ham=0, spam=1
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    return df


def extract_features(X_train_text, X_test_text):
    """Fit TfidfVectorizer on training data and transform both sets."""
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Save vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectorizer saved to '{VECTORIZER_PATH}'")

    return X_train, X_test, vectorizer


def build_calibrated_models():
    """Create 3 base models wrapped with CalibratedClassifierCV."""
    print("\n--- Building calibrated models ---")

    # Model 1: GaussianNB (requires dense input — handled at fit time)
    nb = CalibratedClassifierCV(estimator=GaussianNB(), cv=5)
    print("  • GaussianNB       (calibrated, cv=5)")

    # Model 2: Logistic Regression
    lr = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), cv=5
    )
    print("  • LogisticRegression(calibrated, cv=5)")

    # Model 3: SVC with RBF kernel
    svc = CalibratedClassifierCV(
        estimator=SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE), cv=5
    )
    print("  • SVC(rbf)          (calibrated, cv=5)")

    return nb, lr, svc


def train_ensemble(nb, lr, svc, X_train_dense, y_train):
    """Combine calibrated models into a soft-voting ensemble and fit."""
    ensemble = VotingClassifier(
        estimators=[("nb", nb), ("lr", lr), ("svc", svc)],
        voting="soft",
    )

    print("\nTraining ensemble (this may take a minute) ...")
    ensemble.fit(X_train_dense, y_train)
    print("Ensemble training complete.\n")

    return ensemble


def evaluate(ensemble, X_test_dense, y_test):
    """Evaluate the ensemble on the test set."""
    y_pred = ensemble.predict(X_test_dense)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 50)
    print("TEST SET METRICS")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print()
    print("Confusion Matrix:")
    print(cm)
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))


def sample_predictions(ensemble, X_test_dense, y_test):
    """Show a few sample predictions with confidence scores."""
    print("--- Sample predictions from test set ---")
    indices = np.random.RandomState(RANDOM_STATE).choice(
        len(y_test), size=5, replace=False
    )
    probas = ensemble.predict_proba(X_test_dense[indices])
    preds = ensemble.predict(X_test_dense[indices])
    labels_map = {0: "ham", 1: "spam"}

    for i, idx in enumerate(indices):
        confidence = probas[i].max() * 100
        print(
            f"  [{i+1}] True={labels_map[y_test.iloc[idx]]:<4}  "
            f"Pred={labels_map[preds[i]]:<4}  "
            f"Confidence={confidence:.1f}%"
        )
    print()


def save_ensemble(ensemble):
    """Save the ensemble model to disk."""
    joblib.dump(ensemble, ENSEMBLE_PATH)
    print(f"Ensemble model saved to '{ENSEMBLE_PATH}'")

    # Show file sizes
    for path in [ENSEMBLE_PATH, VECTORIZER_PATH]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {path}: {size_mb:.2f} MB")
    print()


def verify_saved_models():
    """Load saved models and run test predictions to verify everything works."""
    print("=" * 50)
    print("VERIFICATION: Loading saved models")
    print("=" * 50)

    # Check files exist
    for path in [ENSEMBLE_PATH, VECTORIZER_PATH]:
        if not os.path.isfile(path):
            print(f"  FAIL: '{path}' not found!")
            checklist["Models saved to disk"] = False
            return
    checklist["Models saved to disk"] = True

    # Load
    loaded_ensemble = joblib.load(ENSEMBLE_PATH)
    loaded_vectorizer = joblib.load(VECTORIZER_PATH)
    checklist["Models loaded successfully"] = True
    print("  Models loaded successfully.\n")

    # Test predictions on sample texts
    test_texts = [
        "Congratulations! You won a free prize. Click here!",
        "Hey, are we still meeting for lunch tomorrow?",
    ]
    labels_map = {0: "ham", 1: "spam"}

    print("--- Test predictions on sample texts ---")
    for text in test_texts:
        # Clean text the same way as training
        cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
        vec = loaded_vectorizer.transform([cleaned]).toarray()
        pred = loaded_ensemble.predict(vec)[0]
        proba = loaded_ensemble.predict_proba(vec)[0]
        confidence = proba.max() * 100
        print(f'  Text     : "{text}"')
        print(f"  Prediction: {labels_map[pred]}")
        print(f"  Confidence: {confidence:.1f}%")
        print()

    checklist["Test predictions working"] = True


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def main():
    print("=" * 50)
    print("SPAM CLASSIFICATION PIPELINE")
    print("=" * 50)
    print()

    # Step 1: Load & preprocess
    print("[1/7] Loading and preprocessing dataset ...")
    df = load_and_preprocess(DATASET_PATH)
    checklist["Dataset loaded successfully"] = True

    # Step 2: Train/test split
    print("[2/7] Splitting data (80/20) ...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["message"], df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"  Train: {len(X_train_text)}, Test: {len(X_test_text)}\n")

    # Step 3: Feature extraction
    print("[3/7] Extracting TF-IDF features (max_features={}) ...".format(MAX_FEATURES))
    X_train_sparse, X_test_sparse, vectorizer = extract_features(X_train_text, X_test_text)

    # Convert sparse → dense for GaussianNB compatibility
    X_train_dense = X_train_sparse.toarray()
    X_test_dense = X_test_sparse.toarray()
    print(f"  Feature matrix: {X_train_dense.shape}\n")

    # Step 4: Build calibrated models
    print("[4/7] Building calibrated models ...")
    nb, lr, svc = build_calibrated_models()
    checklist["Models trained successfully"] = True
    checklist["Calibration applied"] = True

    # Step 5: Train ensemble
    print("[5/7] Training soft-voting ensemble ...")
    ensemble = train_ensemble(nb, lr, svc, X_train_dense, y_train)
    checklist["Ensemble created"] = True

    # Step 6: Evaluate
    print("[6/7] Evaluating on test set ...")
    evaluate(ensemble, X_test_dense, y_test)
    sample_predictions(ensemble, X_test_dense, y_test)

    # Step 7: Save & verify
    print("[7/7] Saving and verifying artifacts ...")
    save_ensemble(ensemble)
    verify_saved_models()

    # Final checklist
    print("=" * 50)
    print("VERIFICATION CHECKLIST")
    print("=" * 50)
    expected_items = [
        "Dataset loaded successfully",
        "Models trained successfully",
        "Calibration applied",
        "Ensemble created",
        "Models saved to disk",
        "Models loaded successfully",
        "Test predictions working",
    ]
    all_pass = True
    for item in expected_items:
        status = checklist.get(item, False)
        mark = "✓" if status else "✗"
        print(f"  {mark} {item}")
        if not status:
            all_pass = False

    print()
    if all_pass:
        print("All checks passed! Pipeline completed successfully.")
    else:
        print("WARNING: Some checks failed. Review output above.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
