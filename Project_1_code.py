"""
Customer Insurance Purchase Predictor
======================================
Trains and evaluates six ML classifiers on synthetic customer data
(Age + Salary => Purchase) and persists the best model for inference.

Usage:
    python Project_1_code.py train              # train, evaluate, and save best model
    python Project_1_code.py predict <age> <salary>  # predict with saved model
"""

import sys
import os
import time
import random
import logging
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_CUSTOMERS = 500
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3

MODEL_DIR = Path(__file__).parent / "saved_model"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
META_PATH = MODEL_DIR / "meta.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def salary_by_age(age: int) -> float:
    """Return a synthetic salary (INR) drawn from an age-dependent exponential."""
    if age < 25:
        base = np.random.exponential(2_00_000)
    elif age < 35:
        base = np.random.exponential(3_50_000)
    elif age < 45:
        base = np.random.exponential(5_00_000)
    else:
        base = np.random.exponential(7_00_000)
    return min(base * np.random.uniform(0.8, 1.5), 25_00_000)


def purchase_chance(age: int, salary: float) -> float:
    """Heuristic probability of purchasing insurance."""
    p = 0.1
    if 25 <= age <= 45:
        p += 0.3
    elif 45 < age <= 55:
        p += 0.1
    if salary > 6_00_000:
        p += 0.3
    elif salary > 4_00_000:
        p += 0.2
    elif salary > 2_50_000:
        p += 0.1
    if 30 <= age <= 50:
        p += 0.15
    return p


def generate_dataset(n: int = N_CUSTOMERS) -> pd.DataFrame:
    """Create a synthetic dataset of customer ages, salaries, and purchases."""
    ages = np.random.randint(18, 66, n)
    salaries = np.array([salary_by_age(a) for a in ages])
    purchases = np.array([
        1 if purchase_chance(a, s) + np.random.normal(0, 0.1) + random.uniform(-0.2, 0.2) > 0.5 else 0
        for a, s in zip(ages, salaries)
    ])
    return pd.DataFrame({
        'Age': ages,
        'EstimatedSalary': salaries,
        'Purchased': purchases,
    })

# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """Return a dict of candidate classifiers."""
    return {
        'LogReg': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(5),
        'SVM': SVC(probability=True, kernel='rbf'),
        'Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=20),
        'Forest': RandomForestClassifier(n_estimators=100, max_depth=10),
        'NN': MLPClassifier((100, 50), max_iter=2000),
    }

# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train(silent: bool = False) -> list:
    """
    Train all models, evaluate, select the best, and persist it.

    Parameters
    ----------
    silent : bool
        If True, suppress console print (used by Flask API).

    Returns
    -------
    list[dict] : leaderboard results sorted by CVAcc descending.
    """
    log.info("Generating synthetic dataset (%d customers)...", N_CUSTOMERS)
    df = generate_dataset()
    purchase_rate = df['Purchased'].mean() * 100
    log.info("Purchase rate: %.1f%%", purchase_rate)

    X = df[['Age', 'EstimatedSalary']]
    y = df['Purchased']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = get_models()
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, model in models.items():
        start = time.time()
        model.fit(X_train_sc, y_train)
        train_time = time.time() - start

        preds = model.predict(X_test_sc)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted')
        rec = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')
        cv_acc = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring='accuracy').mean()

        results.append({
            'Model': name,
            'TestAcc': round(acc, 4),
            'CVAcc': round(cv_acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'TrainTime': f'{train_time:.2f}s',
        })

    results_df = pd.DataFrame(results).sort_values('CVAcc', ascending=False)

    if not silent:
        print("\n=== Model Leaderboard ===")
        print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]['Model']
    best_model = models[best_name]
    log.info("Best model: %s (CV Accuracy: %.2f%%)", best_name,
             results_df.iloc[0]['CVAcc'] * 100)

    # Persist model + scaler + metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({
        'best_model_name': best_name,
        'features': ['Age', 'EstimatedSalary'],
        'n_customers': N_CUSTOMERS,
        'purchase_rate': round(purchase_rate, 1),
    }, META_PATH)
    log.info("Saved model artefacts to %s", MODEL_DIR)

    return results_df.to_dict('records')


def get_model_info() -> dict:
    """Return metadata about the currently saved model."""
    if not META_PATH.exists():
        return None
    meta = joblib.load(META_PATH)
    return meta

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_insurance(age: int, salary: float) -> Tuple[str, float]:
    """
    Load the persisted model and predict insurance purchase.

    Returns
    -------
    label : str   - 'Yes' or 'No'
    prob  : float - probability of purchase
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No saved model found. Run `python Project_1_code.py train` first."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    data = scaler.transform([[age, salary]])
    prob = model.predict_proba(data)[0][1]
    label = 'Yes' if prob > 0.5 else 'No'
    return label, prob

# ---------------------------------------------------------------------------
# CLI Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == 'train':
        train()
        # Quick demo predictions after training
        print("\n=== Sample Predictions ===")
        profiles = [(30, 7_20_000), (40, 0), (40, 8_50_000), (50, 0)]
        for age, sal in profiles:
            label, prob = predict_insurance(age, sal)
            print(f"  Age {age:>3}, Salary Rs.{sal:>10,}  =>  {label}  ({prob:.0%})")

    elif command == 'predict':
        if len(sys.argv) != 4:
            print("Usage: python Project_1_code.py predict <age> <salary>")
            sys.exit(1)
        age = int(sys.argv[2])
        salary = float(sys.argv[3])
        label, prob = predict_insurance(age, salary)
        print(f"Prediction:  Age {age}, Salary Rs.{salary:,.0f}  =>  {label}  ({prob:.0%})")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
