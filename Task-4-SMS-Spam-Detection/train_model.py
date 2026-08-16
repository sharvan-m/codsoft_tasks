from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent

DATASET = BASE_DIR / "data" / "spam.csv"
MODEL_FILE = BASE_DIR / "spam_model.joblib"


# Load dataset
df = pd.read_csv(
    DATASET,
    encoding="latin-1",
    usecols=[0, 1],
    names=["label", "message"],
    header=0
)

df = df.dropna(subset=["label", "message"])

df["label"] = df["label"].str.strip().str.lower()

df = df[df["label"].isin(["ham", "spam"])]

print(f"Dataset rows: {len(df):,}")
print(f"Ham messages: {(df['label'] == 'ham').sum():,}")
print(f"Spam messages: {(df['label'] == 'spam').sum():,}")


# Features and target
X = df["message"]
y = df["label"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")


# TF-IDF + Linear SVM
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=50000
        )
    ),
    (
        "classifier",
        LinearSVC(class_weight="balanced")
    )
])


print("\nTraining model...")
model.fit(X_train, y_train)


# Predictions
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# Save model
joblib.dump(model, MODEL_FILE)

print(f"\nModel saved successfully: {MODEL_FILE.name}")