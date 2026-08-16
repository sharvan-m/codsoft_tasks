from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Always work relative to this script's folder
BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "data" / "train_data.txt"
TEST_FILE = BASE_DIR / "data" / "test_data.txt"
MODEL_FILE = BASE_DIR / "movie_genre_model.joblib"
OUTPUT_FILE = BASE_DIR / "test_predictions.csv"


# Load training data
train = pd.read_csv(
    TRAIN_FILE,
    sep=":::",
    engine="python",
    header=None,
    names=["id", "title", "genre", "description"]
)

train = train.dropna(subset=["genre", "description"])

train["text"] = (
    train["title"].fillna("") + " " +
    train["description"].fillna("")
)

train["genre"] = train["genre"].str.strip().str.lower()


# Split data
X_train, X_val, y_train, y_val = train_test_split(
    train["text"],
    train["genre"],
    test_size=0.20,
    random_state=42,
    stratify=train["genre"]
)


# Build model
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "clf",
        LinearSVC(class_weight="balanced")
    )
])


# Train
print(f"Training rows: {len(train):,}")
print(f"Genres: {train['genre'].nunique()}")
print("Training model...")

model.fit(X_train, y_train)


# Validation
pred = model.predict(X_val)

acc = accuracy_score(y_val, pred)

print(f"Validation Accuracy: {acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_val, pred, zero_division=0))


# Save model
joblib.dump(model, MODEL_FILE)

print(f"\nModel saved: {MODEL_FILE.name}")


# Predict supplied test data
try:
    test = pd.read_csv(
        TEST_FILE,
        sep=":::",
        engine="python",
        header=None,
        names=["id", "title", "description"]
    )

    test["text"] = (
        test["title"].fillna("") + " " +
        test["description"].fillna("")
    )

    test_pred = model.predict(test["text"])

    output = pd.DataFrame({
        "id": test["id"],
        "title": test["title"],
        "predicted_genre": test_pred
    })

    output.to_csv(OUTPUT_FILE, index=False)

    print(
        f"Test predictions saved: "
        f"{OUTPUT_FILE.name} ({len(output):,} rows)"
    )

except Exception as e:
    print("Test prediction step failed:", e)