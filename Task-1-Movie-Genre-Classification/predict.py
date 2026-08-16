from pathlib import Path
import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "movie_genre_model.joblib"

model = joblib.load(MODEL_FILE)

print("Movie Genre Predictor")
print("Type a movie plot/description. Type 'exit' to stop.\n")

while True:
    title = input("Movie title: ").strip()

    if title.lower() == "exit":
        break

    description = input("Movie description: ").strip()

    if description.lower() == "exit":
        break

    text = title + " " + description

    prediction = model.predict([text])[0]

    print(f"Predicted Genre: {prediction}\n")