from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "spam_model.joblib"

model = joblib.load(MODEL_FILE)

print("SMS Spam Detector")
print("Type an SMS message to classify it.")
print("Type 'exit' to stop.\n")


while True:
    message = input("Enter SMS: ").strip()

    if message.lower() == "exit":
        break

    if not message:
        print("Please enter a message.\n")
        continue

    prediction = model.predict([message])[0]

    print(f"Prediction: {prediction.upper()}\n")