from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "churn_model.joblib"

model = joblib.load(MODEL_FILE)

print("Customer Churn Predictor")
print("Enter customer details below.")
print("Type exit at any prompt to stop.\n")


def get_input(prompt, converter=str):
    value = input(prompt).strip()

    if value.lower() == "exit":
        return None

    try:
        return converter(value)
    except ValueError:
        print("Invalid input. Please try again.")
        return get_input(prompt, converter)


while True:
    credit_score = get_input("Credit Score: ", int)
    if credit_score is None:
        break

    geography = get_input("Geography (France/Spain/Germany): ")
    if geography is None:
        break

    gender = get_input("Gender (Male/Female): ")
    if gender is None:
        break

    age = get_input("Age: ", int)
    if age is None:
        break

    tenure = get_input("Tenure: ", int)
    if tenure is None:
        break

    balance = get_input("Balance: ", float)
    if balance is None:
        break

    num_products = get_input("Number of Products: ", int)
    if num_products is None:
        break

    has_card = get_input("Has Credit Card? (0/1): ", int)
    if has_card is None:
        break

    active_member = get_input("Is Active Member? (0/1): ", int)
    if active_member is None:
        break

    salary = get_input("Estimated Salary: ", float)
    if salary is None:
        break

    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_card,
        "IsActiveMember": active_member,
        "EstimatedSalary": salary
    }])

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    print("\n------------------------------")

    if prediction == 1:
        print("Prediction: CUSTOMER MAY CHURN")
    else:
        print("Prediction: CUSTOMER LIKELY TO STAY")

    print(f"Churn Probability: {probability * 100:.2f}%")
    print("------------------------------\n")