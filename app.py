from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "saved_model/credit_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model not found. Run model.py first.")

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["person_age"]),
        float(request.form["person_income"]),
        float(request.form["person_emp_length"]),
        float(request.form["loan_amnt"]),
        float(request.form["loan_int_rate"]),
        float(request.form["loan_percent_income"]),
        float(request.form["cb_person_cred_hist_length"])
    ]

    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]
    risk_percent = round(max(probability) * 100, 2)

    risk = "LOW RISK 🟢" if prediction == 0 else "HIGH RISK 🔴"

    return render_template(
        "result.html",
        prediction=risk,
        probability=risk_percent
    )

if __name__ == "__main__":
    app.run(debug=True)



# from flask import Flask, render_template, request
# import joblib
# import os

# app = Flask(__name__)

# MODEL_PATH = "saved_model/credit_model.pkl"

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError("❌ Model not found! Run model.py first.")

# model = joblib.load(MODEL_PATH)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     features = [
#         float(request.form["person_age"]),
#         float(request.form["person_income"]),
#         float(request.form["person_emp_length"]),
#         float(request.form["loan_amnt"]),
#         float(request.form["loan_int_rate"]),
#         float(request.form["loan_percent_income"]),
#         float(request.form["cb_person_cred_hist_length"])
#     ]

#     prediction = model.predict([features])[0]
#     risk = "🟢 LOW RISK" if prediction == 0 else "🔴 HIGH RISK"

#     return render_template("result.html", prediction=risk)

# if __name__ == "__main__":
#     app.run(debug=True)


