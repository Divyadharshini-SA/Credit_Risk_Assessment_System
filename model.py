import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv("dataset/credit_risk_dataset.csv")

# Separate numeric & categorical columns
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include="object").columns

# Handle missing values
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# # Split features and target
# X = df.drop("loan_status", axis=1)
# y = df["loan_status"]
selected_features = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

X = df[selected_features]
y = df["loan_status"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("saved_model", exist_ok=True)
joblib.dump(model, "saved_model/credit_model.pkl")

print("✅ Model trained successfully!")
print("✅ Accuracy:", accuracy_score(y_test, model.predict(X_test)))





# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score
# import joblib
# import os

# # Load dataset
# df = pd.read_csv("dataset/credit_risk_dataset.csv")

# # Select ONLY the 7 features used in UI
# features = [
#     "person_age",
#     "person_income",
#     "person_emp_length",
#     "loan_amnt",
#     "loan_int_rate",
#     "loan_percent_income",
#     "cb_person_cred_hist_length"
# ]

# X = df[features]
# y = df["loan_status"]

# # Handle missing values
# X = X.fillna(X.median())

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train model
# model = GradientBoostingClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Save model
# os.makedirs("saved_model", exist_ok=True)
# joblib.dump(model, "saved_model/credit_model.pkl")

# print("✅ Model trained successfully")
# print("✅ Features used:", model.n_features_in_)
# print("✅ Accuracy:", accuracy_score(y_test, model.predict(X_test)))


