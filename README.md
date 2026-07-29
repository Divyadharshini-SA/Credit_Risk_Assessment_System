# Intelligent Credit Risk Assessment System

## Overview

The **Intelligent Credit Risk Assessment System** is a Machine Learning web application developed using **Python**, **Flask**, and **Scikit-learn**. It predicts whether a loan applicant is at **Low Risk** or **High Risk** based on financial and personal information provided by the user.

The application uses a trained Machine Learning model stored in the `saved_model` folder and provides predictions through a simple web interface.

---

## Features

- Predicts credit risk using Machine Learning
- User-friendly web interface
- Instant prediction results
- Built using Flask
- Easy to retrain with new datasets

---

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- HTML5
- CSS3

---

## Project Structure

```
CREDIT_RISK_ASSESSMENT_SYSTEM/
│
├── dataset/
│   └── credit_risk_dataset.csv
│
├── saved_model/
│   └── credit_model.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── model.py
├── requirements.txt
└── README.md
```

---

## Input Features

The application accepts the following inputs:

- Age
- Annual Income
- Employment Length (Years)
- Loan Amount
- Interest Rate (%)
- Loan % of Income
- Credit History Length

---

## Output

The model predicts one of the following:

- **Low Risk**
- **High Risk**

---

# Installation

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Credit_Risk_Assessment_System.git
```

or download the ZIP file and extract it.

---

## Step 2: Open the Project Folder

```bash
cd Credit_Risk_Assessment_System
```

---

## Step 3: Create a Virtual Environment (Recommended)

Windows:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
venv\Scripts\activate
```

---

## Step 4: Install Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
Flask
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

---

## Step 5: (Optional) Train the Model

If `model.py` is used to train the model, run:

```bash
python model.py
```

This will train the model using the dataset and save it as:

```
saved_model/credit_model.pkl
```

---

## Step 6: Run the Flask Application

```bash
python app.py
```

You should see:

```
* Running on http://127.0.0.1:5000/
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Sample Test Data

| Feature | Sample Value |
|---------|-------------:|
| Age | 35 |
| Annual Income | 80000 |
| Employment Length | 10 |
| Loan Amount | 12000 |
| Interest Rate | 8.5 |
| Loan % of Income | 15 |
| Credit History Length | 12 |

Another example:

| Feature | Sample Value |
|---------|-------------:|
| Age | 29 |
| Annual Income | 45000 |
| Employment Length | 4 |
| Loan Amount | 18000 |
| Interest Rate | 12 |
| Loan % of Income | 40 |
| Credit History Length | 5 |

---

## Machine Learning Workflow

1. Load the credit risk dataset.
2. Preprocess and clean the data.
3. Select the required features.
4. Train a classification model using Scikit-learn.
5. Save the trained model with Joblib.
6. Load the model in the Flask application.
7. Predict the applicant's credit risk based on user input.

---

## Common Errors

### Model Version Mismatch

**Error:**

```
InconsistentVersionWarning
```

**Solution:**

Install the same version of Scikit-learn used to train the model or retrain the model.

Example:

```bash
pip install scikit-learn==1.7.2
```

---

### Model File Not Found

**Error:**

```
FileNotFoundError
```

**Solution:**

Ensure that the model file exists at:

```
saved_model/credit_model.pkl
```

---

### Missing Python Packages

**Error:**

```
ModuleNotFoundError
```

**Solution:**

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Future Enhancements

- User Authentication
- Explainable AI (SHAP/LIME)
- Loan Approval Recommendation
- Interactive Dashboard
- Database Integration
- Cloud Deployment (Render, Railway, Azure)
- Model Performance Comparison

---
