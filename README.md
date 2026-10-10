# NPA Risk Predictor

A Flask web app that predicts whether a loan is likely to become a Non-Performing Asset (NPA), using an XGBoost model trained on the Lending Club dataset.

## Features
- XGBoost classifier (79.86% accuracy) for loan default prediction
- "Why High Risk" explanations powered by SHAP, with suggestions to reduce risk
- User signup/login with hashed passwords and session-based access
- SQLite database storing each user's prediction history

## Tech Stack
Python, Flask, XGBoost, SHAP, pandas, SQLite

## Setup
1. `pip install -r requirements.txt`
2. `python app.py`
3. Open `http://127.0.0.1:5000`

## Model
Trained on the Lending Club dataset (Kaggle). "Charged Off" loans are treated as NPA, "Fully Paid" as non-NPA. The trained model is saved as `npa_model.pkl`.

## Note
The dataset is US-based, so the target is a proxy for NPA and does not follow RBI's 90-DPD classification.