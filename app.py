import os
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import pickle
import sqlite3
import shap
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-only-change-me')

with open('npa_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

explainer = shap.TreeExplainer(model)

grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
emp_map = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10, 'Unknown': -1
}

FEATURE_INFO = {
    'int_rate': ("High interest rate on this loan", "Try improving your credit grade before applying, so you qualify for a lower interest rate."),
    'grade_num': ("Risky loan grade assigned to this loan", "Work on your credit score — a better grade (A/B) significantly lowers default risk."),
    'dti': ("High debt-to-income ratio", "Pay down existing debts or increase income before taking on new loan obligations."),
    'emi_income_ratio': ("High EMI burden relative to income", "Consider a smaller loan amount or a longer tenure to reduce the monthly EMI."),
    'revol_util': ("High credit utilization", "Pay down revolving credit card balances — aim to keep utilization below 30%."),
    'credit_history_years': ("Short credit history", "A longer, consistent credit history builds trust — avoid large loans too early."),
    'emp_length_num': ("Short employment history", "A longer, stable employment record improves approval odds and lowers risk."),
    'delinq_2yrs': ("Recent missed/late payments", "Maintain consistent, on-time payments over the next few months to rebuild trust."),
    'pub_rec': ("Negative public records on file", "Resolve any outstanding public records or judgments before applying."),
    'loan_amnt': ("Large loan amount requested", "Consider requesting a smaller loan amount relative to your income."),
    'annual_inc': ("Income low relative to loan size", "Increasing income or adding a co-applicant can improve this profile."),
    'revol_bal': ("High outstanding revolving balance", "Reduce outstanding credit card / revolving balances before applying."),
    'open_acc': ("High number of open credit lines", "Consider consolidating or closing unused credit accounts."),
    'total_acc': ("High total number of credit accounts", "A simpler credit profile with fewer accounts can appear less risky."),
    'term': ("Longer loan term selected", "A shorter loan term, if affordable, reduces total risk exposure."),
}

def get_feature_explanation(col_name):
    if col_name in FEATURE_INFO:
        return FEATURE_INFO[col_name]
    if col_name.startswith('home_ownership_'):
        return ("Home ownership status may add risk", "Stable housing status (owned/mortgaged) is viewed more favorably over time.")
    if col_name.startswith('verification_status_Not'):
        return ("Income was not verified", "Provide income verification documents to strengthen the application.")
    if col_name.startswith('purpose_'):
        return ("Loan purpose adds some risk", "Certain loan purposes carry higher risk profiles statistically.")
    return (f"{col_name} contributed to the risk score", "Review this factor with a financial advisor.")

def explain_prediction(df_row):
    shap_values = explainer.shap_values(df_row)
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    feature_impacts = list(zip(df_row.columns, values))
    positive_impacts = [f for f in feature_impacts if f[1] > 0]
    positive_impacts.sort(key=lambda x: x[1], reverse=True)

    top_factors = []
    for col_name, val in positive_impacts[:4]:
        label, suggestion = get_feature_explanation(col_name)
        top_factors.append({'label': label, 'suggestion': suggestion})
    return top_factors

DB_NAME = 'npa_database.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT,
            applicant_name TEXT,
            loan_amnt REAL,
            term INTEGER,
            int_rate REAL,
            installment REAL,
            grade TEXT,
            purpose TEXT,
            annual_inc REAL,
            emp_length TEXT,
            home_ownership TEXT,
            verification_status TEXT,
            credit_history_years REAL,
            dti REAL,
            revol_bal REAL,
            revol_util REAL,
            open_acc REAL,
            total_acc REAL,
            delinq_2yrs REAL,
            pub_rec REAL,
            result TEXT,
            risk_percent REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return render_template('signup.html', error="An account with this email already exists.")

        password_hash = generate_password_hash(password)
        cursor.execute(
            'INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)',
            (name, email, password_hash, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        session['user_id'] = user_id
        session['user_name'] = name
        return redirect(url_for('predictor'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            return redirect(url_for('predictor'))
        else:
            return render_template('login.html', error="Invalid email or password.")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

def save_prediction(data, result, risk_percent):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (
            user_id, timestamp, applicant_name, loan_amnt, term, int_rate, installment, grade, purpose,
            annual_inc, emp_length, home_ownership, verification_status,
            credit_history_years, dti, revol_bal, revol_util, open_acc,
            total_acc, delinq_2yrs, pub_rec, result, risk_percent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session.get('user_id'),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        data['applicant_name'],
        data['loan_amnt'], data['term'], data['int_rate'], data['installment'],
        data['grade'], data['purpose'], data['annual_inc'], data['emp_length'],
        data['home_ownership'], data['verification_status'],
        data['credit_history_years'], data['dti'], data['revol_bal'],
        data['revol_util'], data['open_acc'], data['total_acc'],
        data['delinq_2yrs'], data['pub_rec'], result, risk_percent
    ))
    conn.commit()
    conn.close()

def get_all_predictions():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC', (session.get('user_id'),))
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_stats():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result LIKE 'High%'")
    high_risk = cursor.fetchone()[0]
    conn.close()
    return total, high_risk

@app.route('/')
def landing():
    total, high_risk = get_stats()
    return render_template('landing.html', total=total, high_risk=high_risk)

@app.route('/predictor')
@login_required
def predictor():
    return render_template('predictor.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    form = request.form

    data = {
        'applicant_name': form['applicant_name'],
        'loan_amnt': float(form['loan_amnt']),
        'term': int(form['term']),
        'int_rate': float(form['int_rate']),
        'installment': float(form['installment']),
        'annual_inc': float(form['annual_inc']),
        'dti': float(form['dti']),
        'delinq_2yrs': float(form['delinq_2yrs']),
        'open_acc': float(form['open_acc']),
        'pub_rec': float(form['pub_rec']),
        'revol_bal': float(form['revol_bal']),
        'revol_util': float(form['revol_util']),
        'total_acc': float(form['total_acc']),
        'grade': form['grade'],
        'emp_length': form['emp_length'],
        'home_ownership': form['home_ownership'],
        'verification_status': form['verification_status'],
        'purpose': form['purpose'],
        'credit_history_years': float(form['credit_history_years']),
    }

    df = pd.DataFrame([data]).drop(columns=['applicant_name'])

    df['emi_income_ratio'] = df['installment'] / (df['annual_inc'] / 12 + 1)
    df['emp_length_num'] = df['emp_length'].map(emp_map)
    df['grade_num'] = df['grade'].map(grade_map)

    df = pd.get_dummies(df, columns=['home_ownership', 'purpose', 'verification_status'], drop_first=True)
    df = df.drop(columns=['grade', 'emp_length'])

    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    result = "High Risk (Likely NPA)" if prediction == 1 else "Low Risk (Likely Non-NPA)"
    risk_percent = round(probability * 100, 2)

    top_factors = []
    if prediction == 1:
        top_factors = explain_prediction(df)

    save_prediction(data, result, risk_percent)

    return render_template(
        'predictor.html',
        result=result,
        risk_percent=risk_percent,
        show_result=True,
        form_data=form,
        top_factors=top_factors
    )

@app.route('/history')
@login_required
def history():
    records = get_all_predictions()
    return render_template('history.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)