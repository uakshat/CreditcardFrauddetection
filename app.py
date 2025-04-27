from flask import Flask, render_template, request, redirect, url_for, session, flash
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
import xgboost as xgb

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Hardcoded user login credentials
users = {
    "client1": "pass123",
    "client2": "secure456",
    "admin": "adminpass"
}

# Load the XGBoost model
model = joblib.load('fraud_model_xgb.pkl')

# Rule-based logic
def rule_based_fraud_check(df, credit_limit):
    if df['Amount'].sum() > credit_limit:
        return True
    if any(df['Amount'] > 0.8 * credit_limit):
        return True
    if df['Category'].value_counts().get('Entertainment', 0) >= 4:
        return True
    return False

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        client_id = request.form['client_id']
        password = request.form['password']
        if client_id in users and users[client_id] == password:
            session['user'] = client_id
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# Home Page
@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

# Fraud Detection Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    # Get form values
    name = request.form['name']
    bank = request.form['bank']
    last4 = request.form['last4']
    credit_limit = float(request.form['limit'])
    file = request.files['file']
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # Remove any extra spaces

    # Validate file
    if not {'Amount', 'Category'}.issubset(df.columns):
        return "CSV must contain 'Amount' and 'Category'."

    # Rule-based checks
    rule_flag = rule_based_fraud_check(df, credit_limit)

    # Prepare ML input
    X = pd.get_dummies(df['Category'])
    for col in ['Bills', 'Entertainment', 'Food', 'Personal']:  # Match training column names
        if col not in X:
            X[col] = 0
    X['Amount'] = df['Amount']
    X.columns = [str(col) for col in X.columns]
    
    # Sort columns in same order as during training
    expected_order = ['Amount', 'Bills', 'Entertainment', 'Food', 'Personal']
    X = X.reindex(columns=sorted(expected_order), fill_value=0)
    X_input = pd.DataFrame([X.sum()])

    # Reindex X_input too
    X_input = X_input.reindex(columns=sorted(expected_order), fill_value=0)

    # Predict with XGBoost
    dinput = xgb.DMatrix(X_input, feature_names=X_input.columns.tolist())
    pred = model.predict(dinput)[0]
    fraud = rule_flag or (pred >= 0.5)

    # Create charts
    os.makedirs("static", exist_ok=True)
    category_summary = df.groupby("Category")["Amount"].sum()

    # Pie Chart
    plt.figure(figsize=(4, 4))
    plt.pie(category_summary, labels=category_summary.index, autopct='%1.1f%%', startangle=140)
    plt.title("Spending by Category")
    plt.tight_layout()
    plt.savefig("static/pie_chart.png")
    plt.close()

    # Bar Chart
    plt.figure(figsize=(5, 3))
    category_summary.plot(kind='bar', color='skyblue')
    plt.title("Amount per Category")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.savefig("static/bar_chart.png")
    plt.close()

    # Return result page
    return render_template('result.html',
        name=name,
        bank=bank,
        last4=last4,
        limit=credit_limit,
        total=df['Amount'].sum(),
        category_summary=category_summary.to_dict(),
        fraud=fraud
    )

# Run app
if __name__ == '__main__':
    app.run(debug=True)
