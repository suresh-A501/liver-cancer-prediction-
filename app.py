from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import numpy as np
import pandas as pd
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
app.secret_key = 'super_secret_health_key'  # Needed for session management

# --- DATABASE SETUP ---
def init_db():
    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

init_db()  # Run once on start

# --- MODEL LOADING ---
try:
    model = pickle.load(open('liver.pkl', 'rb'))
except FileNotFoundError:
    model = None
    print("WARNING: 'liver.pkl' not found. Predictions will fail until you run train_model.py")

# --- PAGE ROUTES ---

@app.route('/')
def home():
    return render_template('liver.html', page='home', user=session.get('user'))

@app.route('/services')
def services():
    return render_template('liver.html', page='services', user=session.get('user'))

@app.route('/about')
def about():
    return render_template('liver.html', page='about', user=session.get('user'))

@app.route('/contact')
def contact():
    return render_template('liver.html', page='contact', user=session.get('user'))

# --- AUTH ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()
            
            if user and check_password_hash(user[3], password):
                session['user'] = user[1] # Store name in session
                session['user_id'] = user[0]
                flash(f"Welcome back, {user[1]}!", "success")
                return redirect(url_for('diagnosis'))
            else:
                flash("Invalid email or password.", "error")
                
    return render_template('liver.html', page='login')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        hashed_pw = generate_password_hash(password)
        
        try:
            with sqlite3.connect("users.db") as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
                               (name, email, hashed_pw))
                conn.commit()
            flash("Account created! Please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already registered. Try logging in.", "error")
            return redirect(url_for('login'))
            
    return render_template('liver.html', page='signup')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# --- GOOGLE LOGIN (SIMULATION) ---
# This fixes the 404 Error
@app.route('/login/google')
def google_login():
    # Since we don't have a real Google Client Secret set up, 
    # we simulate a successful login for demonstration.
    session['user'] = "Google User"
    session['user_id'] = 999
    flash("Successfully logged in with Google (Simulation Mode)", "success")
    return redirect(url_for('diagnosis'))

# --- APP FEATURES ---

@app.route('/diagnosis')
def diagnosis():
    # Protect the route: Only logged in users can see this
    if 'user' not in session:
        flash("Please log in to access the Diagnosis Tool.", "error")
        return redirect(url_for('login'))
    
    return render_template('liver.html', page='diagnosis', user=session.get('user'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if model is None:
        return "Model not loaded. Please run train_model.py first."

    # 1. Get data
    try:
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]
        
        features_name = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                         'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                         'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
                         'Albumin_and_Globulin_Ratio']
        
        df = pd.DataFrame(features_value, columns=features_name)
        
        # 2. Predict
        output = model.predict(df)
        
        if output[0] == 1:
            res_val = "POSITIVE: Potential Liver Issue Detected"
            res_color = "danger"
        else:
            res_val = "NEGATIVE: Liver Function Appears Normal"
            res_color = "success"
            
        return render_template('liver.html', page='diagnosis', 
                               prediction_text=res_val, 
                               prediction_color=res_color,
                               user=session.get('user'))
                               
    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)