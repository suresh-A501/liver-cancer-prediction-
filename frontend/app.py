from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import numpy as np
import pandas as pd
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import os
# NEW: Import Authlib for Google Login
# You must run: pip install authlib requests
from authlib.integrations.flask_client import OAuth

app = Flask(__name__)
app.secret_key = 'super_secret_health_key'  # Needed for session management

# --- GOOGLE OAUTH CONFIGURATION ---
# 1. Go to Google Cloud Console -> Credentials -> Create OAuth Client ID
# 2. Set Authorized Redirect URI to: http://127.0.0.1:5000/auth/callback
app.config['GOOGLE_CLIENT_ID'] = 'ENTER_YOUR_GOOGLE_CLIENT_ID'
app.config['GOOGLE_CLIENT_SECRET'] = 'ENTER_YOUR_GOOGLE_CLIENT_SECRET'

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=app.config['GOOGLE_CLIENT_ID'],
    client_secret=app.config['GOOGLE_CLIENT_SECRET'],
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# Allow HTTP for OAuth (Only for local testing, remove in production)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'


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

# --- PAGE ROUTES (Fixes the 404 Errors) ---

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

# --- GOOGLE AUTH ROUTES (NEW) ---

@app.route('/login/google')
def login_google():
    # Redirect user to Google for authorization
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth/callback')
def google_callback():
    try:
        token = google.authorize_access_token()
        # Fetch user info using the token
        user_info = google.get('https://www.googleapis.com/oauth2/v3/userinfo').json()
        
        name = user_info.get('name')
        email = user_info.get('email')
        
        # Check if user exists in our DB
        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()
            
            if user:
                # User exists - Log them in
                session['user'] = user[1]
                session['user_id'] = user[0]
            else:
                # User doesn't exist - Create new account automatically
                # We set a dummy password since they use Google to login
                dummy_pw = generate_password_hash("GOOGLE_OAUTH_USER_SECURE")
                cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
                               (name, email, dummy_pw))
                conn.commit()
                
                # Get the ID of the new user
                user_id = cursor.lastrowid
                session['user'] = name
                session['user_id'] = user_id
        
        flash(f"Welcome back, {name}!", "success")
        return redirect(url_for('diagnosis'))
        
    except Exception as e:
        flash(f"Google Login failed: {str(e)}", "error")
        return redirect(url_for('login'))

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