import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# GENERATE ROBUST SYNTHETIC DATA
# ==========================================
# We will generate 1000 samples: 500 Healthy, 500 Disease
# Based on standard medical ranges.
np.random.seed(42)
n_samples = 500

# --- 1. HEALTHY PATIENTS (Class 0) ---
# Normal ranges: Bilirubin < 1.2, Enzymes < 40-50, Albumin > 3.5
healthy_data = {
    'Age': np.random.randint(20, 70, n_samples),
    'Gender': np.random.randint(0, 2, n_samples), # 0 or 1
    'Total_Bilirubin': np.round(np.random.uniform(0.4, 1.2, n_samples), 1),
    'Direct_Bilirubin': np.round(np.random.uniform(0.1, 0.3, n_samples), 1),
    'Alkaline_Phosphotase': np.random.randint(150, 220, n_samples),
    'Alamine_Aminotransferase': np.random.randint(15, 40, n_samples),
    'Aspartate_Aminotransferase': np.random.randint(15, 40, n_samples),
    'Total_Protiens': np.round(np.random.uniform(6.0, 8.5, n_samples), 1),
    'Albumin': np.round(np.random.uniform(3.5, 5.0, n_samples), 1),
    'Albumin_and_Globulin_Ratio': np.round(np.random.uniform(0.9, 1.5, n_samples), 2),
    'Dataset': np.zeros(n_samples) # 0 = Negative (Healthy)
}

# --- 2. DISEASE PATIENTS (Class 1) ---
# Abnormal ranges: Bilirubin High, Enzymes High, Albumin Low
disease_data = {
    'Age': np.random.randint(30, 85, n_samples),
    'Gender': np.random.randint(0, 2, n_samples),
    'Total_Bilirubin': np.round(np.random.uniform(1.3, 15.0, n_samples), 1),
    'Direct_Bilirubin': np.round(np.random.uniform(0.4, 8.0, n_samples), 1),
    'Alkaline_Phosphotase': np.random.randint(230, 1500, n_samples),
    'Alamine_Aminotransferase': np.random.randint(45, 1000, n_samples),
    'Aspartate_Aminotransferase': np.random.randint(45, 1000, n_samples),
    'Total_Protiens': np.round(np.random.uniform(3.0, 8.0, n_samples), 1),
    'Albumin': np.round(np.random.uniform(1.5, 3.4, n_samples), 1),
    'Albumin_and_Globulin_Ratio': np.round(np.random.uniform(0.1, 0.8, n_samples), 2),
    'Dataset': np.ones(n_samples) # 1 = Positive (Disease)
}

# --- 3. COMBINE DATA ---
df_healthy = pd.DataFrame(healthy_data)
df_disease = pd.DataFrame(disease_data)
df = pd.concat([df_healthy, df_disease], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- 4. TRAIN MODEL ---
X = df.drop('Dataset', axis=1)
y = df['Dataset']

# Split to verify accuracy (Internal check)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check Accuracy
print(f"Model trained on {len(df)} records.")
print(f"Internal Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")

# --- 5. SAVE FINAL MODEL ---
# Train on FULL dataset for the final file
model.fit(X, y) 
pickle.dump(model, open('liver.pkl', 'wb'))

print("Success! 'liver.pkl' has been updated with high-accuracy logic.")