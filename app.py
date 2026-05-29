import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Diabetes AI Predictor Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NAMA DATASET SESUAI FILE KAMU ---
NAMA_FILE = "diabetes.csv"

# --- PROSES PEMBACAAN DATA & TRAINING ---
if os.path.exists(NAMA_FILE):
    df = pd.read_csv(NAMA_FILE)
    df.columns = df.columns.str.strip() # Hapus spasi gaib pada nama kolom
    
    # Memisahkan fitur dan target (Outcome)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Membagi data train & test dengan parameter yang benar (test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardisasi Fitur Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Membangun Model Machine Learning Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Menghitung Akurasi Model
    model_accuracy = model.score(X_test_scaled, y_test) * 100
else:
    df, model, scaler, model_accuracy = None, None, None, 0

# --- SIDEBAR INTERFACE ---
st.sidebar.title("🩺 DIABETES AI")
st.sidebar.subheader("Sistem Prediksi Klinis")
menu = st.sidebar.radio("Navigasi Menu", ["Dashboard Utama", "Prediksi AI", "Informasi Dataset"])

# --- MENU 1: DASHBOARD UTAMA ---
if menu == "Dashboard Utama":
    st.title("Enterprise-grade Diabetes Prediction Dashboard")
    st.write("Sistem kecerdasan buatan berbasis Machine Learning untuk mendeteksi risiko diabetes secara dini.")
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Dataset Records", value=f"{len(df)} Pasien", delta="Validated")
        with col2:
            st.metric(label="AI Prediction Accuracy", value=f"{model_accuracy:.1f} %", delta="Random Forest Classifier")
        with col3:
            st.metric(label="System Status", value="Online / Operational", delta="Real-time Engine")
            
        st.markdown("---")
        st.subheader("Visualisasi Sebaran Fitur Klinis Utama")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Distribusi Glukosa Berdasarkan Diagnosa (0: Sehat, 1: Diabetes)**")
            st.bar_chart(df.groupby('Outcome')['Glucose'].mean())
        with c2:
            st.write("**Hubungan Faktor Usia (Age) terhadap Nilai BMI Pasien**")
            st.scatter_chart(data=df, x='Age', y='BMI', color='Outcome')
    else:
        st.error(f"File '{NAMA_FILE}' tidak ditemukan di folder ini. Pastikan file csv kamu diletakkan satu folder dengan file app.py ini.")

# --- MENU 2: PREDIKSI AI ---
elif menu == "Prediksi AI":
    st.title("AI Medical Predictor Engine")
    st.write("Isi data klinis pasien di bawah ini untuk mendapatkan hasil analisis prediksi risiko diabetes.")
    
    if model is None:
        st.error(f"Model gagal dimuat karena file '{NAMA_FILE}' tidak ditemukan.")
    else:
        st.markdown("### 📋 Patient Vitals (Data Klinis Pasien)")
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies (Jumlah Kehamilan)", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose (Kadar Glukosa Darah)", min_value=0, max_value=300, value=110)
            blood_pressure = st.number_input("Blood Pressure (Tekanan Darah Diastolik)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness (Ketebalan Lipatan Kulit)", min_value=0, max_value=100, value=20)
            
        with col2:
            insulin = st.number_input("Insulin (Kadar Insulin Serum 2-Jam)", min_value=0, max_value=900, value=80)
            bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function (Riwayat Keturunan)", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age (Usia Pasien)", min_value=1, max_value=120, value=30)
            
        st.markdown("---")
        
        if st.button("Run AI Cardiac & Diabetes Analysis", type="primary"):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            input_scaled = scaler.transform(input_data)
            
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            st.subheader("📊 Hasil Analisis Kecerdasan Buatan")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if prediction == 1:
                    st.error(f"⚠️ **RISIKO TINGGI (DIABETES DETECTED)**")
                else:
                    st.success(f"✅ **RISIKO RENDAH (NO DIABETES DETECTED)**")
                    
            with res_col2:
                confidence = prediction_proba[1] * 100 if prediction == 1 else prediction_proba[0] * 100
                st.metric(label="AI Model Confidence Score", value=f"{confidence:.2f} %")

# --- MENU 3: INFORMASI DATASET ---
elif menu == "Informasi Dataset":
    st.title("Dataset Insights & Reference")
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.error(f"File '{NAMA_FILE}' tidak ditemukan.")