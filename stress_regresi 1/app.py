import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("Aplikasi Prediksi Stres Mahasiswa Menggunakan Regresi Linier")

uploaded_file = st.file_uploader("Upload Dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Awal")
    st.write(df.head())

    if "stress_level" not in df.columns:
        st.error("Kolom 'stress_level' tidak ditemukan di dataset. Pastikan dataset memiliki kolom target tersebut.")
    else:
        # Pisahkan fitur dan target
        X = df.drop(columns=["stress_level"])
        y = df["stress_level"]

        # Gunakan hanya fitur numerik
        X = X.select_dtypes(include='number')

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        st.subheader("Hasil Evaluasi Model")
        st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
        st.write("R-squared (R²):", r2_score(y_test, y_pred))

        # Visualisasi
        st.subheader("Visualisasi Prediksi vs Aktual")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax.set_xlabel("Nilai Aktual")
        ax.set_ylabel("Nilai Prediksi")
        ax.set_title("Prediksi vs Aktual")
        st.pyplot(fig)

        # Tampilkan koefisien model
        st.subheader("Koefisien Regresi")
        coef_df = pd.DataFrame(model.coef_, index=X.columns, columns=["Koefisien"])
        st.write("Intercept:", model.intercept_)
        st.write(coef_df)
