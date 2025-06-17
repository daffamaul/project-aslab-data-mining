import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediksi Stres Mahasiswa", layout="centered")

st.title("📈 Prediksi Tingkat Stres Mahasiswa")

try:
    df = pd.read_csv("StressLevelDataset.csv")  # pastikan file ini ada di folder yang sama
except FileNotFoundError:
    st.error("Oh no! tidak ada dataset untuk dilatih. Hubungi administrator")
    st.stop()

if "stress_level" not in df.columns:
    st.error("Dataset tidak memiliki kolom 'stress_level'.")
else:
    # Pisahkan fitur dan target
    X = df.drop(columns=["stress_level"])
    y = df["stress_level"]

    # Hanya gunakan fitur numerik
    X = X.select_dtypes(include="number")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Melatih model regresi
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Mapping nama kolom ke label bahasa Indonesia
    label_mapping = {
        "anxiety_level": "Tingkat Kecemasan",
        "self_esteem": "Percaya Diri",
        "mental_health_history": "Riwayat Kesehatan Mental",
        "depression": "Depresi",
        "headache": "Sakit Kepala",
        "blood_pressure": "Tekanan Darah",            
        "sleep_quality": "Kualitas Tidur",
        "breathing_problem": "Permasalahan Pernafasan",
        "noise_level": "Tingkat Kebisingan",
        "living_conditions": "Kondisi kehidupan",
        "safety": "Rasa Aman",
        "basic_needs": "Kebutuhan Dasar Terpenuhi",
        "study_load": "Beban Studi",
        "teacher_student_relationship": "Hubungan dengan Dosen",
        "future_career_concerns": "Kecemasan Karir",
        "social_support": "Dukungan Lingkungan",
        "peer_pressure": "Tekanan Teman Sebaya",
        "extracurricular_activities": "Aktivitas Ekstrakulikuler",
        "bullying": "Penindasan",
        "family_support": "Dukungan Keluarga",
        "academic_performance": "Performa Akademik",
        "social_interaction": "Interaksi Sosial"
    }        

    # Koefisien dari model linear
    coefficients = model.coef_
    feature_names = X.columns
    means = X.mean()

    kontribusi_rata2 = {
        label_mapping.get(f, f.replace("_", " ").capitalize()): means[f] * coef
        for f, coef in zip(feature_names, coefficients)
    }

    # Buat dataframe dan urutkan berdasarkan pengaruh (langsung saat membuat)
    kontribusi_df = pd.DataFrame(
        sorted(kontribusi_rata2.items(), key=lambda x: x[1], reverse=True),
        columns=["Faktor", "Pengaruh"]
    )

    st.markdown("""
    ### 🧠 Tentang Model Prediksi

    Model yang digunakan untuk memprediksi tingkat stres pada mahasiswa adalah **Regresi Linear (Linear Regression)**.  
    Model ini bekerja dengan mengukur seberapa besar pengaruh masing-masing faktor (seperti kecemasan, tekanan teman sebaya, kualitas tidur, dan lainnya) terhadap tingkat stres berdasarkan data yang diberikan.

    Semakin besar nilai koefisien dari suatu faktor, maka faktor tersebut cenderung memiliki pengaruh lebih besar terhadap stres mahasiswa.  
    Tabel di bawah ini menampilkan faktor-faktor tersebut diurutkan berdasarkan kontribusinya terhadap hasil prediksi stres, dihitung dari nilai rata-rata dataset dan bobot model.
    """)

    st.subheader("📌 Faktor yang Paling Mempengaruhi Stres (berdasarkan data rata-rata)")
    st.markdown("Berikut ini adalah estimasi seberapa besar pengaruh rata-rata masing-masing faktor terhadap tingkat stres, berdasarkan model:")
    st.dataframe(kontribusi_df, use_container_width=True)

    st.subheader("🧾 Input Data Mahasiswa")
    user_data = {}
    
    for col in X.columns:
        label = label_mapping.get(col, col.replace("_", " ").capitalize())
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        col_median = float(df[col].median())
        user_data[col] = st.slider(
            label,
            min_value=col_min,
            max_value=col_max,
            value=col_median,
            step=0.1
        )

    if st.button("🔮 Prediksi Tingkat Stres"):
        input_df = pd.DataFrame([user_data])
        pred = model.predict(input_df)[0]

        # Interpretasi narasi yang lebih bermakna
        st.subheader("📊 Hasil Prediksi")

        st.metric("Prediksi Nilai Stres", f"{pred:.2f}")

        if pred <= 0.5:
            kategori = "Sangat Rendah"
            narasi = "Mahasiswa hampir tidak mengalami stres. Kondisi mental dan lingkungan belajar terjaga sangat baik."
        elif pred <= 1.0:
            kategori = "Rendah"
            narasi = "Mahasiswa menunjukkan tingkat stres yang tergolong rendah. Tetap pertahankan manajemen waktu dan keseimbangan hidup."
        elif pred <= 1.5:
            kategori = "Sedang"
            narasi = "Tingkat stres mahasiswa tergolong sedang. Perlu mulai memperhatikan kebiasaan harian dan mencari dukungan sosial atau keluarga."
        elif pred <= 2.0:
            kategori = "Tinggi"
            narasi = "Mahasiswa mengalami stres yang cukup tinggi. Disarankan untuk berbicara dengan konselor atau mengambil waktu untuk istirahat mental."
        else:
            kategori = "Sangat Tinggi"
            narasi = "Tingkat stres mahasiswa sangat tinggi. Sangat disarankan segera berkonsultasi dengan profesional untuk penanganan lebih lanjut."

        st.success(f"Tingkat Stres: {kategori}")            
        st.markdown(f"🧠 {narasi}")

