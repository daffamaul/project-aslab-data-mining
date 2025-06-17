import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")

st.title("Aplikasi Prediksi Stres Mahasiswa")

menu = st.sidebar.selectbox("Navigasi", [
    "Upload & Data Awal",
    "Evaluasi Model",
    "Visualisasi Prediksi",
    "Analisis Fitur",
    "Simulasi What-If",
    "Segmentasi Mahasiswa",
    "Download Hasil"
])

uploaded_file = st.file_uploader("Upload Dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "stress_level" not in df.columns:
        st.error("Kolom 'stress_level' tidak ditemukan di dataset. Pastikan dataset memiliki kolom target tersebut.")
    else:
        # Preprocessing
        X = df.drop(columns=["stress_level"])
        y = df["stress_level"]
        X = X.select_dtypes(include='number')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if menu == "Upload & Data Awal":
            st.subheader("Data Awal")
            st.write(df.head())

        elif menu == "Evaluasi Model":
            st.subheader("Hasil Evaluasi Model")
            mse_val = mean_squared_error(y_test, y_pred)
            r2_val = r2_score(y_test, y_pred)
            st.write("Mean Squared Error (MSE):", mse_val)
            st.write("R-squared (R²):", r2_val)

            st.markdown("**Interpretasi:**")
            st.markdown(f"- MSE lebih rendah = error prediksi kecil. Nilai saat ini: `{mse_val:.2f}`.")
            st.markdown(f"- R² = `{r2_val:.2f}` berarti sekitar `{r2_val*100:.1f}%` variasi tingkat stres dapat dijelaskan oleh fitur-fitur input.")

        elif menu == "Visualisasi Prediksi":
            st.subheader("Visualisasi Prediksi vs Aktual")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            ax.set_xlabel("Nilai Aktual")
            ax.set_ylabel("Nilai Prediksi")
            ax.set_title("Prediksi vs Aktual")
            st.pyplot(fig)

        elif menu == "Analisis Fitur":
            st.subheader("Koefisien Regresi & Korelasi")
            coef_df = pd.DataFrame(model.coef_, index=X.columns, columns=["Koefisien"])
            st.write("Intercept:", model.intercept_)
            st.write(coef_df)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            coef_df.sort_values(by="Koefisien", ascending=True).plot(kind='barh', ax=ax2)
            ax2.set_title("Pengaruh Fitur terhadap Tingkat Stres")
            st.pyplot(fig2)

            most_impact = coef_df['Koefisien'].abs().idxmax()
            impact_value = coef_df.loc[most_impact, 'Koefisien']
            direction = "meningkatkan" if impact_value > 0 else "menurunkan"
            st.markdown(f"📌 Fitur `{most_impact}` memiliki pengaruh paling besar, dan dapat {direction} tingkat stres sebesar `{abs(impact_value):.2f}`.")

        elif menu == "Simulasi What-If":
            st.subheader("Simulasi What-If")
            selected_feature = st.selectbox("Pilih fitur:", X.columns)
            delta = st.slider("Perubahan nilai fitur (delta):", -10.0, 10.0, 1.0, step=0.5)
            X_sim = X_test.copy()
            X_sim[selected_feature] += delta
            y_sim = model.predict(X_sim)

            st.write(f"Rata-rata prediksi awal: {np.mean(y_pred):.2f}")
            st.write(f"Setelah perubahan: {np.mean(y_sim):.2f}")

            delta_effect = np.mean(y_sim) - np.mean(y_pred)
            if delta_effect > 0:
                st.markdown(f"📌 Perubahan meningkatkan rata-rata stres sebesar `{delta_effect:.2f}`")
            else:
                st.markdown(f"📌 Perubahan menurunkan rata-rata stres sebesar `{abs(delta_effect):.2f}`")
        
        elif menu == "Segmentasi Mahasiswa":
            st.subheader("Segmentasi Mahasiswa berdasarkan Pola Fitur")

            num_clusters = st.slider("Jumlah cluster:", 2, 6, 3)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            X_cluster = X.copy()
            X_cluster["Cluster"] = kmeans.fit_predict(X)

            st.write("Rata-rata fitur tiap cluster:")
            st.write(X_cluster.groupby("Cluster").mean())

            color_map = plt.cm.get_cmap('Set2', num_clusters)

            if X.shape[1] >= 2:
                fig, ax = plt.subplots()
                scatter = ax.scatter(
                    X.iloc[:, 0],
                    X.iloc[:, 1],
                    c=X_cluster["Cluster"],
                    cmap=color_map
                )
                ax.set_xlabel(X.columns[0])
                ax.set_ylabel(X.columns[1])
                ax.set_title("Visualisasi Cluster Mahasiswa")

                # Buat legend berdasarkan cluster
                handles = []
                for i in range(num_clusters):
                    handles.append(
                        plt.Line2D(
                            [0], [0],
                            marker='o',
                            color='w',
                            label=f"Cluster {i}",
                            markerfacecolor=color_map(i),
                            markersize=10
                        )
                    )
                ax.legend(handles=handles, title="Cluster")
                st.pyplot(fig)

            # Narasi interpretatif berdasarkan cluster
            st.markdown("### Narasi Interpretatif:")
            cluster_summary = X_cluster.groupby("Cluster").mean()

            for i in range(num_clusters):
                st.markdown(f"**🟢 Cluster {i}:**")
                dominant_feature = cluster_summary.iloc[i].sort_values(ascending=False).index[0]
                st.markdown(f"- Mahasiswa dalam kelompok ini cenderung menonjol pada fitur `{dominant_feature}`.")
                if "stress_level" in df.columns:
                    avg_stress = df.loc[X_cluster["Cluster"] == i, "stress_level"].mean()
                    st.markdown(f"- Rata-rata tingkat stres: `{avg_stress:.2f}`.")
                    if avg_stress >= 70:
                        st.markdown("- ⚠️ Indikasi kelompok dengan risiko stres tinggi. Perlu perhatian khusus.")
                    elif avg_stress <= 40:
                        st.markdown("- ✅ Kelompok ini menunjukkan kecenderungan stres rendah.")
                    else:
                        st.markdown("- ⚖️ Tingkat stres tergolong sedang.")
                else:
                    st.markdown("- Data stres tidak tersedia untuk narasi lebih lanjut.")

            st.markdown("### 🧠 Insight & Rekomendasi:")
            st.markdown("Dengan segmentasi ini, pendekatan individual bisa dikembangkan berdasarkan karakteristik setiap kelompok:")
            st.markdown("- 💼 Cluster dengan dominasi beban kerja bisa dibimbing manajemen waktu.")
            st.markdown("- 💤 Cluster dengan waktu istirahat rendah bisa diberi edukasi pentingnya tidur.")
            st.markdown("- 🙌 Cluster dengan stres rendah bisa dijadikan panutan/mentor dalam kegiatan kampus.")
        
        elif menu == "Download Hasil":
            st.subheader("Unduh Hasil Prediksi")
            result_df = X_test.copy()
            result_df["Actual_Stress"] = y_test.values
            result_df["Predicted_Stress"] = y_pred
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download sebagai CSV", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")
