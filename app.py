import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Rekomendasi Jurusan SMK Di SMPN 279 Jakarta Utara",
    page_icon="🎓",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.markdown("""
# 🎓 Dashboard Rekomendasi Jurusan SMK Di SMPN 279 Jakarta Utara
Sistem Pendukung Keputusan berbasis Machine Learning (Random Forest)
""")

st.write("---")

# =========================
# LOAD MODEL
# =========================
model = joblib.load('model_rekomendasi.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# =========================
# MAPPING + ANALISIS
# =========================
mapping_jurusan = {
    'AKL': 'Akuntansi dan Keuangan Lembaga',
    'OTKP': 'Otomatisasi dan Tata Kelola Perkantoran',
    'RPL': 'Rekayasa Perangkat Lunak',
    'TKJ': 'Teknik Komputer dan Jaringan',
    'TKR': 'Teknik Kendaraan Ringan'
}

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("📌 Input Nilai Siswa")

mtk = st.sidebar.number_input("Matematika", 0, 100)
ipa = st.sidebar.number_input("IPA", 0, 100)
ips = st.sidebar.number_input("IPS", 0, 100)
bhs = st.sidebar.number_input("Bahasa", 0, 100)

# =========================
# PREDIKSI
# =========================
if st.sidebar.button("🎯 Proses Rekomendasi"):

    data = np.array([[mtk, ipa, ips, bhs]])
    data_scaled = scaler.transform(data)

    pred = model.predict(data_scaled)
    hasil = le.inverse_transform(pred)[0]

    jurusan = mapping_jurusan[hasil]

    # =========================
    # MAIN DASHBOARD
    # =========================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Jurusan Rekomendasi", hasil)

    with col2:
        st.metric("Nama Jurusan", jurusan)

    with col3:
        st.metric("Status", "Selesai")

    st.write("---")

    # =========================
    # GRAFIK NILAI
    # =========================
    st.subheader("📊 Visualisasi Nilai Siswa")

    df_chart = pd.DataFrame({
        "Mata Pelajaran": ["MTK", "IPA", "IPS", "BHS"],
        "Nilai": [mtk, ipa, ips, bhs]
    })

    st.bar_chart(df_chart.set_index("Mata Pelajaran"))

    st.write("---")

    # =========================
    # ANALISIS CERDAS
    # =========================
    st.subheader("🧠 Analisis Sistem")

    nilai_dict = {
        "Matematika": mtk,
        "IPA": ipa,
        "IPS": ips,
        "Bahasa": bhs
    }

    tertinggi = max(nilai_dict, key=nilai_dict.get)

    if tertinggi in ["Matematika", "IPA"]:
        tipe = "Sains & Teknologi"
    elif tertinggi == "IPS":
        tipe = "Sosial & Bisnis"
    else:
        tipe = "Bahasa & Komunikasi"

    st.info(f"""
    🔍 Mata pelajaran tertinggi: **{tertinggi}**  
    🎯 Tipe siswa: **{tipe}**  
    🎓 Rekomendasi jurusan: **{jurusan}**
    """)

    st.success("✔ Analisis selesai dilakukan oleh sistem Machine Learning")

# =========================
# FOOTER
# =========================
st.write("---")
st.caption("© 2026 Sistem Rekomendasi Jurusan SMK - Machine Learning Project")
