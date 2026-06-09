import streamlit as st
import joblib
import numpy as np

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Rekomendasi Jurusan SMK Di SMPN 279 Jakarta Utara",
    page_icon="🎓",
    layout="centered"
)

# =========================
# HEADER
# =========================
st.markdown("""
<div style='text-align: center;'>
    <h1>🎓 Sistem Rekomendasi Jurusan SMK di SMPN 279 Jakarta Utara</h1>
    <p>Menentukan jurusan terbaik berdasarkan nilai akademik siswa</p>
</div>
""", unsafe_allow_html=True)

st.write("---")

# =========================
# LOAD MODEL
# =========================
model = joblib.load('model_rekomendasi.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# =========================
# MAPPING + DESKRIPSI
# =========================
mapping_jurusan = {
    'AKL': 'AKL (Akuntansi dan Keuangan Lembaga)',
    'OTKP': 'OTKP (Otomatisasi dan Tata Kelola Perkantoran)',
    'RPL': 'RPL (Rekayasa Perangkat Lunak)',
    'TKJ': 'TKJ (Teknik Komputer dan Jaringan)',
    'TKR': 'TKR (Teknik Kendaraan Ringan)'
}

deskripsi = {
    'AKL': 'Cocok untuk siswa yang suka akuntansi dan keuangan.',
    'OTKP': 'Cocok untuk administrasi dan perkantoran.',
    'RPL': 'Cocok untuk siswa yang tertarik pada coding dan software.',
    'TKJ': 'Cocok untuk jaringan komputer dan IT.',
    'TKR': 'Cocok untuk otomotif dan mesin.'
}

# =========================
# INPUT (PAKAI KOLOM BIAR RAPI)
# =========================
st.subheader("📊 Input Nilai Siswa")

col1, col2 = st.columns(2)

with col1:
    mtk = st.number_input("Matematika", 0, 100)
    ipa = st.number_input("IPA", 0, 100)

with col2:
    ips = st.number_input("IPS", 0, 100)
    bhs = st.number_input("Bahasa", 0, 100)

st.write("")

# =========================
# BUTTON
# =========================
if st.button("🎯 Prediksi Jurusan"):

    with st.spinner("⏳ Sedang menganalisis..."):
        
        data = np.array([[mtk, ipa, ips, bhs]])
        data_scaled = scaler.transform(data)
        prediksi = model.predict(data_scaled)
        hasil = le.inverse_transform(prediksi)

        jurusan = hasil[0]
        jurusan_lengkap = mapping_jurusan[jurusan]

    # =========================
    # OUTPUT (CARD STYLE)
    # =========================
    st.success("✅ Prediksi selesai!")

    st.markdown(f"""
    <div style="
        background-color:#f0f2f6;
        padding:20px;
        border-radius:10px;
        text-align:center;
    ">
        <h2>🎯 {jurusan_lengkap}</h2>
        <p>{deskripsi[jurusan]}</p>
    </div>
    """, unsafe_allow_html=True)
