# Home.py
"""
Modul Halaman Utama (Landing Page)

Halaman ini berfungsi sebagai "pintu depan" atau lobi sistem.
User yang pertama kali masuk akan melihat halaman ini.

Fungsi:
1. Menyambut user.
2. Menjelaskan sistem secara umum.
3. Mengarahkan user ke halaman yang sesuai.
"""

import streamlit as st

# --- Konfigurasi Halaman ---
# Mengatur konfigurasi dasar halaman Streamlit.
st.set_page_config(
    page_title="Digital Talent Platform",
    page_icon="🌐",  # Mengganti ikon emoji dengan yang lebih netral
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Custom CSS (Styling) ---
# Menyuntikkan CSS kustom untuk mempercantik tampilan.
st.markdown("""
<style>
    /* Header Utama */
    .main-header {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        color: #1E88E5; /* Warna biru primer */
        margin-bottom: 0.5em;
    }
    
    /* Sub-Header */
    .sub-header {
        font-size: 1.5em;
        text-align: center;
        color: #555; /* Sedikit lebih gelap untuk kontras */
        margin-bottom: 2em;
    }
    
    /* Kotak Fitur (Info Cards) */
    .feature-box {
        background-color: #f0f2f6; /* Latar belakang abu-abu sangat muda */
        padding: 1.5em;
        border-radius: 10px;
        margin: 1em 0;
        color: #333333;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05); /* Bayangan halus */
        transition: box-shadow 0.3s ease-in-out, transform 0.2s ease;
        height: 100%; /* Memastikan tinggi yang sama untuk kolom */
    }
    
    .feature-box:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.1); /* Bayangan lebih jelas saat hover */
        transform: translateY(-3px); /* Efek sedikit terangkat */
    }
    
    .feature-box h3 {
        color: #1E88E5; /* Warna judul fitur disamakan dengan header */
        margin-top: 0;
    }

</style>
""", unsafe_allow_html=True)


# --- Bagian Header ---
st.markdown('<p class="main-header">Digital Talent Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Platform AI-Powered untuk Pemetaan dan Validasi Talenta Digital Indonesia</p>', unsafe_allow_html=True)

st.markdown("---")


# --- Penjelasan Sistem ---
st.markdown("## Apa itu Digital Talent Platform?")

st.markdown("""
**Digital Talent Platform (DTP)** adalah sistem berbasis AI yang dirancang untuk menjembatani kesenjangan antara talenta digital dan kebutuhan industri. Platform ini membantu:

* **Talenta Digital**: Mendapatkan validasi kompetensi objektif dan rekomendasi karier yang terpersonalisasi.
* **Perusahaan**: Menemukan talenta yang paling sesuai dengan kebutuhan spesifik dan standar kompetensi yang jelas.
* **Pemerintah**: Melakukan monitoring dan pemetaan ketersediaan SDM digital secara nasional.
* **Lembaga Edukasi**: Merancang kurikulum yang relevan dengan data kebutuhan industri secara *real-time*.

Sistem ini mengadopsi **PON TIK (Profil Okupasi Nasional Teknologi Informasi dan Komunikasi)** sebagai acuan standar kompetensi utama.
""")

st.markdown("---")


# --- Fitur Utama ---
st.markdown("## Fitur Utama")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h3>AI-Powered Mapping</h3>
        <p>AI menganalisis CV Anda dan memetakan keahlian Anda ke okupasi PON TIK yang relevan menggunakan teknologi <i>semantic search</i>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
        <h3>Asesmen Kompetensi</h3>
        <p>AI men-generate soal asesmen secara dinamis sesuai dengan okupasi yang terpetakan. Dapatkan validasi kompetensi dengan sistem scoring yang objektif.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h3>Rekomendasi Karier</h3>
        <p>Dapatkan rekomendasi lowongan pekerjaan dan program pelatihan yang paling sesuai untuk mengisi <i>skill gap</i> dan mengembangkan profil profesional Anda.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-box">
        <h3>Dashboard Nasional</h3>
        <p>Visualisasi data agregat untuk pemangku kepentingan, menampilkan wawasan seperti distribusi okupasi, sebaran talenta, dan <i>skill gap</i> nasional.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# --- Alur Kerja Sistem ---
st.markdown("## Alur Kerja Sistem")

# Menggunakan daftar bernomor standar untuk tampilan yang lebih bersih
st.markdown("""
1.  **Upload CV Anda**: Sistem akan secara otomatis membaca, mengekstrak, dan memetakan kompetensi utama dari CV Anda.
2.  **AI Mapping & Verifikasi**: Profil Anda akan dicocokkan dengan standar okupasi nasional (PON TIK) dan Anda dapat memverifikasi hasilnya.
3.  **Asesmen Kompetensi**: Ambil tes yang dibuat secara otomatis oleh AI untuk memvalidasi tingkat keahlian Anda pada okupasi tersebut.
4.  **Rekomendasi Karier**: Berdasarkan hasil asesmen, sistem akan memberikan rekomendasi lowongan pekerjaan dan pelatihan yang relevan.
5.  **Dashboard Nasional**: Data Anda (secara anonim dan agregat) berkontribusi pada pemetaan SDM Digital nasional untuk analisis kebijakan.
""")
