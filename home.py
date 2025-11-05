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
# --- Custom CSS (Styling) ---
# Menyuntikkan CSS kustom untuk mempercantik tampilan.
st.markdown("""
<style>
    /* Header Utama */
    .main-header {
        font-size: 3.2em;
        font-weight: bold;
        text-align: center;
        color: #1A73E8; /* Warna biru yang lebih segar */
        margin-bottom: 0.5em;
        letter-spacing: -1px;
    }
    
    /* Sub-Header */
    .sub-header {
        font-size: 1.6em;
        text-align: center;
        color: #444; /* Lebih gelap untuk kontras */
        margin-bottom: 2em;
        font-weight: 300;
    }
    
    /* Kotak Fitur (Info Cards) */
    .feature-box {
        background-color: #ffffff; /* Latar belakang putih bersih */
        padding: 2em;
        border-radius: 12px;
        margin: 1em 0;
        color: #333333;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06); /* Bayangan lebih lembut */
        transition: all 0.3s ease;
        min-height: 220px; /* Menetapkan tinggi minimum untuk keseragaman */
        display: flex;
        flex-direction: column;
        border: 1px solid #e0e0e0; /* Border halus */
    }
    
    .feature-box:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.1); /* Bayangan lebih jelas */
        transform: translateY(-5px); /* Efek terangkat */
        border-color: #1A73E8; /* Highlight border saat hover */
    }
    
    .feature-box h3 {
        color: #1A73E8; /* Warna judul fitur disamakan dengan header */
        margin-top: 0;
        font-size: 1.5em;
        margin-bottom: 0.5em;
    }

    .feature-box p {
        font-size: 1em;
        color: #555;
        line-height: 1.6;
    }

    /* Garis pemisah yang lebih halus */
    hr {
        border: none;
        height: 1px;
        background-color: #e0e0e0;
        margin: 2.5em 0; /* Jarak lebih besar */
    }

    /* Styling untuk sub-header h2 Markdown */
    h2 {
        color: #222;
        border-bottom: 3px solid #1A73E8;
        padding-bottom: 8px;
        margin-top: 1em;
    }

    /* Styling untuk container alur kerja */
    .stContainer {
        border-radius: 10px;
        margin-bottom: 1em;
    }
    .stContainer h3 {
        margin-top: 0.25em;
        color: #1A73E8;
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

# Menggunakan st.container untuk setiap langkah agar lebih rapi dan modern
with st.container(border=True):
    st.markdown("### 1. Upload CV Anda")
    st.markdown("Sistem akan secara otomatis membaca, mengekstrak, dan memetakan kompetensi utama dari CV Anda.")

with st.container(border=True):
    st.markdown("### 2. AI Mapping & Verifikasi")
    st.markdown("Profil Anda akan dicocokkan dengan standar okupasi nasional (PON TIK) dan Anda dapat memverifikasi hasilnya.")

with st.container(border=True):
    st.markdown("### 3. Asesmen Kompetensi")
    st.markdown("Ambil tes yang dibuat secara otomatis oleh AI untuk memvalidasi tingkat keahlian Anda pada okupasi tersebut.")

with st.container(border=True):
    st.markdown("### 4. Rekomendasi Karier")
    st.markdown("Berdasarkan hasil asesmen, sistem akan memberikan rekomendasi lowongan pekerjaan dan pelatihan yang relevan.")

with st.container(border=True):
    st.markdown("### 5. Dashboard Nasional")
    st.markdown("Data Anda (secara anonim dan agregat) berkontribusi pada pemetaan SDM Digital nasional untuk analisis kebijakan.")
