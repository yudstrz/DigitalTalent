# -*- coding: utf-8 -*-
"""
HALAMAN 1: PROFIL TALENTA (UPLOAD CV & PEMETAAN SEMANTIK)
"""

import streamlit as st
import pandas as pd
import re
import io

st.title("Profil Talenta (CV & Pemetaan Semantik)")

# 1️⃣ Upload CV
uploaded_file = st.file_uploader("Unggah file CV Anda (PDF atau TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    st.write(f"**File diunggah:** {file_name}")

    # 2️⃣ Ekstraksi teks CV
    if file_name.endswith(".txt"):
        raw_cv = uploaded_file.read().decode("utf-8")
    elif file_name.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        raw_cv = " ".join([page.extract_text() for page in reader.pages])
    else:
        raw_cv = ""

    st.text_area("Isi CV Anda (hasil ekstraksi):", raw_cv[:1500] + "...", height=200)

    # 3️⃣ Simulasi hasil pemetaan semantik (contoh sederhana)
    def map_to_okupasi(cv_text):
        text = cv_text.lower()
        if "data" in text and "python" in text:
            return "Ilmuwan Data"
        elif "network" in text or "infrastruktur" in text:
            return "Administrator Jaringan"
        elif "design" in text:
            return "Desainer Grafis"
        else:
            return "Okupasi Umum"

    okupasi_nama = map_to_okupasi(raw_cv)

    st.success(f"Hasil pemetaan semantik: **{okupasi_nama}**")

    # 4️⃣ Simpan hasil ke session state (lama + tambahan baru)
    st.session_state.profile_text = raw_cv
    st.session_state.mapped_okupasi_nama = okupasi_nama

    # 🔗 Tambahan baru agar tersambung otomatis ke Career Assistant
    st.session_state['profil_teks'] = raw_cv
    st.session_state['profil_sumber'] = "cv_semantik"
    st.session_state['profil_okupasi'] = okupasi_nama

    # 5️⃣ Tampilkan tombol navigasi opsional
    st.info("Data profil Anda telah disimpan. Silakan lanjut ke halaman 'Rekomendasi & Career Assistant'.")
