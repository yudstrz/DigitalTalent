# config.py
"""
Konfigurasi sistem - hanya berisi path dan konstanta
"""

import os

# Path ke file Excel database
EXCEL_PATH = os.path.join("data", "DTP_Database.xlsx")

# Nama-nama sheet di Excel
SHEET_TALENTA = "Talenta"
SHEET_PENDIDIKAN = "Riwayat_Pendidikan"
SHEET_PEKERJAAN = "Riwayat_Pekerjaan"
SHEET_SKILL = "Keterampilan_Sertifikasi"
SHEET_PON = "PON_TIK_Master"
SHEET_LOWONGAN = "Lowongan_Industri"
SHEET_HASIL = "Hasil_Pemetaan_Asesmen"

# API Configuration
GEMINI_API_KEY = "AIzaSyCR8xgDIv5oYBaDmMyuGGWjqpFi7U8SGA4"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-flash-latest"

# Konstanta
JUMLAH_SOAL = 5
