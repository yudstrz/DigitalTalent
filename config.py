# -*- coding: utf-8 -*-
"""
config.py - Configuration File untuk Career Assistant AI

Instruksi:
1. Copy file ini dan rename menjadi 'config.py'
2. Ganti nilai-nilai di bawah sesuai dengan setup Anda
3. Simpan di folder yang sama dengan script utama
"""

import os

# ========================================
# DATABASE CONFIGURATION
# ========================================

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

# ========================================
# GEMINI AI CONFIGURATION
# ========================================

# Gemini API Key (Dapatkan dari: https://makersuite.google.com/app/apikey)
GEMINI_API_KEY = "AIzaSyCR8xgDIv5oYBaDmMyuGGWjqpFi7U8SGA4"

# Gemini API Configuration
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-flash-latest"

# ========================================
# MACHINE LEARNING CONFIGURATION
# ========================================

# Reinforcement Learning Parameters
RL_LEARNING_RATE = 0.1
RL_DISCOUNT_FACTOR = 0.9
RL_EPSILON = 0.2  # Exploration rate

# Recommendation System Parameters
DEFAULT_TOP_K = 8  # Jumlah rekomendasi default
CONTENT_WEIGHT = 0.4  # Bobot content-based filtering
RL_WEIGHT = 0.3  # Bobot RL score
Q_WEIGHT = 0.3  # Bobot Q-learning

# ========================================
# APPLICATION SETTINGS
# ========================================

# Jumlah soal default untuk asesmen
JUMLAH_SOAL = 5  # <-- TAMBAHKAN INI

# Cache TTL (Time To Live) dalam detik
CACHE_TTL = 600  # 10 menit

# Session timeout (dalam menit)
SESSION_TIMEOUT = 30

# Maximum chat history to keep
MAX_CHAT_HISTORY = 50

# ========================================
# FEATURE FLAGS
# ========================================

# Enable/Disable features
ENABLE_COLLABORATIVE_FILTERING = True
ENABLE_GEMINI_AI = True
ENABLE_EXPORT_PROFILE = True
ENABLE_ANALYTICS = True

# ========================================
# DEBUGGING
# ========================================

# Set to True untuk menampilkan debug information
DEBUG_MODE = False

# Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = 'INFO'
