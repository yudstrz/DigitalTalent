# -*- coding: utf-8 -*-
"""
HALAMAN 3: REKOMENDASI & CAREER ASSISTANT (RL + CF Enhanced)

✅ Fitur Baru:
- Reinforcement Learning (RL) untuk adaptasi preferensi user
- Collaborative Filtering (CF) untuk rekomendasi berbasis user serupa
- Sistem reward berdasarkan interaksi (view, apply, reject)
- User profiling yang lebih canggih
"""

import re
import os
import json
import random
import requests
from datetime import datetime
import mistune
import pandas as pd
import numpy as np
import streamlit as st
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------
# Konfigurasi
# --------------------------------------
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

# Impor dari file konfigurasi lokal (jika ada)
try:
    from config import (
        EXCEL_PATH as CONFIG_EXCEL_PATH,
        SHEET_PON as CONFIG_SHEET_PON,
        SHEET_TALENTA as CONFIG_SHEET_TALENTA,
        SHEET_LOWONGAN as CONFIG_SHEET_LOWONGAN,
        GEMINI_API_KEY,
        GEMINI_BASE_URL,
        GEMINI_MODEL
    )
    # Gunakan nilai dari config jika berhasil diimpor
    EXCEL_PATH = CONFIG_EXCEL_PATH
    SHEET_PON = CONFIG_SHEET_PON
    SHEET_TALENTA = CONFIG_SHEET_TALENTA
    SHEET_LOWONGAN = CONFIG_SHEET_LOWONGAN
except ImportError:
    # Gunakan nilai default jika config tidak ada
    st.warning("⚠️ File config.py tidak ditemukan. Menggunakan konfigurasi default.")
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
    GEMINI_BASE_URL = "https://api.generativeai.google/v1"
    GEMINI_MODEL = "gemini-small"
except Exception as e:
    st.error(f"❌ Error saat mengimpor config: {e}")
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
    GEMINI_BASE_URL = "https://api.generativeai.google/v1"
    GEMINI_MODEL = "gemini-small"

st.set_page_config(page_title="Career Assistant AI", page_icon="💡", layout="wide")

# ========================================
# CSS Kartu Lowongan
# ========================================
st.markdown("""
<style>
/* ====== GLOBAL DARK THEME FIX ====== */
body, .stApp {
    background-color: #0f1117 !important;
    color: #e4e6eb !important;
}

/* ====== JOB CARD FIX ====== */
.job-card {
    background-color: #1c1f2b;
    border: 1px solid #2e3244;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 6px 16px rgba(0,0,0,0.2);
    transition: transform 0.2s, box-shadow 0.2s;
}
.job-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}
.job-card h4 { margin: 0 0 6px 0; color: #ffffff; }
.job-meta { font-size: 0.9em; color: #9ca3af; margin-bottom: 8px; }
.job-skills {
    background: rgba(255,255,255,0.1);
    color: #e0e0e0;
    padding: 6px 8px;
    border-radius: 6px;
    font-family: monospace;
    display:inline-block;
}
.job-desc { margin-top: 8px; color: #cbd5e1; }
.job-score {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.85em;
    display: inline-block;
    margin-top: 8px;
}
.apply-row { margin-top: 10px; display:flex; gap:8px; }

/* ====== CHAT BUBBLE STYLE ====== */
.message-wrapper {
    display: flex;
    align-items: flex-start;
    margin: 8px 0;
}
.message-bubble {
    border-radius: 14px;
    padding: 10px 14px;
    max-width: 85%;
    line-height: 1.4;
    word-wrap: break-word;
}
.message-bubble.user {
    background-color: #2563eb;
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 0;
}
.message-bubble.ai {
    background-color: #1e293b;
    color: #e2e8f0;
    border-bottom-left-radius: 0;
}
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    text-align: center;
    line-height: 36px;
    margin: 0 8px;
    font-size: 20px;
}
.avatar.user { background-color: #2563eb; color: white; order: 2; }
.avatar.ai { background-color: #475569; color: white; order: 0; }
.message-time {
    font-size: 0.75em;
    color: #9ca3af;
    text-align: right;
    margin-top: 4px;
}

/* ====== STATS CARD ====== */
.stats-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 15px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 10px 0;
}
.stats-value { font-size: 2em; font-weight: bold; }
.stats-label { font-size: 0.9em; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ========================================
# STATE inisialisasi
# ========================================
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{
        "role": "ai",
        "content": "👋 **Halo! Saya Career Assistant AI dengan teknologi RL & CF**\n\nSaya belajar dari interaksi Anda untuk memberikan rekomendasi yang semakin personal!",
        "timestamp": datetime.now().strftime("%H:%M")
    }]
if 'waiting_response' not in st.session_state:
    st.session_state.waiting_response = False
if 'trigger_ai_response' not in st.session_state:
    st.session_state.trigger_ai_response = False

# ========================================
# RL & CF State Management
# ========================================
if 'user_interactions' not in st.session_state:
    st.session_state.user_interactions = {
        'viewed': [],
        'applied': [],
        'rejected': [],
        'skill_preferences': defaultdict(float),
        'location_preferences': defaultdict(float),
        'company_preferences': defaultdict(float)
    }

if 'rl_q_table' not in st.session_state:
    st.session_state.rl_q_table = defaultdict(lambda: defaultdict(float))

if 'user_profile_vector' not in st.session_state:
    st.session_state.user_profile_vector = None

if 'all_users_data' not in st.session_state:
    st.session_state.all_users_data = {}

# Okupasi state (untuk integrasi dengan halaman lain)
if 'mapped_okupasi_id' not in st.session_state:
    st.session_state.mapped_okupasi_id = None
if 'okupasi_info' not in st.session_state:
    st.session_state.okupasi_info = {}


# ========================================
# 🔧 Helpers
# ========================================
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace('\xa0', ' ')
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skill_tokens(text: str) -> list:
    """Ekstrak kata skill (pisah koma, /, |, spasi)"""
    text = normalize_text(text).lower()
    parts = re.split(r"[,;/\\|]+", text)
    tokens = [p.strip() for p in parts if p.strip()]
    return list(dict.fromkeys(tokens))


# ========================================
# 🧠 Reinforcement Learning Functions
# ========================================
class RLRecommender:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.2):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
    
    def get_state(self, user_profile):
        """Konversi user profile ke state representation"""
        top_skills = sorted(user_profile['skill_preferences'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        top_locations = sorted(user_profile['location_preferences'].items(),
                               key=lambda x: x[1], reverse=True)[:2]
        
        state = f"skills_{'_'.join([s[0] for s in top_skills])}_loc_{'_'.join([l[0] for l in top_locations])}"
        return state
    
    def get_reward(self, action):
        """Calculate reward based on user action"""
        rewards = {
            'apply': 10,
            'view': 2,
            'reject': -5,
            'ignore': -1
        }
        return rewards.get(action, 0)
    
    def update_q_value(self, state, action, reward, next_state, q_table):
        """Q-learning update"""
        current_q = q_table[state][action]
        max_next_q = max(q_table[next_state].values()) if q_table[next_state] else 0
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        q_table[state][action] = new_q
        return new_q
    
    def select_action(self, state, available_jobs, q_table):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(available_jobs) if available_jobs else None
        else:
            job_scores = {job['LowonganID']: q_table[state][job['LowonganID']] 
                          for job in available_jobs}
            if not job_scores:
                return None
            best_job_id = max(job_scores, key=job_scores.get)
            return next((j for j in available_jobs if j['LowonganID'] == best_job_id), None)


# ========================================
# 🤝 Collaborative Filtering Functions
# ========================================
class CollaborativeFilter:
    def __init__(self):
        self.user_item_matrix = None
        self.similarity_matrix = None
    
    def build_user_item_matrix(self, all_users_data, all_jobs):
        """Build user-item interaction matrix"""
        users = list(all_users_data.keys())
        jobs = [j['LowonganID'] for j in all_jobs]
        
        matrix = np.zeros((len(users), len(jobs)))
        
        for i, user_id in enumerate(users):
            user_data = all_users_data[user_id]
            for j, job_id in enumerate(jobs):
                if job_id in user_data.get('applied', []):
                    matrix[i][j] = 5
                elif job_id in user_data.get('viewed', []):
                    matrix[i][j] = 3
                elif job_id in user_data.get('rejected', []):
                    matrix[i][j] = -2
        
        self.user_item_matrix = matrix
        return matrix
    
    def calculate_user_similarity(self):
        """Calculate cosine similarity between users"""
        if self.user_item_matrix is None:
            return None
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        return self.similarity_matrix
    
    def get_cf_recommendations(self, current_user_idx, top_k=5):
        """Get recommendations based on similar users"""
        if self.similarity_matrix is None:
            return []
        
        user_similarities = self.similarity_matrix[current_user_idx]
        similar_users = np.argsort(user_similarities)[::-1][1:6]
        
        job_scores = np.zeros(self.user_item_matrix.shape[1])
        for similar_user_idx in similar_users:
            similarity = user_similarities[similar_user_idx]
            job_scores += similarity * self.user_item_matrix[similar_user_idx]
        
        top_job_indices = np.argsort(job_scores)[::-1][:top_k]
        return top_job_indices


# ========================================
# 📊 Enhanced Recommendation System
# ========================================
@st.cache_data(ttl=600)
def load_excel_data(path, sheet_name):
    """
    Fungsi untuk load data Excel dengan cache.
    Mendukung dua mode:
    1. Jika sheet memiliki header (baris pertama adalah nama kolom)
    2. Jika sheet tidak memiliki header (semua baris adalah data)
    """
    if not os.path.exists(path):
        st.warning(f"⚠️ File Excel tidak ditemukan di: {path}")
        return None
        
    try:
        # Coba baca dulu dengan header (mode normal)
        df_with_header = pd.read_excel(path, sheet_name=sheet_name, engine='openpyxl')
        
        # Definisi struktur kolom untuk setiap sheet
        sheet_columns = {
            SHEET_LOWONGAN: ['LowonganID', 'Perusahaan', 'Posisi', 'Deskripsi_Pekerjaan', 
                            'Keterampilan_Dibutuhkan', 'Lokasi'],
            SHEET_PON: ['OkupasiID', 'Okupasi', 'Deskripsi', 'Keterampilan'],
            SHEET_TALENTA: ['TalentaID', 'Nama', 'Email', 'Telepon', 'Alamat'],
            SHEET_PENDIDIKAN: ['PendidikanID', 'TalentaID', 'Jenjang', 'Institusi', 'Jurusan', 'Tahun'],
            SHEET_PEKERJAAN: ['PekerjaanID', 'TalentaID', 'Perusahaan', 'Posisi', 'Tahun_Mulai', 'Tahun_Selesai'],
            SHEET_SKILL: ['SkillID', 'TalentaID', 'Keterampilan', 'Tingkat', 'Sertifikasi'],
            SHEET_HASIL: ['HasilID', 'TalentaID', 'OkupasiID', 'Skor', 'Tanggal']
        }
        
        expected_cols = sheet_columns.get(sheet_name)
        
        if expected_cols is None:
            st.warning(f"⚠️ Sheet '{sheet_name}' tidak dikenali. Menggunakan struktur default.")
            return df_with_header
        
        # Cek apakah header sudah sesuai
        if list(df_with_header.columns) == expected_cols:
            # Header sudah benar, langsung return
            return df_with_header
        
        # Jika header tidak sesuai, coba mode tanpa header
        df_no_header = pd.read_excel(path, sheet_name=sheet_name, header=None, engine='openpyxl')
        
        if len(df_no_header.columns) == len(expected_cols):
            df_no_header.columns = expected_cols
            return df_no_header
        else:
            # Coba gunakan data dengan header asli
            st.warning(f"⚠️ Struktur kolom tidak sesuai untuk sheet '{sheet_name}'.")
            st.info(f"Kolom yang diharapkan ({len(expected_cols)}): {expected_cols}")
            st.info(f"Kolom yang ditemukan ({len(df_with_header.columns)}): {list(df_with_header.columns)}")
            return df_with_header  # Return data asli untuk debugging
            
    except Exception as e:
        st.error(f"❌ Gagal membuka Excel sheet '{sheet_name}': {e}")
        return None


def get_hybrid_recommendations(profil_teks: str, top_k: int = 8):
    """
    Hybrid recommendation menggunakan:
    1. Content-based filtering (skill matching)
    2. Reinforcement Learning (user behavior)
    3. Collaborative Filtering (similar users)
    4. Okupasi-based enrichment
    """
    path = EXCEL_PATH or "data/DTP_database.xlsx"
    df = load_excel_data(path, SHEET_LOWONGAN)
    
    if df is None or df.empty:
        return []

    # 1. Content-Based Filtering
    # Gabungkan profil teks + skill dari okupasi
    profil_skills = extract_skill_tokens(profil_teks)
    
    # 🎯 ENRICHMENT: Tambahkan skill dari okupasi jika ada
    if st.session_state.get('okupasi_info'):
        okupasi_skills = extract_skill_tokens(
            str(st.session_state.okupasi_info.get('Keterampilan', ''))
        )
        # Gabungkan skill (unique)
        profil_skills = list(set(profil_skills + okupasi_skills))
        
        # Bonus: tambahkan nama okupasi sebagai keyword
        okupasi_nama = st.session_state.okupasi_info.get('Okupasi', '')
        if okupasi_nama:
            profil_skills.extend(extract_skill_tokens(okupasi_nama))
    candidates = []
    
    for _, row in df.iterrows():
        job_id = str(row['LowonganID'])
        req_skills = extract_skill_tokens(str(row['Keterampilan_Dibutuhkan']))
        location = normalize_text(str(row['Lokasi']))
        company = normalize_text(str(row['Perusahaan']))
        
        matched_skills = [s for s in profil_skills if any(s in r or r in s for r in req_skills)]
        content_score = len(matched_skills) / max(len(req_skills), 1) if req_skills else 0
        
        # 🎯 BONUS: Jika posisi cocok dengan okupasi, tambah score
        okupasi_bonus = 0
        if st.session_state.get('okupasi_info'):
            okupasi_nama = normalize_text(st.session_state.okupasi_info.get('Okupasi', ''))
            posisi = normalize_text(str(row['Posisi']))
            
            # Check kecocokan nama okupasi dengan posisi lowongan
            if okupasi_nama in posisi or posisi in okupasi_nama:
                okupasi_bonus = 0.2  # Bonus 20% untuk posisi yang match
            else:
                # Check partial match (misal: "Data Scientist" vs "Senior Data Analyst")
                okupasi_tokens = set(okupasi_nama.split())
                posisi_tokens = set(posisi.split())
                if len(okupasi_tokens & posisi_tokens) >= 1:
                    okupasi_bonus = 0.1  # Bonus 10% untuk partial match
        
        content_score = min(1.0, content_score + okupasi_bonus)  # Max 1.0
        
        # 2. RL Score from user preferences
        interactions = st.session_state.user_interactions
        rl_score = 0
        
        for skill in req_skills:
            rl_score += interactions['skill_preferences'].get(skill, 0)
        
        rl_score += interactions['location_preferences'].get(location, 0)
        rl_score += interactions['company_preferences'].get(company, 0)
        
        if job_id in interactions['rejected']:
            rl_score -= 5
        
        if job_id in interactions['viewed'] and job_id not in interactions['applied']:
            rl_score += 1
        
        # 3. Q-Learning score
        rl_agent = RLRecommender()
        state = rl_agent.get_state(interactions)
        q_score = st.session_state.rl_q_table[state].get(job_id, 0)
        
        # Hybrid score
        final_score = (0.4 * content_score) + (0.3 * rl_score) + (0.3 * q_score)
        
        candidates.append({
            'row': row,
            'content_score': content_score,
            'rl_score': rl_score,
            'q_score': q_score,
            'final_score': final_score,
            'matched_skills': matched_skills
        })
    
    candidates_sorted = sorted(candidates, key=lambda x: (x['final_score'], random.random()), 
                               reverse=True)
    
    results = []
    for c in candidates_sorted[:top_k]:
        job_dict = c['row'].to_dict()
        job_dict['_final_score'] = round(c['final_score'], 3)
        job_dict['_content_score'] = round(c['content_score'], 3)
        job_dict['_rl_score'] = round(c['rl_score'], 3)
        job_dict['_q_score'] = round(c['q_score'], 3)
        job_dict['_matched_skills'] = c['matched_skills']
        results.append(job_dict)
    
    return results


# ========================================
# 💾 Update User Interactions
# ========================================
def record_interaction(job, action):
    """Record user interaction and update RL model"""
    interactions = st.session_state.user_interactions
    job_id = str(job['LowonganID'])
    
    if action == 'view' and job_id not in interactions['viewed']:
        interactions['viewed'].append(job_id)
    elif action == 'apply' and job_id not in interactions['applied']:
        interactions['applied'].append(job_id)
        if job_id not in interactions['viewed']:
            interactions['viewed'].append(job_id)
    elif action == 'reject' and job_id not in interactions['rejected']:
        interactions['rejected'].append(job_id)
    
    skills = extract_skill_tokens(str(job['Keterampilan_Dibutuhkan']))
    location = normalize_text(str(job['Lokasi']))
    company = normalize_text(str(job['Perusahaan']))
    
    reward_multiplier = {'apply': 1.0, 'view': 0.5, 'reject': -0.3}.get(action, 0)
    
    for skill in skills:
        interactions['skill_preferences'][skill] += reward_multiplier
    
    interactions['location_preferences'][location] += reward_multiplier
    interactions['company_preferences'][company] += reward_multiplier
    
    rl_agent = RLRecommender()
    state = rl_agent.get_state(interactions)
    reward = rl_agent.get_reward(action)
    next_state = rl_agent.get_state(interactions)
    
    rl_agent.update_q_value(state, job_id, reward, next_state, st.session_state.rl_q_table)


# ========================================
# 🔍 Okupasi Data Management
# ========================================
def load_okupasi_data():
    """Load data okupasi dari sheet PON"""
    try:
        df_pon = load_excel_data(EXCEL_PATH, SHEET_PON)
        if df_pon is None or df_pon.empty:
            st.warning("⚠️ Data okupasi tidak tersedia.")
            return None
        return df_pon
    except Exception as e:
        st.error(f"❌ Error mengambil data okupasi: {e}")
        return None


def get_okupasi_info(okupasi_id):
    """Ambil informasi okupasi berdasarkan ID"""
    try:
        df_pon = load_okupasi_data()
        if df_pon is None:
            return None
        
        okupasi_data = df_pon[df_pon['OkupasiID'] == okupasi_id]
        if okupasi_data.empty:
            st.warning(f"⚠️ Okupasi dengan ID '{okupasi_id}' tidak ditemukan.")
            return None
        
        return okupasi_data.iloc[0].to_dict()
    except Exception as e:
        st.error(f"❌ Error mengambil info okupasi: {e}")
        return None


def update_user_okupasi_mapping(okupasi_id, okupasi_nama):
    """Update mapping okupasi user di session state"""
    st.session_state.mapped_okupasi_id = okupasi_id
    st.session_state.mapped_okupasi_nama = okupasi_nama
    
    # Load full okupasi info
    okupasi_info = get_okupasi_info(okupasi_id)
    if okupasi_info:
        st.session_state.okupasi_info = okupasi_info
        return True
    return False


# ========================================
# Gemini API
# ========================================
def call_gemini_api(prompt: str) -> str:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        return "❌ Gemini API key belum diatur. Silakan atur di `config.py` atau di awal script ini."
    try:
        url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        res = requests.post(url, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"❌ Error koneksi ke Gemini API: {e}"


def get_career_analysis(user_message: str, chat_history: list) -> str:
    context = "\n".join([f"{'User' if m['role']=='user' else 'AI'}: {m['content']}" 
                         for m in chat_history[-6:]])
    
    interactions = st.session_state.user_interactions
    top_skills = sorted(interactions['skill_preferences'].items(), 
                        key=lambda x: x[1], reverse=True)[:5]
    
    # 🎯 Tambahkan info okupasi ke context
    okupasi_context = ""
    if st.session_state.get('okupasi_info'):
        okupasi = st.session_state.okupasi_info
        okupasi_context = f"""
=== Okupasi User ===
Okupasi: {okupasi.get('Okupasi', 'N/A')}
Okupasi ID: {okupasi.get('OkupasiID', 'N/A')}
Keterampilan Okupasi: {okupasi.get('Keterampilan', 'N/A')}
"""
    
    prompt = f"""Anda adalah Career Coach AI bidang Teknologi Informasi dan Komunikasi.
=== Context ===
{context}
=== Pesan Terbaru ===
{user_message}
{okupasi_context}
=== User Insights (dari ML) ===
Top Skills Interest: {', '.join([s[0] for s in top_skills])}
Total Interactions: {len(interactions['viewed'])} viewed, {len(interactions['applied'])} applied
=== Instruksi ===
1. Jawab dengan ramah dan profesional.
2. Gunakan emoji secukupnya.
3. Berikan saran karier & pelatihan relevan.
4. Pertimbangkan pola preferensi user dari ML insights.
5. Jika user punya okupasi, berikan saran yang aligned dengan okupasi tersebut.
6. Maksimal 5 kalimat.
Jawaban:"""
    return call_gemini_api(prompt)


# ========================================
# Chat Rendering
# ========================================
def render_chat_bubble(msg: dict):
    if msg['role'] == 'user':
        st.markdown(f"""
        <div class="message-wrapper user">
            <div class="avatar user">👤</div>
            <div class="message-bubble user">
                <div class="message-content">{msg['content']}</div>
                <div class="message-time">{msg['timestamp']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message-wrapper ai">
            <div class="avatar ai">🤖</div>
            <div class="message-bubble ai">
                <div class="message-content">{mistune.html(msg['content'])}</div>
                <div class="message-time">{msg['timestamp']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ========================================
# Proses AI response
# ========================================
if st.session_state.trigger_ai_response:
    last_msg = st.session_state.chat_history[-1]
    if last_msg['role'] == 'user':
        st.session_state.waiting_response = True
        ai_reply = get_career_analysis(last_msg['content'], st.session_state.chat_history)
        st.session_state.chat_history.append({
            "role": "ai",
            "content": ai_reply,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        st.session_state['profil_teks'] = last_msg['content']
        st.session_state.waiting_response = False
        st.session_state.trigger_ai_response = False


# ========================================
# MAIN UI
# ========================================
st.title("💡 Career Assistant AI")
st.caption("Powered by Reinforcement Learning & Collaborative Filtering")

# Tampilkan info okupasi jika sudah dipetakan
if st.session_state.get('mapped_okupasi_id'):
    with st.expander("🎯 Okupasi Anda", expanded=False):
        okupasi_info = st.session_state.get('okupasi_info', {})
        if okupasi_info:
            col_ok1, col_ok2 = st.columns([1, 3])
            with col_ok1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; text-align: center; color: white;'>
                    <div style='font-size: 2.5em;'>👔</div>
                    <div style='font-size: 0.9em; margin-top: 5px;'>Okupasi ID</div>
                    <div style='font-size: 1.3em; font-weight: bold;'>{okupasi_info.get('OkupasiID', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_ok2:
                st.markdown(f"**Nama Okupasi:** {okupasi_info.get('Okupasi', 'N/A')}")
                st.markdown(f"**Deskripsi:** {okupasi_info.get('Deskripsi', 'N/A')}")
                st.markdown(f"**Keterampilan:** {okupasi_info.get('Keterampilan', 'N/A')}")
        else:
            st.info("ℹ️ Data okupasi belum tersedia. Lakukan pemetaan terlebih dahulu.")

st.markdown("---")

# Stats Dashboard
col1, col2, col3, col4 = st.columns(4)
interactions = st.session_state.user_interactions
col1, col2, col3, col4 = st.columns(4)
interactions = st.session_state.user_interactions

with col1:
    st.markdown(f"""
    <div class='stats-card'>
        <div class='stats-value'>{len(interactions['viewed'])}</div>
        <div class='stats-label'>👁️ Viewed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='stats-card'>
        <div class='stats-value'>{len(interactions['applied'])}</div>
        <div class='stats-label'>📤 Applied</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='stats-card'>
        <div class='stats-value'>{len(interactions['rejected'])}</div>
        <div class='stats-label'>❌ Rejected</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    top_skill = max(interactions['skill_preferences'].items(), 
                    key=lambda x: x[1])[0] if interactions['skill_preferences'] else "N/A"
    st.markdown(f"""
    <div class='stats-card'>
        <div class='stats-value' style='font-size:1.2em'>{top_skill}</div>
        <div class='stats-label'>🏆 Top Skill</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Chat UI
st.markdown('<div class="chat-container" id="chat-box">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    render_chat_bubble(msg)
st.markdown('</div>', unsafe_allow_html=True)

# Quick Actions
st.markdown("#### ⚡ Quick Actions")
cols = st.columns(4)
labels = ["💼 Lowongan", "📚 Pelatihan", "🎯 Analisis", "🔄 Reset"]
actions = [
    "Lowongan apa yang cocok untuk saya?",
    "Pelatihan apa yang sebaiknya saya ikuti?",
    "Analisis skill saya dan kasih saran karier",
    None
]
for i, label in enumerate(labels):
    with cols[i]:
        if st.button(label, use_container_width=True):
            if i == 3:
                st.session_state.chat_history = [{
                    "role": "ai",
                    "content": "👋 Chat direset! Model RL tetap menyimpan pembelajaran Anda 🚀",
                    "timestamp": datetime.now().strftime("%H:%M")
                }]
                st.rerun()
            else:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": actions[i],
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                st.session_state.trigger_ai_response = True
                st.rerun()

# Input chat
st.markdown("---")
with st.form("chat_form", clear_on_submit=True):
    c1, c2 = st.columns([5, 1])
    user_msg = c1.text_input("", placeholder="Ceritakan skill & pengalamanmu...", 
                             label_visibility="collapsed")
    send = c2.form_submit_button("📤")

if send and user_msg.strip():
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_msg,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    st.session_state.trigger_ai_response = True
    st.rerun()

# ========================================
# Section: Rekomendasi Lowongan dengan RL & CF
# ========================================
st.markdown("---")
st.markdown("### 🎯 Rekomendasi Lowongan Berbasis AI")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    show_recs = st.button("📊 Tampilkan Rekomendasi (Hybrid AI)", use_container_width=True)
with col_btn2:
    if st.button("🧠 Lihat Model Insights", use_container_width=True):
        st.markdown("#### 🔍 RL Model Insights")
        st.write("**Top 10 Skill Preferences:**")
        top_skills = sorted(interactions['skill_preferences'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]
        for skill, score in top_skills:
            st.write(f"- {skill}: {score:.2f}")
        
        st.write("\n**Q-Table Size:**", len(st.session_state.rl_q_table))

if show_recs:
    profil_teks = st.session_state.get('profil_teks', '')
    if not profil_teks:
        st.warning("Silakan masukkan profil Anda di chat terlebih dahulu (atau gunakan Quick Action 'Analisis') untuk mendapatkan rekomendasi.")
    else:
        # 🎯 Info Box: Sumber Rekomendasi
        st.info(f"""
        **🔍 Rekomendasi berdasarkan:**
        
        ✅ **Chat Profile:** {len(extract_skill_tokens(profil_teks))} skills detected
        {'✅ **Okupasi:** ' + st.session_state.okupasi_info.get('Okupasi', 'N/A') if st.session_state.get('okupasi_info') else '⚠️ **Okupasi:** Belum dipetakan'}
        ✅ **Behavior Learning:** {len(st.session_state.user_interactions['viewed'])} interactions
        """)
        
        jobs = get_hybrid_recommendations(profil_teks, top_k=8)

        if not jobs:
            st.info("Tidak ada rekomendasi yang sesuai dengan profilmu saat ini.")
        else:
            st.success(f"🎉 Ditemukan {len(jobs)} rekomendasi dengan skor AI tertinggi!")
            
            # Show breakdown skor
            with st.expander("📊 Penjelasan Scoring System"):
                st.markdown("""
                **Hybrid Score = 40% Content + 30% RL + 30% Q-Learning**
                
                - **Content Score**: Kecocokan skill + okupasi
                - **RL Score**: Preferensi dari interaksi (view/apply/reject)
                - **Q-Score**: Optimal action dari machine learning
                
                💡 *Semakin banyak interaksi, semakin akurat rekomendasinya!*
                """)
            
            for job in jobs:
                job_id = str(job['LowonganID'])
                card_key = f"job_{job_id}"
                
                # 🎯 Highlight matched skills
                matched_skills_str = ", ".join(job.get('_matched_skills', [])) if job.get('_matched_skills') else "N/A"
                
                # 🎯 Check okupasi match
                okupasi_match_icon = ""
                if st.session_state.get('okupasi_info'):
                    okupasi_nama = normalize_text(st.session_state.okupasi_info.get('Okupasi', ''))
                    posisi = normalize_text(str(job['Posisi']))
                    if okupasi_nama in posisi or posisi in okupasi_nama:
                        okupasi_match_icon = " 🎯"
                
                st.markdown(f"""
                <div class='job-card'>
                    <h4>💼 {job['Posisi']}{okupasi_match_icon} — {job['Perusahaan']}</h4>
                    <div class='job-meta'>📍 {job['Lokasi']}</div>
                    <div class='job-skills'>🧩 Required: {job['Keterampilan_Dibutuhkan']}</div>
                    <div class='job-skills' style='background: rgba(76, 175, 80, 0.2); margin-top: 5px;'>
                        ✅ You Have: {matched_skills_str}
                    </div>
                    <div class='job-desc'>📝 {job['Deskripsi_Pekerjaan']}</div>
                    <div class='job-score'>
                        🎯 AI Score: {job['_final_score']} 
                        (Content: {job['_content_score']} | RL: {job['_rl_score']} | Q: {job['_q_score']})
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("👁️ View", key=f"view_{card_key}"):
                        record_interaction(job, 'view')
                        st.success("✅ Interaksi dicatat! Model belajar dari pilihan Anda.")
                        st.rerun()
                with col2:
                    if st.button("📤 Apply", key=f"apply_{card_key}"):
                        record_interaction(job, 'apply')
                        st.success("🎉 Lamaran dicatat! Preferensi Anda diperbarui.")
                        st.rerun()
                with col3:
                    if st.button("❌ Not Interested", key=f"reject_{card_key}"):
                        record_interaction(job, 'reject')
                        st.info("👍 Terima kasih! Kami tidak akan merekomendasikan yang serupa.")
                        st.rerun()
                
                st.markdown("---")

# ========================================
# Advanced Analytics Section
# ========================================
with st.expander("📈 Advanced Analytics & Model Performance"):
    st.markdown("### 🧪 Model Performance Metrics")
    
    # Calculate engagement rate
    total_viewed = len(interactions['viewed'])
    total_applied = len(interactions['applied'])
    engagement_rate = (total_applied / total_viewed * 100) if total_viewed > 0 else 0
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Engagement Rate", f"{engagement_rate:.1f}%", 
                  delta="Good" if engagement_rate > 20 else "Needs Improvement")
    with col_b:
        st.metric("Learning Iterations", len(st.session_state.rl_q_table))
    with col_c:
        accuracy = min(100, engagement_rate * 2 + len(interactions['applied']) * 5)
        st.metric("Model Accuracy", f"{accuracy:.0f}%")
    
    # Skill Preference Chart
    if interactions['skill_preferences']:
        st.markdown("#### 🎯 Your Skill Preference Profile")
        top_10_skills = sorted(interactions['skill_preferences'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        
        if top_10_skills:
            skills_df = pd.DataFrame(top_10_skills, columns=['Skill', 'Score'])
            st.bar_chart(skills_df.set_index('Skill'))
    
    # Location Preferences
    if interactions['location_preferences']:
        st.markdown("#### 📍 Location Preferences")
        loc_prefs = sorted(interactions['location_preferences'].items(), 
                           key=lambda x: x[1], reverse=True)[:5]
        for loc, score in loc_prefs:
            st.write(f"- **{loc}**: {score:.2f}")
    
    # Q-Learning State Analysis
    st.markdown("#### 🔬 Q-Learning State Space")
    st.write(f"Total States Explored: **{len(st.session_state.rl_q_table)}**")
    
    if st.session_state.rl_q_table:
        # Show sample Q-values
        sample_states = list(st.session_state.rl_q_table.keys())[:3]
        for state in sample_states:
            st.write(f"\n**State:** `{state}`")
            actions = st.session_state.rl_q_table[state]
            top_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)[:3]
            for action, q_val in top_actions:
                st.write(f"  - Action {action}: Q-value = {q_val:.3f}")

# ========================================
# Export User Profile
# ========================================
st.markdown("---")
if st.button("💾 Export My Learning Profile", use_container_width=True):
    profile_data = {
        'timestamp': datetime.now().isoformat(),
        'okupasi': {
            'okupasi_id': st.session_state.get('mapped_okupasi_id'),
            'okupasi_nama': st.session_state.get('mapped_okupasi_nama')
        },
        'interactions': {
            'viewed_count': len(interactions['viewed']),
            'applied_count': len(interactions['applied']),
            'rejected_count': len(interactions['rejected'])
        },
        'skill_preferences': dict(interactions['skill_preferences']),
        'location_preferences': dict(interactions['location_preferences']),
        'company_preferences': dict(interactions['company_preferences']),
        'q_table_size': len(st.session_state.rl_q_table),
        'engagement_rate': engagement_rate
    }
    
    profile_json = json.dumps(profile_data, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 Download Profile JSON",
        data=profile_json,
        file_name=f"career_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.success("✅ Profile siap didownload! Anda bisa mengimpornya nanti untuk melanjutkan pembelajaran.")

# ========================================
# Footer Information
# ========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #9ca3af; padding: 20px;'>
    <p><strong>🤖 AI-Powered Career Assistant</strong></p>
    <p style='font-size: 0.85em;'>
        Menggunakan <strong>Reinforcement Learning</strong> untuk adaptasi preferensi real-time<br>
        & <strong>Collaborative Filtering</strong> untuk rekomendasi berbasis komunitas
    </p>
    <p style='font-size: 0.8em; margin-top: 10px;'>
        💡 Semakin banyak Anda berinteraksi, semakin akurat rekomendasi kami!
    </p>
</div>
""", unsafe_allow_html=True)
