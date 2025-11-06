# pages/2_Asesmen_Kompetensi.py
"""
HALAMAN 2: ASESMEN KOMPETENSI (VERSI ADAPTIF)

Halaman ini mengimplementasikan Use Case 3 & 4:
1.  Memverifikasi bahwa pengguna telah melengkapi profil.
2.  Menginisialisasi 'Adaptive Assessment Engine' (Use Case 4).
3.  Membuat soal SATU PER SATU secara dinamis (Use Case 3).
4.  Menampilkan 1 soal, pengguna menjawab, lalu sistem menilai.
5.  Berdasarkan hasil, sistem menentukan level soal berikutnya (Probing Loop).
6.  Setelah selesai, menghitung skor akhir dan menentukan level.
"""

import streamlit as st
import pandas as pd
import json
import re
import requests
import datetime
import traceback
import time # Untuk simulasi 'typing'
import mistune
# Impor dari file konfigurasi lokal
try:
    from config import (
        EXCEL_PATH, SHEET_PON,
        GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL,
        JUMLAH_SOAL
    )
except ImportError:
    st.error("Gagal mengimpor 'config.py'. Pastikan file tersebut ada.")
    # Definisikan nilai default agar aplikasi tidak crash
    EXCEL_PATH = "data_pon.xlsx"
    SHEET_PON = "pon_tik"
    GEMINI_API_KEY = "YOUR_API_KEY"
    GEMINI_BASE_URL = "https.generativelace.google.com/v1beta"
    GEMINI_MODEL = "gemini-pro"
    JUMLAH_SOAL = 5 # Default jika config gagal

# ========================================
# KONFIGURASI HALAMAN
# ========================================
st.set_page_config(
    page_title="Asesmen Adaptif",
    page_icon="🤖",
    layout="wide"
)

# ========================================
# FUNGSI 1: LOAD EXCEL (Tidak Berubah)
# ========================================
@st.cache_data
def load_excel_sheet(file_path, sheet_name):
    """
    Membaca sheet dari file Excel.
    Menggunakan cache data Streamlit untuk performa.
    Header dimulai dari baris pertama (baris ke-1).
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)  # ← ubah header=0
        df.columns = df.columns.str.strip()
        df = df.fillna('')
        return df
    except Exception as e:
        st.error(f"Gagal memuat sheet '{sheet_name}': {e}")
        return None


# ========================================
# FUNGSI 2: CALL GEMINI API (Sedikit Modifikasi)
# ========================================
def call_gemini_api(prompt: str) -> str:
    """
    Mengirim request ke Gemini API dan mengembalikan respons teks.
    """
    url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    # Instruksi format JSON yang ketat untuk AI
    # DIUBAH: Sekarang hanya meminta SATU soal, bukan list.
    json_instruction = """
PENTING: Respons HARUS berupa satu objek JSON yang valid dan lengkap.
Format yang DIWAJIBKAN (HANYA SATU SOAL, BUKAN LIST):
{
  "id": "q1", 
  "teks": "Teks pertanyaan studi kasus...", 
  "opsi": ["Opsi A", "Opsi B", "Opsi C", "Opsi D"], 
  "jawaban_benar": "Opsi A"
}
"""
    
    payload = {
        "contents": [{"parts": [{"text": prompt + json_instruction}]}],
        "generationConfig": {
            "temperature": 0.8, # Sedikit lebih kreatif untuk variasi soal
            "maxOutputTokens": 2048
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status() 
        result = response.json()
        
        # Ekstrak teks konten
        content = result['candidates'][0]['content']['parts'][0]['text']
        
        # Membersihkan markdown "fence"
        content = content.strip().lstrip("```json").lstrip("```").rstrip("```")
        content = content.strip()
        
        return content
        
    except Exception as e:
        st.error(f"Error saat memanggil Gemini API: {e}")
        with st.expander("Detail Error API"):
            st.code(traceback.format_exc())
        raise Exception(f"Error calling Gemini: {e}")

# ========================================
# FUNGSI 3: SANITIZE JSON (Tidak Berubah)
# ========================================
def sanitize_json_response(text: str) -> str:
    """
    Membersihkan string JSON mentah dari AI.
    """
    text = re.sub(r'\\(?![ntr"\\/bfuU])', '', text)
    text = text.replace("\\'", "'")
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\}\s*\{', '},{', text)
    text = re.sub(r'\]\s*"', '],"', text)
    return text.strip()

# ========================================
# FUNGSI 4: GENERATE ADAPTIVE QUESTION (BARU - Use Case 3 & 4)
# ========================================
def generate_adaptive_question(okupasi_info: dict, level: str, history_text: list):
    """
    Merakit prompt untuk SATU soal adaptif, memanggil AI, 
    mem-parsing, dan memvalidasi soal.
    
    Args:
        okupasi_info (dict): Data dari PON TIK (Okupasi, UK, Keywords).
        level (str): Level kesulitan yang diminta ('Junior', 'Menengah', 'Ahli').
        history_text (list): List dari teks soal sebelumnya (untuk menghindari duplikat).
        
    Returns:
        dict: Dictionary satu soal, atau raise Exception jika gagal.
    """
    okupasi_nama = okupasi_info['Okupasi']
    unit_kompetensi = okupasi_info['Unit_Kompetensi']
    kuk_keywords = okupasi_info['Kuk_Keywords']
    
    # 1. Buat prompt engineering (Task-Oriented Prompt)
    prompt = f"""
Anda adalah seorang Asesor Kompetensi TIK profesional di Indonesia.

Tugas Anda adalah membuat TEPAT 1 (SATU) soal asesmen pilihan ganda 
untuk menguji seorang kandidat pada:

**Okupasi:** {okupasi_nama}
**Unit Kompetensi Terkait:** {unit_kompetensi}
**Keterampilan Kunci (Keywords):** {kuk_keywords}

**TARGET LEVEL SAAT INI:** {level}
- Jika 'Junior': Buat soal konseptual atau definisi dasar.
- Jika 'Menengah': Buat soal studi kasus praktis sederhana.
- Jika 'Ahli': Buat soal skenario kompleks, analisis, atau troubleshooting mendalam.

**Kriteria Soal:**
1.  **Relevansi:** Soal harus sangat relevan dengan okupasi DAN level di atas.
2.  **Bentuk Soal:** Berikan skenario praktis, bukan hanya definisi (kecuali untuk Junior).
3.  **Opsi:** Harus ada TEPAT 4 opsi pilihan ganda.
4.  **Jawaban:** Harus ada TEPAT 1 jawaban yang benar.
5.  **Bahasa:** Gunakan Bahasa Indonesia yang formal dan profesional.
6.  **Karakter:** HINDARI penggunaan karakter backslash (\\), newline (\\n), atau petik ganda (") di dalam teks soal atau opsi. Gunakan petik tunggal (') jika perlu.
7.  **BARU/UNIK:** Soal ini HARUS BERBEDA dari soal-soal sebelumnya:
    {history_text}

**Format JSON (WAJIB SATU OBJEK):**
(Selanjutnya akan ditambahkan instruksi JSON oleh fungsi pemanggil)
"""

    try:
        # 2. Panggil API
        response_text = call_gemini_api(prompt)
        
        # 3. Bersihkan dan parse JSON
        response_text = sanitize_json_response(response_text)
        question = json.loads(response_text)
        
        # 4. Validasi struktur (karena ini hanya 1 objek)
        if not isinstance(question, dict):
            raise ValueError("AI tidak mengembalikan objek JSON.")
            
        if not all(k in question for k in ["teks", "opsi", "jawaban_benar"]):
            raise ValueError("Soal memiliki struktur tidak lengkap.")
        
        if len(question.get("opsi", [])) != 4:
            # Coba perbaiki jika opsi hanya 3
            if len(question["opsi"]) == 3:
                question["opsi"].append("Semua jawaban salah")
            else:
                raise ValueError(f"Soal tidak memiliki TEPAT 4 opsi.")
        
        if question.get("jawaban_benar") not in question.get("opsi", []):
            st.warning(f"Jawaban '{question['jawaban_benar']}' tidak ada di opsi. Memilih opsi pertama sebagai default.")
            question["jawaban_benar"] = question["opsi"][0]
            
        # 5. Tambahkan metadata level
        question["level_ditargetkan"] = level
        
        return question
        
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing JSON dari AI: {e}\n\nTeks Respons: {response_text[:1000]}"
        st.error(error_msg)
        raise Exception(error_msg)
        
    except Exception as e:
        st.error(f"Error pada proses generate soal adaptif: {e}")
        st.code(traceback.format_exc())
        raise

# ========================================
# FUNGSI 5: TENTUKAN LEVEL BERIKUTNYA (Use Case 4)
# ========================================
def get_next_level(current_level: str, is_correct: bool) -> str:
    """
    Rule-based engine untuk menentukan level soal berikutnya.
    Ini adalah inti dari 'Probing Loop'.
    """
    if is_correct:
        # Probing NAIK
        if current_level == "Junior":
            return "Menengah"
        elif current_level == "Menengah":
            return "Ahli"
        elif current_level == "Ahli":
            return "Ahli" # Tetap di level tertinggi
    else:
        # Probing TURUN
        if current_level == "Ahli":
            return "Menengah"
        elif current_level == "Menengah":
            return "Junior"
        elif current_level == "Junior":
            return "Junior" # Tetap di level terendah
    return "Menengah" # Fallback

# ========================================
# FUNGSI 6: HITUNG LEVEL FINAL
# ========================================
def calculate_final_level(skor: int) -> str:
    """Tentukan level final berdasarkan skor akhir."""
    if skor >= 90:
        return "Ahli (Expert)"
    elif skor >= 70:
        return "Menengah (Intermediate)"
    elif skor >= 50:
        return "Junior (Junior)"
    else:
        return "Pemula (Beginner)"

# ========================================
# --- UI STREAMLIT DIMULAI ---
# ========================================

st.title("Asesmen Kompetensi Adaptif")

# ========================================
# VALIDASI USER
# ========================================
if not st.session_state.get('talent_id'):
    st.error("Anda harus mengisi dan menyimpan **Profil Talenta** di halaman 1 terlebih dahulu.")
    st.warning("Silakan kembali ke halaman 'Profil Talenta'.")
    st.stop()

# ========================================
# INISIALISASI SESSION STATE (PENTING!)
# ========================================
# Inisialisasi ini HANYA berjalan sekali saat halaman dibuka
if 'assessment_started' not in st.session_state:
    st.session_state.assessment_started = True
    st.session_state.current_question_index = 0
    st.session_state.current_level = "Menengah" # Mulai dari tengah
    st.session_state.correct_count = 0
    st.session_state.question_history = [] # Menyimpan (soal, jawaban_user, benar/salah)
    st.session_state.current_question = None # Menyimpan soal yang sedang ditampilkan
    st.session_state.assessment_complete = False
    st.session_state.waiting_for_answer = True # Flag untuk UI

# ========================================
# INFO TALENTA & OKUPASI
# ========================================
st.info(f"Login sebagai: **{st.session_state.talent_id}**")

# Muat data okupasi sekali saja
if 'okupasi_info' not in st.session_state:
    try:
        df_pon = load_excel_sheet(EXCEL_PATH, SHEET_PON)
        if df_pon is None:
            st.error("Gagal memuat data PON. Tidak bisa melanjutkan.")
            st.stop()
        
        okupasi_data = df_pon[df_pon['OkupasiID'] == st.session_state.mapped_okupasi_id]
        
        if okupasi_data.empty:
            st.error(f"Data Okupasi {st.session_state.mapped_okupasi_id} tidak ditemukan.")
            st.stop()
            
        st.session_state.okupasi_info = okupasi_data.iloc[0].to_dict()
    except Exception as e:
        st.error(f"Error mengambil data okupasi: {e}")
        st.stop()

st.header(f"Asesmen untuk: {st.session_state.okupasi_info['Okupasi']}")
st.markdown("---")

# ========================================
# ALUR ASESMEN ADAPTIF (Use Case 4)
# ========================================

# Tampilkan UI berdasarkan status
if st.session_state.assessment_complete:
    # --- 1. TAMPILKAN HASIL AKHIR ---
    st.success("Asesmen Selesai!")
    st.balloons()
    
    skor = int((st.session_state.correct_count / JUMLAH_SOAL) * 100)
    level = calculate_final_level(skor)
    
    st.subheader("Hasil Validasi Kompetensi:")
    col1, col2 = st.columns(2)
    col1.metric("Skor Anda", f"{skor}/100")
    col2.metric("Level Kompetensi", level)
    
    # Simpan hasil akhir ke session state
    st.session_state.assessment_score = skor
    st.session_state.assessment_level = level
    st.session_state.assessment_date = datetime.datetime.now()
    
    # Tampilkan detail jawaban
    with st.expander("Lihat Detail Jawaban"):
        for i, (q, user_ans, is_correct) in enumerate(st.session_state.question_history):
            if is_correct:
                st.success(f"""
                **Soal {i+1} (Level: {q['level_ditargetkan']})**: {q['teks']}\n
                **Jawaban Anda:** {user_ans} (Benar)
                """)
            else:
                st.error(f"""
                **Soal {i+1} (Level: {q['level_ditargetkan']})**: {q['teks']}\n
                **Jawaban Anda:** {user_ans} (Kurang Tepat)\n
                **Jawaban Tepat:** {q['jawaban_benar']}
                """)
    st.info("Silakan lanjut ke halaman **Rekomendasi**.")
    
    if st.button("Ulangi Asesmen"):
        # Reset state
        st.session_state.assessment_started = False
        del st.session_state.assessment_started
        st.rerun()

else:
    # --- 2. JALANKAN PROSES ASESMEN ---
    st.subheader(f"Pertanyaan {st.session_state.current_question_index + 1} dari {JUMLAH_SOAL}")
    st.caption(f"Target Level Soal Saat Ini: **{st.session_state.current_level}**")
    
    # Cek apakah kita perlu men-generate soal baru
    if st.session_state.current_question is None:
        try:
            placeholder = st.empty()
            with placeholder.status("AI sedang membuat soal..."):
                st.write(f"Membuat soal Level: {st.session_state.current_level}...")
                
                # Kumpulkan teks soal sebelumnya untuk menghindari duplikat
                history_texts = [q['teks'] for q, _, _ in st.session_state.question_history]
                
                new_question = generate_adaptive_question(
                    st.session_state.okupasi_info,
                    st.session_state.current_level,
                    history_texts
                )
                st.session_state.current_question = new_question
                st.session_state.waiting_for_answer = True
                placeholder.empty() # Hapus status spinner
        except Exception as e:
            st.error(f"Gagal men-generate soal: {e}. Coba muat ulang halaman.")
            st.stop()

    # Tampilkan soal yang sudah di-generate
    q = st.session_state.current_question
    if q:
        st.markdown(f"**{q['teks']}**")
        
        # Gunakan st.radio (bukan di dalam form)
        user_answer = st.radio(
            "Pilih satu jawaban:", 
            q['opsi'], 
            key=f"q_{q['id']}",
            index=None # Wajibkan pengguna memilih
        )

        submit_button = st.button("Jawab & Lanjutkan", disabled=(user_answer is None))

        if submit_button and st.session_state.waiting_for_answer:
            st.session_state.waiting_for_answer = False # Mencegah double submit
            
            # --- Ini adalah inti PROBING LOOP ---
            is_correct = (user_answer == q['jawaban_benar'])
            
            # 1. Tampilkan hasil
            if is_correct:
                st.success("Jawaban Anda Benar!")
                st.session_state.correct_count += 1
            else:
                st.error(f"Jawaban kurang tepat. Jawaban yang benar adalah: **{q['jawaban_benar']}**")
            
            time.sleep(1) # Beri jeda agar pengguna bisa membaca feedback
            
            # 2. Simpan histori
            st.session_state.question_history.append( (q, user_answer, is_correct) )
            
            # 3. Tentukan level berikutnya (Use Case 4)
            next_level = get_next_level(st.session_state.current_level, is_correct)
            st.session_state.current_level = next_level
            
            # 4. Maju ke soal berikutnya
            st.session_state.current_question_index += 1
            st.session_state.current_question = None # Hapus soal, agar di-generate ulang
            
            # 5. Cek apakah asesmen selesai
            if st.session_state.current_question_index >= JUMLAH_SOAL:
                st.session_state.assessment_complete = True
            
            # 6. Rerun aplikasi untuk menampilkan soal baru / hasil akhir
            st.rerun()
