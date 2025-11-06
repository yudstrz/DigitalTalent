# Impor pustaka standar
import re
import io
import os
import traceback
import pickle  # BARU: Untuk menyimpan data frame yang sudah diproses

# Impor pustaka pihak ketiga
import pandas as pd
import streamlit as st
import docx
from PyPDF2 import PdfReader
import faiss  # BARU: Database Vektor
from sentence_transformers import SentenceTransformer  # BARU: Model Embedding

# Impor dari file konfigurasi lokal
try:
    from config import EXCEL_PATH, SHEET_PON, SHEET_TALENTA
except ImportError:
    st.error("Gagal mengimpor 'config.py'. Pastikan file tersebut ada.")
    EXCEL_PATH = "data_pon.xlsx"
    SHEET_PON = "pon_tik"
    SHEET_TALENTA = "talenta"

# ========================================
# FUNGSI 1: MEMUAT EXCEL
# ========================================

def load_excel_sheet(file_path, sheet_name):
    """
    Membaca satu sheet dari file Excel dengan penanganan error yang lengkap.
    Menyesuaikan jika header ada di baris pertama (seperti contoh PON_TIK_Master).
    """
    try:
        if not os.path.exists(file_path):
            st.error(f"File tidak ditemukan: '{file_path}'")
            return None

        xls = pd.ExcelFile(file_path)
        if sheet_name not in xls.sheet_names:
            st.error(f"Sheet '{sheet_name}' tidak ditemukan!")
            st.info(f"Sheet yang tersedia adalah: {', '.join(xls.sheet_names)}")
            return None

        # ✅ Header di baris pertama
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)

        # ✅ Bersihkan nama kolom
        df.columns = df.columns.astype(str).str.replace('\xa0', '').str.strip()
        df = df.fillna('')

        # Debug singkat opsional
        # st.write("Kolom terbaca:", list(df.columns))

        return df

    except Exception as e:
        st.error(f"Error saat memuat sheet '{sheet_name}': {str(e)}")
        with st.expander("Detail Error Traceback"):
            st.code(traceback.format_exc())
        return None



# ========================================
# FUNGSI 2: EKSTRAK TEKS CV
# ========================================

def extract_text_from_pdf(file_io):
    """Ekstrak teks lengkap dari objek file PDF."""
    reader = PdfReader(file_io)
    return "".join([page.extract_text() for page in reader.pages if page.extract_text()])


def extract_text_from_docx(file_io):
    """Ekstrak teks lengkap dari objek file DOCX."""
    doc = docx.Document(file_io)
    return "\n".join([p.text for p in doc.paragraphs if p.text])


# ========================================
# FUNGSI 3: PARSING CV (Regex)
# ========================================

def parse_cv_data(cv_text):
    """
    Mengekstrak informasi dasar (email, nama, linkedin, lokasi) dari 
    teks mentah CV menggunakan regular expressions.
    """
    data = {
        "email": "",
        "nama": "",
        "linkedin": "",
        "lokasi": "",
        "cv_text": cv_text  # Menyimpan teks lengkap
    }
    
    # Ekstrak Email
    if match := re.search(r'[\w\.-]+@[\w\.-]+\.\w+', cv_text):
        data["email"] = match.group(0)

    # Ekstrak LinkedIn
    if match := re.search(r'linkedin\.com/in/([\w-]+)', cv_text, re.IGNORECASE):
        data["linkedin"] = f"https://www.linkedin.com/in/{match.group(1)}"

    # Ekstrak Nama (Heuristik: baris pertama, bukan email, maks 4 kata)
    first_line = cv_text.split('\n')[0].strip()
    if first_line and '@' not in first_line and len(first_line.split()) < 5:
        data["nama"] = first_line.title()
        
    # Ekstrak Lokasi (Heuristik: daftar kota besar di Indonesia)
    if match := re.search(
        r'(Jakarta|Bandung|Surabaya|Yogyakarta|Jogja|Medan|Semarang|Makassar|Denpasar|Palembang)', 
        cv_text, 
        re.IGNORECASE
    ):
        lokasi = match.group(0).title()
        if lokasi == "Jogja":
            lokasi = "Yogyakarta"
        data["lokasi"] = lokasi
    
    return data


# ========================================
# FUNGSI 4: EKSTRAK ENTITAS (KEYWORDS)
# ========================================
# CATATAN: Fungsi ini tidak lagi digunakan untuk pemetaan utama,
# karena model semantik akan menganalisis SELURUH teks CV.
# Namun, fungsi ini masih bisa berguna untuk menampilkan 'skills' 
# yang terdeteksi di UI.
def extract_profile_entities(raw_cv: str) -> str:
    """
    Mengekstrak keterampilan teknis, tools, dan teknologi dari teks CV
    menggunakan pattern matching dan kamus keyword yang telah ditentukan.
    """
    # Dictionary keyword teknologi (dapat diperluas)
    tech_keywords = {
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 
        'go', 'golang', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab',
        
        # Web Technologies
        'html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express', 'django',
        'flask', 'fastapi', 'spring', 'laravel', 'asp.net', 'nextjs', 'nuxt',
        
        # Databases
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite',
        'cassandra', 'elasticsearch', 'dynamodb', 'mariadb',
        
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab',
        'terraform', 'ansible', 'ci/cd', 'devops', 'git', 'github', 'bitbucket',
        
        # Data Science & AI
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
        'scikit-learn', 'pandas', 'numpy', 'data science', 'analytics', 'bi',
        'power bi', 'tableau', 'hadoop', 'spark', 'kafka',
        
        # Mobile Development
        'android', 'ios', 'flutter', 'react native', 'xamarin', 'ionic',
        
        # Testing & QA
        'selenium', 'junit', 'pytest', 'jest', 'cypress', 'qa', 'testing',
        
        # Project Management & Methodologies
        'agile', 'scrum', 'kanban', 'jira', 'trello', 'confluence',
        
        # Security
        'security', 'cybersecurity', 'penetration testing', 'ethical hacking',
        'firewall', 'encryption', 'ssl', 'oauth',
        
        # Networking
        'networking', 'cisco', 'tcp/ip', 'vpn', 'dns', 'load balancing',
        
        # Other Tools
        'api', 'rest', 'graphql', 'microservices', 'soap', 'json', 'xml',
        'linux', 'unix', 'windows server', 'bash', 'powershell'
    }
    
    text_lower = raw_cv.lower()
    found_keywords = set()
    
    for keyword in tech_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text_lower):
            found_keywords.add(keyword)
    
    acronyms = re.findall(r'\b[A-Z]{2,5}\b', raw_cv)
    common_tech_acronyms = {
        'API', 'SDK', 'IDE', 'SQL', 'NoSQL', 'AWS', 'GCP', 'CI', 'CD', 
        'ML', 'AI', 'BI', 'ETL', 'UI', 'UX', 'QA', 'REST', 'SOAP', 'JSON',
        'XML', 'HTML', 'CSS', 'JS', 'TS', 'SPA', 'SSR', 'SEO', 'CMS',
        'ERP', 'CRM', 'IoT', 'AR', 'VR', 'NLP', 'CNN', 'RNN', 'GAN'
    }
    relevant_acronyms = [a for a in acronyms if a in common_tech_acronyms]
    found_keywords.update([a.lower() for a in relevant_acronyms])
    
    version_patterns = re.findall(
        r'\b(python|java|node\.?js|php|ruby|go)\s*\d+(?:\.\d+)*\b',
        text_lower
    )
    found_keywords.update(version_patterns)
    
    framework_context = re.findall(
        r'(?:framework|library|tool|platform)[:\s]+([a-z0_9\-\.]+)',
        text_lower
    )
    found_keywords.update(framework_context)
    
    entities = ' '.join(sorted(found_keywords))
    
    return entities if entities else raw_cv


# ========================================
# FUNGSI 5: INISIALISASI SEMANTIC SEARCH
# ========================================

@st.cache_resource
def initialize_semantic_search(excel_path, sheet_name):
    """
    Membuat atau memuat AI Embedding Model dan FAISS Vector Index.
    Proses ini (membuat index) HANYA berjalan sekali dan hasilnya disimpan
    ke file 'pon_index.faiss' dan 'pon_data.pkl' untuk loading cepat.
    """
    st.info("Inisialisasi AI Semantic Search Engine...")
    
    INDEX_FILE = "pon_index.faiss"
    DATA_FILE = "pon_data.pkl"
    
    # Nama model semantic. 
    # 'paraphrase-multilingual-MiniLM-L12-v2' bagus untuk multi-bahasa (Indonesia + Inggris/Teknis)
    MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
    
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        st.error(f"Gagal mengunduh/memuat model embedding '{MODEL_NAME}'. Cek koneksi internet.")
        st.error(e)
        return None, None, None

    # Cek apakah index sudah ada
    if os.path.exists(INDEX_FILE) and os.path.exists(DATA_FILE):
        try:
            st.write("Memuat index FAISS yang sudah ada...")
            index = faiss.read_index(INDEX_FILE)
            
            with open(DATA_FILE, 'rb') as f:
                df_pon = pickle.load(f)
            
            st.success(f"Semantic Engine siap (dimuat dari file).")
            return model, index, df_pon
        
        except Exception as e:
            st.warning(f"Gagal memuat index dari file: {e}. Membuat ulang index...")

    # --- Jika index tidak ada, buat baru ---
    st.warning("Membuat index semantik baru. Ini mungkin perlu beberapa menit...")
    
    df_pon = load_excel_sheet(excel_path, sheet_name)
    
    if df_pon is None or df_pon.empty:
        st.error("Data PON TIK tidak bisa dimuat atau kosong. Pemetaan tidak dapat dilanjutkan.")
        return None, None, None

    # Validasi kolom
    required_cols = ['Okupasi', 'Unit_Kompetensi', 'Kuk_Keywords']
    if not all(c in df_pon.columns for c in required_cols):
        st.error(f"Kolom {required_cols} tidak ditemukan di sheet {sheet_name}.")
        return None, None, None

    # Gabungkan teks dari kolom relevan menjadi satu "dokumen" per okupasi
    # Ini adalah teks yang akan dipahami oleh AI
    pon_corpus = (
        "Okupasi: " + df_pon['Okupasi'].astype(str) + ". " + 
        "Unit Kompetensi: " + df_pon['Unit_Kompetensi'].astype(str) + ". " + 
        "Keterampilan Kunci: " + df_pon['Kuk_Keywords'].astype(str)
    )
    
    st.write(f"Membuat embedding untuk {len(pon_corpus)} deskripsi okupasi...")
    
    # Ini adalah langkah AI: mengubah semua teks PON TIK menjadi vektor
    pon_vectors = model.encode(pon_corpus.tolist(), show_progress_bar=True)
    
    # Tentukan dimensi vektor (dari model)
    d = pon_vectors.shape[1]
    
    # Buat index FAISS
    # 'IndexFlatL2' adalah index standar untuk pencarian L2 (jarak)
    # 'IndexFlatIP' juga bisa digunakan (Inner Product)
    index = faiss.IndexFlatIP(d) 
    
    # Normalisasi vektor (wajib untuk IndexFlatIP agar setara cosine similarity)
    faiss.normalize_L2(pon_vectors)
    
    # Tambahkan vektor ke index
    index.add(pon_vectors)
    
    # Simpan index dan data frame ke disk
    faiss.write_index(index, INDEX_FILE)
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(df_pon, f)
        
    st.success(f"Semantic Engine baru berhasil dibuat dan disimpan.")
    return model, index, df_pon


# ========================================
# FUNGSI 6: PEMETAAN PROFIL
# ========================================

def map_profile_semantically(profile_text: str):
    """
    Mencari okupasi PON TIK yang paling relevan secara SEMANTIK
    menggunakan AI embedding dan FAISS.
    """
    # Panggil semantic engine yang sudah di-cache
    model, index, df_pon = initialize_semantic_search(EXCEL_PATH, SHEET_PON)
    
    if model is None or index is None:
        return None, None, 0, ""
    
    try:
        # 1. Ubah teks profil (query) menjadi vector
        # Kita embed seluruh teks CV, bukan cuma keywords
        query_vector = model.encode([profile_text])
        
        # Normalisasi query vector
        faiss.normalize_L2(query_vector)
        
        # 2. Cari 1 tetangga terdekat (k=1)
        # index.search akan mengembalikan (Scores, Indices)
        scores, indices = index.search(query_vector, k=1)
        
        idx = indices[0][0]       # Index dari baris yang paling cocok
        best_score = scores[0][0] # Skor kecocokan (Inner Product)

        # 3. Ambil data lengkap dari okupasi terbaik
        data = df_pon.iloc[idx]
        
        # 4. Hitung Skill Gap Dinamis (logika ini tetap sama)
        required_keywords_raw = str(data.get('Kuk_Keywords', '')).lower().split()
        required_keywords = set(k for k in required_keywords_raw if k and len(k) > 2)
        
        # Gunakan teks profil mentah untuk perbandingan skill
        user_keywords = set(profile_text.lower().split()) 
        
        missing_skills = [
            s.title() for s in required_keywords 
            if s not in user_keywords
        ]
        
        if missing_skills:
            skill_gap_text = ", ".join(sorted(missing_skills)[:5])
        else:
            skill_gap_text = "Tidak ada gap signifikan yang terdeteksi."

        return (
            data.get('OkupasiID', 'N/A'),
            data.get('Okupasi', 'N/A'),
            best_score, # Skor ini sekarang adalah Inner Product, bukan Cosine Sim TF-IDF
            skill_gap_text
        )
        
    except Exception as e:
        st.error(f"Error saat melakukan pemetaan semantik: {e}")
        st.code(traceback.format_exc())
        return None, None, 0, ""


# ========================================
# INISIALISASI SESSION STATE
# ========================================
default_keys = ['form_email', 'form_nama', 'form_lokasi', 'form_linkedin', 'form_cv_text']
for key in default_keys:
    st.session_state.setdefault(key, "")

# ========================================
# UI: JUDUL DAN UPLOADER
# ========================================
st.title("Profil Talenta")
st.markdown("Unggah CV Anda untuk dianalisis dan dipetakan ke standar **PON TIK**.")

uploaded_file = st.file_uploader(
    "Unggah CV (PDF, DOCX, atau TXT)", 
    type=["pdf", "docx", "txt"]
)

if uploaded_file:
    with st.spinner("Membaca dan memproses CV..."):
        try:
            if uploaded_file.type == "application/pdf":
                raw_text = extract_text_from_pdf(io.BytesIO(uploaded_file.getvalue()))
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                raw_text = extract_text_from_docx(io.BytesIO(uploaded_file.getvalue()))
            else: # TXT
                raw_text = uploaded_file.getvalue().decode("utf-8")

            parsed = parse_cv_data(raw_text)
            for k, v in parsed.items():
                st.session_state[f"form_{k}"] = v
            st.success("CV berhasil diproses! Silakan periksa data di bawah.")
            
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

st.markdown("---")

# ========================================
# UI: FORM INPUT DATA TALENTA
# ========================================
st.markdown("### Lengkapi Profil Anda")
st.caption("Data yang diekstrak dari CV Anda akan muncul di sini. Anda dapat mengeditnya jika perlu.")

with st.form("profil_form"):
    
    email = st.text_input("Email*", st.session_state.form_email)
    nama = st.text_input("Nama Lengkap*", st.session_state.form_nama)
    lokasi = st.text_input("Lokasi", st.session_state.form_lokasi)
    linkedin = st.text_input("URL LinkedIn", st.session_state.form_linkedin)
    raw_cv = st.text_area(
        "CV atau Deskripsi Diri*", 
        st.session_state.form_cv_text, 
        height=250, 
        help="AI akan menganalisis MAKNA dari seluruh teks ini."
    )
    
    submitted = st.form_submit_button("Simpan & Petakan Profil")


# ========================================
# PROSES SUBMIT FORM
# ========================================

if submitted:
    if not email or not nama or not raw_cv:
        st.warning("Mohon isi field yang wajib diisi (Email, Nama, dan CV).")
    else:
        # Update session state
        st.session_state.form_email = email
        st.session_state.form_nama = nama
        st.session_state.form_lokasi = lokasi
        st.session_state.form_linkedin = linkedin
        st.session_state.form_cv_text = raw_cv
        
        with st.spinner("Menganalisis profil dan memetakan ke PON TIK secara semantik..."):
            
            # 1. Tampilkan entitas (keywords) yang terdeteksi untuk info
            #    (Langkah ini opsional, tidak dipakai untuk mapping)
            entities = extract_profile_entities(raw_cv)
            if entities != raw_cv:
                with st.expander("Teknologi & Skill (Keywords) Terdeteksi"):
                    st.write(entities.replace(" ", ", "))
            
            # 2. Lakukan pemetaan semantik (DIUBAH)
            #    Kita memetakan menggunakan SELURUH teks CV (raw_cv),
            #    bukan hanya entitas/keywords. Ini jauh lebih akurat.
            okupasi_id, okupasi_nama, skor, gap = map_profile_semantically(raw_cv)

            if okupasi_id:
                # 3. Simpan hasil penting
                st.session_state.talent_id = email
                st.session_state.mapped_okupasi_id = okupasi_id
                st.session_state.mapped_okupasi_nama = okupasi_nama
                st.session_state.skill_gap = gap
                st.session_state.profile_text = raw_cv # Simpan teks mentah

                # 4. Tampilkan hasil
                st.success("Profil Berhasil Dipetakan (Secara Semantik)!")
                
                col1, col2 = st.columns(2)
                col1.metric("Okupasi Paling Sesuai", okupasi_nama)
                # Skor IP (Inner Product) biasanya antara 0 dan 1 (setelah normalisasi)
                col2.metric("Tingkat Kecocokan", f"{skor*100:.2f}%")
                
                st.warning(f"Contoh Skill Gap Terdeteksi: {gap}")
                st.info("Silakan lanjut ke halaman **Asesmen Kompetensi** untuk validasi kemampuan Anda.")
            else:
                st.error("Gagal memetakan profil. Periksa kembali data atau log error di atas (jika ada).")
