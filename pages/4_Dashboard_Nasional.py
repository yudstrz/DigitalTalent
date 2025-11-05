"""
HALAMAN DASHBOARD - Versi AI (Clean Version)
Dashboard tanpa pesan debug/info yang mengganggu
"""

import streamlit as st
import pandas as pd
import altair as alt
import requests
import json
import traceback

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Dashboard Nasional",
    page_icon="📊",
    layout="wide"
)

# --- Konfigurasi API Gemini ---
try:
    from config import GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL
except ImportError:
    GEMINI_API_KEY = ""
    GEMINI_BASE_URL = ""
    GEMINI_MODEL = ""

# ========================================
# FUNGSI HELPER (LLM)
# ========================================
def call_gemini_api(prompt: str) -> str:
    """Mengirim request ke Gemini API dan mengembalikan respons teks."""
    if not GEMINI_API_KEY or not GEMINI_BASE_URL:
        return "Error: Konfigurasi Gemini API tidak ditemukan."
        
    url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.5}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        result = response.json()
        content = result['candidates'][0]['content']['parts'][0]['text']
        return content
    except Exception as e:
        return f"Error: Gagal menghubungi AI. {e}"

def get_llm_insight(data_okupasi, data_gap, data_lokasi, data_klaster):
    """Merakit prompt dan memanggil AI untuk insight dashboard."""
    okupasi_str = data_okupasi.head(5).to_string() if not data_okupasi.empty else "Tidak ada data"
    gap_str = data_gap.head(5).to_string() if not data_gap.empty else "Tidak ada data"
    lokasi_str = data_lokasi.head(5).to_string() if not data_lokasi.empty else "Tidak ada data"
    klaster_str = data_klaster.to_string() if not data_klaster.empty else "Tidak ada data"

    prompt = f"""
Anda adalah seorang Analis Data SDM TIK senior.
Tugas Anda adalah membuat laporan ringkasan eksekutif (insight)
berdasarkan data dashboard berikut:

1. Data Distribusi Okupasi (Top 5):
{okupasi_str}

2. Data Skill Gap Terbesar (Top 5):
{gap_str}

3. Data Sebaran Lokasi Talenta (Top 5):
{lokasi_str}

4. Data Analisis Klaster Talenta:
{klaster_str}

**Instruksi (WAJIB):**
Berikan 3 insight utama dan 2 rekomendasi kebijakan strategis
berdasarkan HANYA pada data di atas.
Gunakan format poin-poin (bullet points) Markdown.
Gunakan Bahasa Indonesia.
"""
    
    response = call_gemini_api(prompt)
    return response

# ========================================
# FUNGSI PEMBACA EXCEL
# ========================================
def read_excel_smart(file_path, sheet_name):
    """Membaca Excel dengan deteksi header otomatis (SILENT MODE)."""
    try:
        for header_row in [0, 1, 2]:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
                df.columns = df.columns.astype(str).str.strip()
                
                if df.columns.duplicated().any():
                    cols = pd.Series(df.columns)
                    for dup in cols[cols.duplicated()].unique():
                        cols[cols == dup] = [f"{dup}_{i}" if i != 0 else dup 
                                            for i in range(sum(cols == dup))]
                    df.columns = cols
                
                if len(df) > 0 and len(df.columns) > 0:
                    valid_cols = sum([1 for col in df.columns if str(col) not in ['nan', 'NaN', '', 'Unnamed']])
                    if valid_cols >= 3:
                        return df
            except:
                continue
        
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        return df
    except Exception as e:
        return pd.DataFrame()

def get_national_dashboard_data(file_path="data/DTP_Database.xlsx"):
    """Membaca data dashboard dari file Excel (SILENT MODE)."""
    
    distribusi_okupasi = pd.DataFrame(columns=["Okupasi", "Jumlah_Talenta"])
    distribusi_okupasi_indexed = pd.DataFrame(columns=["Jumlah_Talenta"])
    sebaran_lokasi = pd.DataFrame(columns=["Lokasi", "Jumlah"])
    skill_gap_umum = pd.DataFrame(columns=["Keterampilan", "Jumlah_Gap"])
    
    try:
        df_talenta = read_excel_smart(file_path, "Talenta")
        df_hasil = read_excel_smart(file_path, "Hasil_Pemetaan_Asesmen")
        df_pon_tik = read_excel_smart(file_path, "PON_TIK_Master")
        
        if df_hasil.empty or df_talenta.empty or df_pon_tik.empty:
            st.error("❌ Gagal membaca file Excel. Periksa path file dan struktur data.")
            return df_hasil, distribusi_okupasi_indexed, distribusi_okupasi, sebaran_lokasi, skill_gap_umum
    except Exception as e:
        st.error(f"❌ Error membaca Excel: {e}")
        return pd.DataFrame(), distribusi_okupasi_indexed, distribusi_okupasi, sebaran_lokasi, skill_gap_umum

    # ========================================
    # DISTRIBUSI OKUPASI
    # ========================================
    try:
        okupasi_col = None
        for col in df_pon_tik.columns:
            if 'okupasi' in str(col).lower():
                okupasi_col = col
                break
        
        if okupasi_col:
            distribusi_okupasi = df_pon_tik[okupasi_col].value_counts().reset_index()
            distribusi_okupasi.columns = ["Okupasi", "Jumlah_Talenta"]
            distribusi_okupasi_indexed = distribusi_okupasi.set_index("Okupasi")
    except:
        pass

    # ========================================
    # SEBARAN LOKASI
    # ========================================
    try:
        lokasi_col = None
        for col in df_talenta.columns:
            if 'lokasi' in str(col).lower():
                lokasi_col = col
                break
        
        if lokasi_col:
            sebaran_lokasi = df_talenta[lokasi_col].value_counts().reset_index()
            sebaran_lokasi.columns = ["Lokasi", "Jumlah"]
            sebaran_lokasi['Lokasi'] = sebaran_lokasi['Lokasi'].fillna('Tidak Diketahui')
    except:
        pass

    # ========================================
    # SKILL GAP
    # ========================================
    try:
        gap_col = None
        for col in df_hasil.columns:
            col_lower = str(col).lower()
            if 'gap' in col_lower and 'keterampilan' in col_lower:
                gap_col = col
                break
        
        if gap_col:
            gaps = df_hasil[gap_col].dropna().astype(str)
            gaps = gaps[gaps != '']
            gaps = gaps[gaps != 'nan']
            gaps = gaps[~gaps.str.lower().str.contains('null', na=False)]
            
            if len(gaps) > 0:
                all_gaps = []
                for gap_str in gaps:
                    if ',' in gap_str:
                        all_gaps.extend([g.strip() for g in gap_str.split(',')])
                    else:
                        all_gaps.append(gap_str.strip())
                
                all_gaps = [g for g in all_gaps if g and g.lower() != 'null']
                
                if len(all_gaps) > 0:
                    skill_gap_counts = pd.Series(all_gaps).value_counts().reset_index()
                    skill_gap_counts.columns = ["Keterampilan", "Jumlah_Gap"]
                    skill_gap_umum = skill_gap_counts.set_index("Keterampilan")
    except:
        pass

    return (
        df_hasil,
        distribusi_okupasi_indexed,
        distribusi_okupasi,
        sebaran_lokasi,
        skill_gap_umum
    )

# ========================================
# --- UI DASHBOARD ---
# ========================================
st.title("📊 Dashboard Talenta Digital Nasional")

if st.button("🔄 Refresh Data"):
    st.rerun()

with st.spinner("⏳ Memuat data..."):
    (
        df_hasil,
        dist_okupasi_indexed,
        dist_okupasi_chart_data,
        sebaran_lokasi,
        skill_gap
    ) = get_national_dashboard_data()

st.markdown("---")

# ========================================
# K-MEANS CLUSTERING
# ========================================
st.header("🎯 Analisis Klaster Talenta (K-Means)")

data_for_clustering = pd.DataFrame()
cluster_analysis = pd.DataFrame()

try:
    col_kecocokan = None
    col_asesmen = None
    
    for col in df_hasil.columns:
        col_lower = str(col).lower()
        if 'skor' in col_lower and 'kecocokan' in col_lower:
            col_kecocokan = col
        if 'skor' in col_lower and 'asesmen' in col_lower:
            col_asesmen = col
    
    if col_kecocokan and col_asesmen:
        data_for_clustering = df_hasil[[col_kecocokan, col_asesmen]].copy()
        data_for_clustering.columns = ["Skor_Kecocokan", "Skor_Asesmen"]
        data_for_clustering = data_for_clustering.dropna()
        data_for_clustering = data_for_clustering[
            (data_for_clustering['Skor_Kecocokan'] > 0) & 
            (data_for_clustering['Skor_Asesmen'] > 0)
        ]
        
        if len(data_for_clustering) >= 4:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_for_clustering)
            
            n_clusters = min(4, len(data_for_clustering))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(data_scaled)
            
            data_for_clustering["Klaster"] = [f"Klaster {c+1}" for c in clusters]
            
            chart_cluster = alt.Chart(data_for_clustering).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X('Skor_Kecocokan:Q', scale=alt.Scale(zero=False), title="Skor Kecocokan Awal"),
                y=alt.Y('Skor_Asesmen:Q', title="Skor Asesmen"),
                color=alt.Color('Klaster:N', title="Klaster"),
                tooltip=['Skor_Kecocokan', 'Skor_Asesmen', 'Klaster']
            ).properties(height=400).interactive()
            
            st.altair_chart(chart_cluster, use_container_width=True)
            
            cluster_analysis = data_for_clustering.groupby('Klaster').agg(
                Jumlah=('Klaster', 'count'),
                Rata_Kecocokan=('Skor_Kecocokan', 'mean'),
                Rata_Asesmen=('Skor_Asesmen', 'mean')
            ).reset_index()
            
            with st.expander("📊 Detail Klaster"):
                st.dataframe(cluster_analysis, use_container_width=True)
                st.info("""
                **Interpretasi:**
                - **Kecocokan Tinggi + Asesmen Tinggi:** Talenta 'Validated' 
                - **Kecocokan Tinggi + Asesmen Rendah:** Talenta 'Over-promise'
                - **Kecocokan Rendah + Asesmen Tinggi:** Talenta 'Hidden Gem'
                - **Kecocokan Rendah + Asesmen Rendah:** Talenta 'Emerging'
                """)
        else:
            st.warning(f"⚠️ Data tidak cukup untuk clustering (tersedia: {len(data_for_clustering)} talenta, minimal: 4)")
    else:
        st.warning("⚠️ Kolom untuk clustering tidak ditemukan. Pastikan kolom 'Skor_Kecocokan_Awal' dan 'Skor_Asesmen' ada di sheet 'Hasil_Pemetaan_Asesmen'.")
        
except Exception as e:
    st.error(f"❌ Error clustering: {e}")

st.markdown("---")

# ========================================
# VISUALISASI
# ========================================
st.header("📈 Visualisasi Data Agregat")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribusi Okupasi")
    if not dist_okupasi_chart_data.empty:
        top_data = dist_okupasi_chart_data.nlargest(10, 'Jumlah_Talenta')
        chart = alt.Chart(top_data).mark_bar().encode(
            x='Jumlah_Talenta:Q',
            y=alt.Y('Okupasi:N', sort='-x'),
            tooltip=['Okupasi', 'Jumlah_Talenta']
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Tidak ada data okupasi")

with col2:
    st.subheader("Sebaran Lokasi")
    if not sebaran_lokasi.empty:
        top_data = sebaran_lokasi.nlargest(10, 'Jumlah')
        chart = alt.Chart(top_data).mark_arc().encode(
            theta='Jumlah:Q',
            color='Lokasi:N',
            tooltip=['Lokasi', 'Jumlah']
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Tidak ada data lokasi")

st.subheader("🎯 Skill Gap Nasional")
if not skill_gap.empty:
    st.bar_chart(skill_gap.head(15)['Jumlah_Gap'])
else:
    st.info("Tidak ada data skill gap")

st.markdown("---")

# ========================================
# AI INSIGHT
# ========================================
st.header("🤖 Analisis AI & Rekomendasi")

if st.button("🚀 Generate Insight AI", type="primary"):
    if dist_okupasi_chart_data.empty and sebaran_lokasi.empty:
        st.error("Data tidak cukup untuk analisis AI")
    else:
        with st.spinner("🧠 AI sedang menganalisis..."):
            insight = get_llm_insight(
                dist_okupasi_chart_data,
                skill_gap.reset_index() if not skill_gap.empty else pd.DataFrame(),
                sebaran_lokasi,
                cluster_analysis
            )
            st.markdown(insight)

st.markdown("---")

# ========================================
# METRICS
# ========================================
st.subheader("📊 Ringkasan Metrik")
col1, col2, col3 = st.columns(3)

total = sebaran_lokasi[sebaran_lokasi['Lokasi'] != 'Tidak Diketahui']['Jumlah'].sum() if not sebaran_lokasi.empty else 0
col1.metric("Total Talenta", f"{int(total):,}")
col2.metric("Jenis Okupasi", len(dist_okupasi_indexed))
col3.metric("Skill Gap", len(skill_gap))
