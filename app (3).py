"""
Analisis Pola Penyaluran Dana Zakat dan Infak
K-Means Clustering — NU CARE Kabupaten Wonosobo

Versi ini disesuaikan dengan metodologi BAB IV skripsi:
- Fitur clustering: Fakir, Miskin, Amil, Sabilillah, Ibnu Sabil (5 fitur asnaf)
- Unit analisis: tingkat Dukuh (bukan Kecamatan/Desa agregat)
- Standardisasi: StandardScaler (Z-score)
- Algoritma: KMeans (scikit-learn), random_state=42, n_init=20
- K optimal (default): K=3 -> label Prioritas Rendah / Sedang / Tinggi
- Evaluasi: Silhouette Score, opsional analisis Elbow Method (K=2 s.d. 8)
- Visualisasi sebaran klaster: proyeksi PCA 2 dimensi
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import io
import re

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Analisis Penyaluran Zakat & Infak — NU CARE Wonosobo",
    page_icon="📊",
    layout="wide"
)

# ======================================================
# KONSTANTA METODOLOGI (mengikuti BAB IV)
# ======================================================
FEATURES = ["Fakir", "Miskin", "Amil", "Sabilillah", "Ibnu Sabil"]

CLUSTER_COLORS = ["#3E5A78", "#B9872C", "#8A4B3B", "#2F6B4F", "#7A5B8C", "#5A3E78", "#78573E", "#3E7867"]

LABELS_BY_K = {
    2: ["Prioritas Rendah", "Prioritas Tinggi"],
    3: ["Prioritas Rendah", "Prioritas Sedang", "Prioritas Tinggi"],
    4: ["Prioritas Sangat Rendah", "Prioritas Rendah", "Prioritas Tinggi", "Prioritas Sangat Tinggi"],
    5: ["Prioritas Sangat Rendah", "Prioritas Rendah", "Prioritas Sedang", "Prioritas Tinggi", "Prioritas Sangat Tinggi"],
    6: [f"Klaster {i+1}" for i in range(6)],
    7: [f"Klaster {i+1}" for i in range(7)],
    8: [f"Klaster {i+1}" for i in range(8)],
}

KECAMATAN_LIST = [
    "Wonosobo", "Kertek", "Selomerto", "Leksono", "Sukoharjo", "Watumalang",
    "Mojotengah", "Garung", "Kejajar", "Kalikajar", "Kepil", "Sapuran",
    "Kaliwiro", "Wadaslintang", "Kalibawang"
]

DATA_COLUMNS = [
    "Kecamatan", "Desa", "Dukuh", "Periode",
    "Fakir", "Miskin", "Amil", "Sabilillah", "Ibnu Sabil", "Total Mustahik"
]

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
    --paper: #F1F0E6;
    --paper-2: #E9E7D8;
    --ink: #1C2A21;
    --ink-soft: #4A574C;
    --deep: #163829;
    --deep-2: #0E271B;
    --gold: #B9872C;
    --gold-light: #E4C77E;
    --line: #D7D2BE;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: var(--paper); }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; color: var(--deep); }
.stButton>button {
    background-color: var(--deep); color: white; border-radius: 8px; border: none;
    font-weight: 600;
}
.stButton>button:hover { background-color: var(--deep-2); color: white; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--paper-2); padding: 4px; border-radius: 12px; }
.stTabs [data-baseweb="tab"] { border-radius: 9px; font-weight: 600; color: var(--ink-soft); }
.stTabs [aria-selected="true"] { background-color: var(--deep) !important; color: white !important; }
.hero {
    background: linear-gradient(155deg, var(--deep), var(--deep-2));
    color: #FBFAF4; padding: 24px 28px; border-radius: 14px; margin-bottom: 20px;
}
.hero .eyebrow { font-family:'IBM Plex Mono', monospace; font-size:11.5px; letter-spacing:.14em;
    text-transform:uppercase; color: var(--gold-light); }
.hero h1 { color: white !important; font-size: 26px; margin: 6px 0; }
.hero .sub { font-size: 13px; color: rgba(251,250,244,0.75); }
.metric-card {
    background: white; border: 1px solid var(--line); border-radius: 10px; padding: 16px;
    border-top: 4px solid var(--gold);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <span class="eyebrow">Sistem Pendukung Analisis · Tugas Akhir</span>
    <h1>Analisis Pola Penyaluran Dana Zakat &amp; Infak dengan K-Means Clustering</h1>
    <div class="sub">NU CARE – LAZISNU Kabupaten Wonosobo · Fitur clustering: Fakir, Miskin, Amil, Sabilillah, dan Ibnu Sabil pada tingkat Dukuh.</div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# DATA CONTOH (mengacu pada nilai riil yang telah diuraikan pada BAB IV,
# termasuk dua dukuh kategori Prioritas Tinggi pada Tabel 4.5)
# ======================================================
def seed_data():
    rows = [
        # Kecamatan, Desa, Dukuh, Periode, Fakir, Miskin, Amil, Sabilillah, Ibnu Sabil
        ("Sukoharjo", "Suroyudan", "Penulih", "2026/1447 H", 18, 32, 13, 24, 20),
        ("Sukoharjo", "Suroyudan", "Puntuk Gandu", "2026/1447 H", 0, 15, 7, 11, 0),
        ("Sukoharjo", "Suroyudan", "Srisip", "2026/1447 H", 30, 24, 12, 8, 0),
        ("Sukoharjo", "Suroyudan", "Pegandulan", "2026/1447 H", 32, 60, 30, 17, 15),
        ("Sukoharjo", "Jebengplampitan", "Kutawuluh", "2026/1447 H", 77, 153, 20, 20, 0),
        ("Sukoharjo", "Jebengplampitan", "Pucung", "2026/1447 H", 28, 20, 8, 8, 0),
        ("Sukoharjo", "Jebengplampitan", "Gondangsari", "2026/1447 H", 0, 0, 0, 0, 0),
        ("Sukoharjo", "Jebengplampitan", "Kenanga", "2026/1447 H", 13, 14, 4, 6, 0),
        ("Sukoharjo", "Jebengplampitan", "Kuwarasan", "2026/1447 H", 0, 29, 0, 8, 4),
        ("Sukoharjo", "Jebengplampitan", "Pagemrosan", "2026/1447 H", 7, 45, 9, 4, 0),
        ("Sukoharjo", "Sukoharjo", "Sampih", "2026/1447 H", 100, 301, 15, 48, 0),
        ("Sukoharjo", "Garunglor", "Karangtengah", "2026/1447 H", 86, 192, 33, 76, 0),
        ("Sukoharjo", "Mergosari", "Karangsari", "2026/1447 H", 20, 60, 15, 12, 5),
        ("Sukoharjo", "Kalibening", "Lamuk", "2026/1447 H", 15, 55, 12, 10, 8),
    ]
    df = pd.DataFrame(rows, columns=["Kecamatan", "Desa", "Dukuh", "Periode",
                                      "Fakir", "Miskin", "Amil", "Sabilillah", "Ibnu Sabil"])
    df["Total Mustahik"] = df[FEATURES].sum(axis=1)
    return df[DATA_COLUMNS]


def parse_rekap_zakat(file):
    """
    Parser untuk file rekap zakat dengan format:
    - Baris judul berisi 'REKAPITULASI ... KEC.<NAMA> TAHUN <PERIODE>'
    - Header bertingkat (baris kolom utama + baris satuan)
    - Data per Desa -> Dukuh, dengan kolom:
      NO, DESA, DUKUH, ZAKAT FITRAH UANG, ZAKAT FITRAH BERAS(KG), ZAKAT MAL UANG,
      (kosong), FAKIR, MISKIN, AMIL, MUALAF, RIQOB, GHORIM, SABILILAH, IBNUSABIL,
      MUSTAHIK ORANG, MUZAKI, MAL ORANG
    - Baris 'JUMLAH' merupakan baris rekapitulasi per desa dan dikeluarkan dari hasil,
      karena unit analisis pada penelitian ini adalah tingkat Dukuh.

    Mengikuti seleksi atribut pada BAB IV (sub-bab 4.2), atribut Muallaf, Riqab, dan
    Ghorim TIDAK diikutsertakan sebagai fitur clustering karena tingkat kekosongan
    data yang sangat tinggi (Riqab bahkan bernilai nol pada seluruh data).

    Menghasilkan data per Dukuh: Kecamatan, Desa, Dukuh, Periode,
    Fakir, Miskin, Amil, Sabilillah, Ibnu Sabil, Total Mustahik.
    """
    raw = pd.read_excel(file, header=None)

    # cari baris judul
    title_text = ""
    for i in range(min(5, len(raw))):
        joined = " ".join(str(v) for v in raw.iloc[i].tolist() if str(v) != "nan")
        if "REKAPITULASI" in joined.upper():
            title_text = joined
            break

    kec_match = re.search(r"KEC\.?\s*([A-Z\s]+?)(?:\s+TAHUN|$)", title_text.upper())
    kecamatan = kec_match.group(1).strip().title() if kec_match else "Tidak Diketahui"

    tahun_match = re.search(r"TAHUN\s+([\d/]+\s*H?)", title_text.upper())
    periode = tahun_match.group(1).strip() if tahun_match else "Tidak diketahui"

    # cari baris header (sel yang isinya persis 'DESA')
    header_row_idx = None
    for i in range(len(raw)):
        vals = [str(v).strip().upper() for v in raw.iloc[i].tolist()]
        if "DESA" in vals:
            header_row_idx = i
            break
    if header_row_idx is None:
        raise ValueError("Format file tidak dikenali: tidak ditemukan baris header 'DESA'.")

    data_start = header_row_idx + 2  # lewati baris satuan
    data = raw.iloc[data_start:].reset_index(drop=True)
    data.columns = range(data.shape[1])
    data[1] = data[1].ffill()  # isi nama desa yang menyatu (merged cell)
    data = data.dropna(how="all")

    is_jumlah = data[2].astype(str).str.strip().str.upper() == "JUMLAH"
    detail = data[~is_jumlah].copy()
    detail = detail.dropna(subset=[2])  # baris harus punya nama dukuh
    detail = detail[detail[2].astype(str).str.strip().str.lower() != "nan"]

    # kolom sesuai struktur berkas sumber (indeks 0-based)
    COL_FAKIR, COL_MISKIN, COL_AMIL = 7, 8, 9
    COL_SABILILAH, COL_IBNUSABIL, COL_MUSTAHIK = 13, 14, 15

    for c in [COL_FAKIR, COL_MISKIN, COL_AMIL, COL_SABILILAH, COL_IBNUSABIL, COL_MUSTAHIK]:
        detail[c] = pd.to_numeric(detail[c], errors="coerce").fillna(0)

    result = pd.DataFrame({
        "Kecamatan": kecamatan,
        "Desa": detail[1].astype(str).str.strip(),
        "Dukuh": detail[2].astype(str).str.strip(),
        "Periode": periode,
        "Fakir": detail[COL_FAKIR],
        "Miskin": detail[COL_MISKIN],
        "Amil": detail[COL_AMIL],
        "Sabilillah": detail[COL_SABILILAH],
        "Ibnu Sabil": detail[COL_IBNUSABIL],
        "Total Mustahik": detail[COL_MUSTAHIK],
    })

    return result.reset_index(drop=True)


if "data" not in st.session_state:
    st.session_state.data = seed_data()

# ======================================================
# TAB NAVIGASI
# ======================================================
tab1, tab2 = st.tabs(["1 · Input Data", "2 · Hasil Clustering"])

# ======================================================
# TAB 1 — INPUT DATA
# ======================================================
with tab1:
    col_form, col_table = st.columns([1, 2], gap="large")

    with col_form:
        st.subheader("Tambah Data Penyaluran")
        st.caption("Satu baris data mewakili penyaluran pada satu Dukuh dalam suatu periode. "
                    "Fitur Fakir, Miskin, Amil, Sabilillah, dan Ibnu Sabil digunakan sebagai fitur clustering.")

        with st.form("form_tambah", clear_on_submit=True):
            kecamatan = st.selectbox("Kecamatan", KECAMATAN_LIST)
            desa = st.text_input("Desa")
            dukuh = st.text_input("Dukuh")
            periode = st.text_input("Periode", placeholder="2026/1447 H")

            c1, c2 = st.columns(2)
            with c1:
                fakir = st.number_input("Fakir", min_value=0, step=1, value=0)
                amil = st.number_input("Amil", min_value=0, step=1, value=0)
                ibnu_sabil = st.number_input("Ibnu Sabil", min_value=0, step=1, value=0)
            with c2:
                miskin = st.number_input("Miskin", min_value=0, step=1, value=0)
                sabilillah = st.number_input("Sabilillah", min_value=0, step=1, value=0)

            submitted = st.form_submit_button("+ Tambah Data", use_container_width=True)

            if submitted:
                if not desa.strip() or not dukuh.strip() or not periode.strip():
                    st.error("Mohon lengkapi Desa, Dukuh, dan Periode.")
                else:
                    total = fakir + miskin + amil + sabilillah + ibnu_sabil
                    new_row = pd.DataFrame([{
                        "Kecamatan": kecamatan, "Desa": desa.strip(), "Dukuh": dukuh.strip(),
                        "Periode": periode.strip(),
                        "Fakir": fakir, "Miskin": miskin, "Amil": amil,
                        "Sabilillah": sabilillah, "Ibnu Sabil": ibnu_sabil,
                        "Total Mustahik": total
                    }])
                    st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                    st.success("Data berhasil ditambahkan.")
                    st.rerun()

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("↺ Muat Data Contoh", use_container_width=True):
                st.session_state.data = pd.concat([st.session_state.data, seed_data()], ignore_index=True)
                st.rerun()
        with c2:
            if st.button("🗑 Hapus Semua", use_container_width=True):
                st.session_state.data = st.session_state.data.iloc[0:0]
                st.rerun()

        import_mode = st.radio(
            "Mode impor",
            ["Template Sederhana (Kecamatan, Desa, Dukuh, Periode, Fakir, Miskin, Amil, Sabilillah, Ibnu Sabil)",
             "Rekap Zakat per Kecamatan (format Desa/Dukuh dari berkas sumber)"],
            help="Pilih 'Rekap Zakat per Kecamatan' untuk berkas seperti REKAPITULASI PEROLEHAN ZAKAT "
                 "dengan rincian per Desa/Dukuh. Kolom asnaf akan diekstrak otomatis sesuai struktur berkas."
        )

        uploaded = st.file_uploader(
            "⬆ Impor Data (CSV atau Excel)",
            type=["csv", "xlsx", "xls"]
        )

        if uploaded is not None:
            try:
                if import_mode.startswith("Rekap Zakat"):
                    if not uploaded.name.endswith((".xlsx", ".xls")):
                        st.error("Mode 'Rekap Zakat per Kecamatan' hanya mendukung file Excel (.xlsx/.xls).")
                    else:
                        parsed = parse_rekap_zakat(uploaded)
                        st.session_state.data = pd.concat([st.session_state.data, parsed], ignore_index=True)
                        st.success(
                            f"{len(parsed)} dukuh berhasil diimpor dari {uploaded.name} "
                            f"(Kecamatan {parsed['Kecamatan'].iloc[0]}, Periode {parsed['Periode'].iloc[0]})."
                        )
                        with st.expander("Lihat detail hasil parsing"):
                            st.dataframe(parsed, use_container_width=True, hide_index=True)
                else:
                    if uploaded.name.endswith(".csv"):
                        imported = pd.read_csv(uploaded)
                    else:
                        imported = pd.read_excel(uploaded)

                    required_cols = set(DATA_COLUMNS) - {"Total Mustahik"}
                    if not required_cols.issubset(set(imported.columns)):
                        st.error(
                            "Kolom pada file tidak sesuai. Pastikan file memiliki kolom: "
                            f"{', '.join(sorted(required_cols))}. "
                            "Kalau file kamu format rekap per Desa/Dukuh dari berkas sumber, gunakan mode "
                            "'Rekap Zakat per Kecamatan' di atas."
                        )
                    else:
                        imported = imported[list(required_cols)].copy()
                        for f in FEATURES:
                            imported[f] = pd.to_numeric(imported[f], errors="coerce").fillna(0)
                        imported["Total Mustahik"] = imported[FEATURES].sum(axis=1)
                        imported = imported[DATA_COLUMNS]
                        st.session_state.data = pd.concat([st.session_state.data, imported], ignore_index=True)
                        st.success(f"{len(imported)} baris berhasil diimpor dari {uploaded.name}.")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    with col_table:
        st.subheader(f"Data Penyaluran ({len(st.session_state.data)} baris)")
        st.caption("Kamu bisa edit langsung di tabel (klik sel), atau hapus baris lewat ikon 🗑 di ujung kiri baris. "
                    "Kolom Total Mustahik dihitung otomatis dari penjumlahan kelima fitur asnaf.")

        edited = st.data_editor(
            st.session_state.data,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_data",
            column_config={
                "Fakir": st.column_config.NumberColumn(format="%d"),
                "Miskin": st.column_config.NumberColumn(format="%d"),
                "Amil": st.column_config.NumberColumn(format="%d"),
                "Sabilillah": st.column_config.NumberColumn(format="%d"),
                "Ibnu Sabil": st.column_config.NumberColumn(format="%d"),
                "Total Mustahik": st.column_config.NumberColumn(format="%d orang"),
            }
        )
        # perbarui Total Mustahik apabila ada perubahan manual pada fitur asnaf
        edited["Total Mustahik"] = edited[FEATURES].sum(axis=1)
        st.session_state.data = edited

        csv_buf = io.StringIO()
        st.session_state.data.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇ Ekspor CSV", data=csv_buf.getvalue(),
            file_name="data_penyaluran_nucare_wonosobo.csv", mime="text/csv"
        )
        st.info("⚠️ Data tersimpan sementara selama sesi ini berjalan. Ekspor CSV secara berkala agar tidak hilang saat halaman ditutup/refresh.")

# ======================================================
# TAB 2 — HASIL CLUSTERING
# ======================================================
with tab2:
    df_all = st.session_state.data.dropna(subset=FEATURES)
    for f in FEATURES:
        df_all[f] = pd.to_numeric(df_all[f], errors="coerce")
    df_all = df_all.dropna(subset=FEATURES)

    st.subheader("Pengaturan K-Means")
    st.caption(
        "Fitur yang distandardisasi (StandardScaler): **Fakir, Miskin, Amil, Sabilillah, Ibnu Sabil**. "
        "Standardisasi dilakukan agar setiap fitur memiliki skala yang setara sebelum diproses "
        "menggunakan algoritma K-Means, sebagaimana dijelaskan pada BAB IV."
    )

    with st.expander("🔎 Analisis Jumlah Klaster Optimal (Elbow Method & Silhouette Score)"):
        st.caption("Menghitung SSE (Inertia) dan Silhouette Score untuk K=2 s.d. K=8, mengikuti tahapan pada sub-bab 4.3.")
        if st.button("Jalankan Analisis Elbow & Silhouette"):
            if len(df_all) < 8:
                st.warning(f"Data valid hanya {len(df_all)} baris, minimal 8 baris diperlukan untuk menguji hingga K=8.")
            else:
                X_elbow = StandardScaler().fit_transform(df_all[FEATURES].values)
                k_range = range(2, 9)
                sse_list, sil_list = [], []
                for kk in k_range:
                    m = KMeans(n_clusters=kk, random_state=42, n_init=20)
                    a = m.fit_predict(X_elbow)
                    sse_list.append(m.inertia_)
                    sil_list.append(silhouette_score(X_elbow, a))

                fig_e, ax1 = plt.subplots(figsize=(7, 4))
                ax1.plot(list(k_range), sse_list, marker="o", color="#163829", label="SSE / Inertia")
                ax1.set_xlabel("Jumlah Cluster (K)")
                ax1.set_ylabel("SSE / Inertia", color="#163829")
                ax2 = ax1.twinx()
                ax2.plot(list(k_range), sil_list, marker="s", linestyle="--", color="#B9872C", label="Silhouette Score")
                ax2.set_ylabel("Silhouette Score", color="#B9872C")
                ax1.set_title("Elbow Method & Silhouette Score untuk Pemilihan K")
                fig_e.tight_layout()
                st.pyplot(fig_e)

                st.dataframe(
                    pd.DataFrame({"K": list(k_range), "SSE": sse_list, "Silhouette Score": sil_list}),
                    use_container_width=True, hide_index=True
                )

    colk, colbtn = st.columns([3, 1])
    with colk:
        k = st.select_slider("Jumlah Klaster (K)", options=[2, 3, 4, 5, 6, 7, 8], value=3,
                              help="Sesuai hasil BAB IV, K=3 ditetapkan sebagai jumlah cluster optimal.")
    with colbtn:
        run = st.button("▶ Jalankan Clustering", use_container_width=True)

    if len(df_all) < k:
        st.warning(f"Data terlalu sedikit untuk K={k} (minimal {k} baris valid). Saat ini tersedia {len(df_all)} baris.")
    elif run or "last_result" in st.session_state:
        if run:
            X = df_all[FEATURES].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = KMeans(n_clusters=k, random_state=42, n_init=20)
            assign = model.fit_predict(X_scaled)

            # urutkan klaster berdasarkan rata-rata nilai centroid (standar), agar label konsisten Rendah->Tinggi
            order = np.argsort(model.cluster_centers_.sum(axis=1))
            rank = {orig: pos for pos, orig in enumerate(order)}
            rank_labels = np.array([rank[a] for a in assign])

            labels_txt = LABELS_BY_K.get(k, [f"Klaster {i+1}" for i in range(k)])

            sil_overall = silhouette_score(X_scaled, assign)
            sil_samples_ = silhouette_samples(X_scaled, assign)

            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            explained = pca.explained_variance_ratio_ * 100

            st.session_state.last_result = {
                "df": df_all.copy(), "X_scaled": X_scaled, "X_pca": X_pca, "explained": explained,
                "rank_labels": rank_labels, "labels_txt": labels_txt,
                "sil_overall": sil_overall, "sil_samples": sil_samples_, "k": k
            }

        r = st.session_state.last_result
        df_r = r["df"].copy()
        df_r["Klaster"] = [f"Klaster {c+1} · {r['labels_txt'][c]}" for c in r["rank_labels"]]
        df_r["_cluster_idx"] = r["rank_labels"]

        st.success(f"Selesai · {len(df_r)} data (dukuh) dikelompokkan ke dalam {r['k']} klaster.")

        # ---------------- Sebaran Klaster (PCA) ----------------
        st.markdown("### Sebaran Hasil Clustering (Proyeksi PCA)")
        st.caption(
            f"Karena clustering menggunakan 5 fitur, sebaran klaster divisualisasikan pada ruang 2 dimensi "
            f"hasil reduksi Principal Component Analysis (PCA), yang menjelaskan sebesar "
            f"±{r['explained'][0] + r['explained'][1]:.0f}% variansi data "
            f"(PC1: {r['explained'][0]:.1f}%, PC2: {r['explained'][1]:.1f}%)."
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in range(r["k"]):
            mask = r["rank_labels"] == c
            ax.scatter(
                r["X_pca"][mask, 0], r["X_pca"][mask, 1],
                color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                label=f"Klaster {c+1} · {r['labels_txt'][c]}",
                s=70, edgecolor="white", alpha=0.85
            )
        ax.axhline(0, color="#D7D2BE", linestyle="--", linewidth=1)
        ax.axvline(0, color="#D7D2BE", linestyle="--", linewidth=1)
        ax.set_xlabel(f"Komponen PCA 1 ({r['explained'][0]:.1f}%)")
        ax.set_ylabel(f"Komponen PCA 2 ({r['explained'][1]:.1f}%)")
        ax.set_title("Sebaran Dukuh Berdasarkan Hasil K-Means (Proyeksi PCA 2D)")
        ax.legend(loc="best", fontsize=8)
        ax.set_facecolor("#F1F0E6")
        fig.patch.set_facecolor("#F1F0E6")
        st.pyplot(fig)

        # ---------------- Karakteristik Centroid ----------------
        st.markdown("### Karakteristik Centroid dan Distribusi Klaster")
        centroid_rows = []
        total_n = len(df_r)
        for c in range(r["k"]):
            sub = df_r[df_r["_cluster_idx"] == c]
            row = {"Kategori Klaster": f"Klaster {c+1} · {r['labels_txt'][c]}"}
            for f in FEATURES:
                row[f] = round(sub[f].mean(), 2) if len(sub) else 0
            row["Rata-rata Total Mustahik"] = round(sub["Total Mustahik"].mean(), 2) if len(sub) else 0
            row["Jumlah Dukuh"] = f"{len(sub)} ({(len(sub) / total_n * 100):.1f}%)" if total_n else "0"
            centroid_rows.append(row)
        st.dataframe(pd.DataFrame(centroid_rows), use_container_width=True, hide_index=True)

        # ---------------- Ringkasan Klaster (kartu) ----------------
        st.markdown("### Ringkasan Klaster")
        cols = st.columns(r["k"])
        for c in range(r["k"]):
            sub = df_r[df_r["_cluster_idx"] == c]
            with cols[c]:
                st.markdown(f"""
                <div class="metric-card" style="border-top-color:{CLUSTER_COLORS[c % len(CLUSTER_COLORS)]}">
                    <b>Klaster {c+1} · {r['labels_txt'][c]}</b><br>
                    <span style="font-size:12px;color:#4A574C;">Jumlah dukuh: {len(sub)}</span><br>
                    <span style="font-size:12px;color:#4A574C;">Rata-rata Total Mustahik: {sub['Total Mustahik'].mean():,.1f}</span>
                </div>
                """, unsafe_allow_html=True)

        # ---------------- Matriks Evaluasi ----------------
        st.markdown("### Matriks Evaluasi Klaster")
        colm1, colm2 = st.columns(2)
        colm1.metric("Silhouette Score", f"{r['sil_overall']:.3f}")
        colm2.metric("Jumlah Klaster (K)", r["k"])

        eval_rows = []
        for c in range(r["k"]):
            mask = r["rank_labels"] == c
            avg_sil = r["sil_samples"][mask].mean() if mask.sum() > 0 else 0
            eval_rows.append({
                "Klaster": f"Klaster {c+1} · {r['labels_txt'][c]}",
                "Jumlah Data": int(mask.sum()),
                "Rata-rata Silhouette": round(avg_sil, 3)
            })
        st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)

        # ---------------- Tabel Detail ----------------
        st.markdown("### Detail Hasil per Data (Dukuh)")
        st.dataframe(
            df_r.drop(columns=["_cluster_idx"]),
            use_container_width=True, hide_index=True
        )

        csv_result = io.StringIO()
        df_r.drop(columns=["_cluster_idx"]).to_csv(csv_result, index=False)
        st.download_button(
            "⬇ Download Hasil Clustering (CSV)", data=csv_result.getvalue(),
            file_name="hasil_clustering_zakat_infak.csv", mime="text/csv"
        )

        st.caption(
            "Metode: K-Means (scikit-learn) dengan fitur Fakir, Miskin, Amil, Sabilillah, dan Ibnu Sabil "
            "distandardisasi menggunakan StandardScaler (Z-score), parameter random_state=42 dan n_init=20. "
            "Evaluasi kualitas klaster menggunakan Silhouette Score. Label klaster (Prioritas Rendah/Sedang/Tinggi, "
            "dst.) ditentukan otomatis dari urutan rata-rata nilai centroid terstandardisasi, konsisten dengan "
            "prosedur pelabelan pada BAB IV."
        )
    else:
        st.info("Klik **▶ Jalankan Clustering** untuk melihat hasil.")
