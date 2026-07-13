"""
Analisis Pola Penyaluran Dana Zakat dan Infak
K-Means Clustering — NU CARE Kabupaten Wonosobo
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import io

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Analisis Penyaluran Zakat & Infak — NU CARE Wonosobo",
    page_icon="📊",
    layout="wide"
)

# ======================================================
# PALET WARNA & FONT (mengikuti referensi desain)
# ======================================================
CLUSTER_COLORS = ["#B9872C", "#2F6B4F", "#8A4B3B", "#3E5A78", "#7A5B8C"]
LABELS_BY_K = {
    2: ["Rendah", "Tinggi"],
    3: ["Rendah", "Sedang", "Tinggi"],
    4: ["Sangat Rendah", "Rendah", "Tinggi", "Sangat Tinggi"],
    5: ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"],
}

KECAMATAN_LIST = [
    "Wonosobo", "Kertek", "Selomerto", "Leksono", "Sukoharjo", "Watumalang",
    "Mojotengah", "Garung", "Kejajar", "Kalikajar", "Kepil", "Sapuran",
    "Kaliwiro", "Wadaslintang", "Kalibawang"
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
  html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
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
  <div class="sub">NU CARE – LAZISNU Kabupaten Wonosobo · Mengelompokkan pola penyaluran berdasarkan jumlah dana dan jumlah mustahik.</div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# STATE AWAL
# ======================================================
def seed_data():
    periods = ["Januari 2026", "Februari 2026", "Maret 2026"]
    rows = [
        ("Wonosobo", 0, 22000000, 95), ("Wonosobo", 1, 24500000, 101), ("Wonosobo", 2, 26000000, 110),
        ("Kertek", 0, 9500000, 38), ("Kertek", 1, 10200000, 41),
        ("Selomerto", 0, 8700000, 35), ("Selomerto", 2, 9100000, 37),
        ("Leksono", 0, 4200000, 17), ("Leksono", 1, 4600000, 19),
        ("Sukoharjo", 0, 3800000, 15),
        ("Watumalang", 0, 5200000, 21), ("Watumalang", 2, 5500000, 23),
        ("Mojotengah", 0, 12500000, 50), ("Mojotengah", 1, 13100000, 53),
        ("Garung", 0, 6300000, 26),
        ("Kejajar", 0, 3100000, 12), ("Kejajar", 2, 3400000, 13),
        ("Kalikajar", 0, 7600000, 31),
        ("Kepil", 0, 4900000, 20),
        ("Sapuran", 0, 6100000, 25), ("Sapuran", 1, 6400000, 26),
        ("Kaliwiro", 0, 3500000, 14),
        ("Wadaslintang", 0, 2800000, 11),
        ("Kalibawang", 0, 2600000, 10),
    ]
    return pd.DataFrame([
        {"Kecamatan": r[0], "Periode": periods[r[1]], "Dana (Rp)": r[2], "Mustahik": r[3]}
        for r in rows
    ])

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
        st.caption("Satu baris data mewakili satu penyaluran pada suatu kecamatan & periode.")
        with st.form("form_tambah", clear_on_submit=True):
            kecamatan = st.selectbox("Kecamatan / Wilayah", KECAMATAN_LIST)
            periode = st.text_input("Periode", placeholder="Januari 2026")
            dana = st.number_input("Jumlah Dana Disalurkan (Rp)", min_value=0, step=100000, value=0)
            mustahik = st.number_input("Jumlah Mustahik (orang)", min_value=0, step=1, value=0)
            submitted = st.form_submit_button("+ Tambah Data", use_container_width=True)

            if submitted:
                if not periode.strip() or dana <= 0 or mustahik <= 0:
                    st.error("Mohon lengkapi semua isian dengan nilai yang valid.")
                else:
                    new_row = pd.DataFrame([{
                        "Kecamatan": kecamatan, "Periode": periode.strip(),
                        "Dana (Rp)": dana, "Mustahik": mustahik
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

        uploaded = st.file_uploader(
            "⬆ Impor Data (CSV atau Excel — kolom: Kecamatan, Periode, Dana (Rp), Mustahik)",
            type=["csv", "xlsx", "xls"]
        )
        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    imported = pd.read_csv(uploaded)
                else:
                    imported = pd.read_excel(uploaded)

                required_cols = {"Kecamatan", "Periode", "Dana (Rp)", "Mustahik"}
                if not required_cols.issubset(set(imported.columns)):
                    st.error(
                        "Kolom pada file tidak sesuai. Pastikan file memiliki kolom: "
                        "Kecamatan, Periode, Dana (Rp), Mustahik."
                    )
                else:
                    imported = imported[["Kecamatan", "Periode", "Dana (Rp)", "Mustahik"]]
                    st.session_state.data = pd.concat([st.session_state.data, imported], ignore_index=True)
                    st.success(f"{len(imported)} baris berhasil diimpor dari {uploaded.name}.")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    with col_table:
        st.subheader(f"Data Penyaluran ({len(st.session_state.data)} baris)")
        st.caption("Kamu bisa edit langsung di tabel (klik sel), atau hapus baris lewat ikon 🗑 di ujung kiri baris.")

        edited = st.data_editor(
            st.session_state.data,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_data",
            column_config={
                "Dana (Rp)": st.column_config.NumberColumn(format="Rp %d"),
                "Mustahik": st.column_config.NumberColumn(format="%d orang"),
            }
        )
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
    df = st.session_state.data.dropna()
    df = df[(df["Dana (Rp)"] > 0) & (df["Mustahik"] > 0)]

    st.subheader("Pengaturan K-Means")
    st.caption(
        "Fitur yang distandardisasi (StandardScaler): **Jumlah Dana** dan **Jumlah Mustahik**. "
        "Kecamatan digunakan sebagai label wilayah pada tiap titik data."
    )

    colk, colbtn = st.columns([3, 1])
    with colk:
        k = st.select_slider("Jumlah Klaster (k)", options=[2, 3, 4, 5], value=3)
    with colbtn:
        run = st.button("▶ Jalankan Clustering", use_container_width=True)

    if len(df) < k:
        st.warning(f"Data terlalu sedikit untuk k={k} (minimal {k} baris valid). Saat ini tersedia {len(df)} baris.")
    elif run or "last_result" in st.session_state:
        if run:
            X = df[["Dana (Rp)", "Mustahik"]].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            assign = model.fit_predict(X_scaled)

            # urutkan klaster berdasarkan rata-rata (dana+mustahik) standar, agar label konsisten Rendah->Tinggi
            order = np.argsort(model.cluster_centers_.sum(axis=1))
            rank = {orig: pos for pos, orig in enumerate(order)}
            rank_labels = np.array([rank[a] for a in assign])

            labels_txt = LABELS_BY_K.get(k, [f"Klaster {i+1}" for i in range(k)])
            sil_overall = silhouette_score(X_scaled, assign)
            sil_samples = silhouette_samples(X_scaled, assign)

            st.session_state.last_result = {
                "df": df.copy(), "X_scaled": X_scaled, "rank_labels": rank_labels,
                "labels_txt": labels_txt, "sil_overall": sil_overall,
                "sil_samples": sil_samples, "k": k
            }

        r = st.session_state.last_result
        df_r = r["df"].copy()
        df_r["Klaster"] = [f"Klaster {c+1} · {r['labels_txt'][c]}" for c in r["rank_labels"]]
        df_r["_cluster_idx"] = r["rank_labels"]

        st.success(f"Selesai · {len(df_r)} data dikelompokkan ke dalam {r['k']} klaster.")

        # ---------------- Peta Klaster ----------------
        st.markdown("### Peta Klaster")
        st.caption("Posisi tiap titik berdasarkan Jumlah Dana (X) dan Jumlah Mustahik (Y) yang telah distandardisasi (Z-score).")

        fig, ax = plt.subplots(figsize=(8, 5))
        for c in range(r["k"]):
            mask = r["rank_labels"] == c
            ax.scatter(
                r["X_scaled"][mask, 0], r["X_scaled"][mask, 1],
                color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                label=f"Klaster {c+1} · {r['labels_txt'][c]}",
                s=70, edgecolor="white", alpha=0.85
            )
        ax.axhline(0, color="#D7D2BE", linestyle="--", linewidth=1)
        ax.axvline(0, color="#D7D2BE", linestyle="--", linewidth=1)
        ax.set_xlabel("Jumlah Dana (Z-score)")
        ax.set_ylabel("Jumlah Mustahik (Z-score)")
        ax.legend(loc="best", fontsize=8)
        ax.set_facecolor("#F1F0E6")
        fig.patch.set_facecolor("#F1F0E6")
        st.pyplot(fig)

        # ---------------- Ringkasan Klaster ----------------
        st.markdown("### Ringkasan Klaster")
        cols = st.columns(r["k"])
        for c in range(r["k"]):
            sub = df_r[df_r["_cluster_idx"] == c]
            with cols[c]:
                st.markdown(f"""
                <div class="metric-card" style="border-top-color:{CLUSTER_COLORS[c % len(CLUSTER_COLORS)]}">
                    <b>Klaster {c+1} · {r['labels_txt'][c]}</b><br>
                    <span style="font-size:12px;color:#4A574C;">Jumlah data: {len(sub)}</span><br>
                    <span style="font-size:12px;color:#4A574C;">Rata-rata dana: Rp {sub['Dana (Rp)'].mean():,.0f}</span><br>
                    <span style="font-size:12px;color:#4A574C;">Rata-rata mustahik: {sub['Mustahik'].mean():,.0f}</span>
                </div>
                """, unsafe_allow_html=True)

        # ---------------- Matriks Evaluasi ----------------
        st.markdown("### Matriks Evaluasi Klaster")
        colm1, colm2 = st.columns(2)
        colm1.metric("Silhouette Score", f"{r['sil_overall']:.3f}")
        colm2.metric("Jumlah Klaster (k)", r["k"])

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
        st.markdown("### Detail Hasil per Data")
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
            "Metode: K-Means (scikit-learn) dengan fitur distandardisasi menggunakan StandardScaler "
            "(Z-score) agar skala Rupiah dan jumlah orang setara bobotnya. Evaluasi kualitas klaster "
            "menggunakan Silhouette Score. Label klaster (Rendah/Sedang/Tinggi, dst.) ditentukan otomatis "
            "dari urutan rata-rata nilai centroid."
        )
    else:
        st.info("Klik **▶ Jalankan Clustering** untuk melihat hasil.")
