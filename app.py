"""
Aplikasi Analisis Pola Penyaluran Dana Zakat dan Infak
Menggunakan Algoritma K-Means Clustering
NU CARE Kabupaten Wonosobo
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import io

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Clustering Penyaluran Zakat & Infak",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analisis Pola Penyaluran Dana Zakat dan Infak")
st.markdown("**K-Means Clustering — NU CARE Kabupaten Wonosobo**")
st.markdown("---")

# ======================================================
# SESSION STATE
# ======================================================
if "df" not in st.session_state:
    st.session_state.df = None
if "clustered_df" not in st.session_state:
    st.session_state.clustered_df = None

# ======================================================
# SIDEBAR - UPLOAD DATA
# ======================================================
st.sidebar.header("1️⃣ Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload file data penyaluran (CSV atau Excel)",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.sidebar.success(f"Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file: {e}")

# ======================================================
# JIKA DATA SUDAH ADA
# ======================================================
if st.session_state.df is not None:
    df = st.session_state.df

    # --------------------------------------------------
    # PREVIEW DATA
    # --------------------------------------------------
    st.header("📄 Preview Data")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Total data: {df.shape[0]} baris | {df.shape[1]} kolom")

    with st.expander("Lihat statistik deskriptif"):
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------
    # PEMILIHAN FITUR / ATRIBUT
    # --------------------------------------------------
    st.header("⚙️ 2. Pemilihan Atribut untuk Clustering")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Data numerik yang tersedia kurang dari 2 kolom. Clustering membutuhkan minimal 2 atribut numerik.")
    else:
        selected_features = st.multiselect(
            "Pilih atribut/kolom numerik yang akan digunakan untuk clustering "
            "(misal: fakir, miskin, amil, mualaf, riqob, ghorim, sabilillah, ibnu sabil)",
            options=numeric_cols,
            default=numeric_cols
        )

        # --------------------------------------------------
        # PENANGANAN MISSING VALUE
        # --------------------------------------------------
        handle_na = st.radio(
            "Data kosong (missing value) ditangani dengan:",
            ["Hapus baris yang kosong", "Isi dengan nilai 0", "Isi dengan rata-rata kolom"],
            horizontal=True
        )

        if len(selected_features) >= 2:
            data = df[selected_features].copy()

            if handle_na == "Hapus baris yang kosong":
                data = data.dropna()
            elif handle_na == "Isi dengan nilai 0":
                data = data.fillna(0)
            else:
                data = data.fillna(data.mean(numeric_only=True))

            valid_index = data.index

            # --------------------------------------------------
            # STANDARISASI DATA
            # --------------------------------------------------
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            st.markdown("---")

            # --------------------------------------------------
            # 3. PENENTUAN JUMLAH CLUSTER
            # --------------------------------------------------
            st.header("🔍 3. Penentuan Jumlah Cluster (k)")

            k_mode = st.radio(
                "Metode penentuan jumlah cluster:",
                ["Otomatis (Elbow Method & Silhouette Score)", "Manual"],
                horizontal=True
            )

            max_k = st.slider("Batas maksimal k yang diuji", min_value=3, max_value=min(10, len(data) - 1), value=8)

            if max_k < 2 or len(data) < 4:
                st.warning("Data terlalu sedikit untuk melakukan clustering yang bermakna.")
            else:
                k_range = range(2, max_k + 1)
                inertias = []
                silhouettes = []

                for k in k_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels_k = km.fit_predict(data_scaled)
                    inertias.append(km.inertia_)
                    silhouettes.append(silhouette_score(data_scaled, labels_k))

                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = plt.subplots()
                    ax.plot(list(k_range), inertias, marker="o", color="#2563eb")
                    ax.set_xlabel("Jumlah Cluster (k)")
                    ax.set_ylabel("Inertia (WCSS)")
                    ax.set_title("Elbow Method")
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)

                with col2:
                    fig2, ax2 = plt.subplots()
                    ax2.plot(list(k_range), silhouettes, marker="o", color="#16a34a")
                    ax2.set_xlabel("Jumlah Cluster (k)")
                    ax2.set_ylabel("Silhouette Score")
                    ax2.set_title("Silhouette Score")
                    ax2.grid(alpha=0.3)
                    st.pyplot(fig2)

                best_k_auto = list(k_range)[int(np.argmax(silhouettes))]
                st.info(f"💡 Rekomendasi k berdasarkan Silhouette Score tertinggi: **k = {best_k_auto}** "
                        f"(Silhouette Score = {max(silhouettes):.4f})")

                if k_mode == "Otomatis (Elbow Method & Silhouette Score)":
                    final_k = best_k_auto
                else:
                    final_k = st.number_input(
                        "Masukkan jumlah cluster (k) secara manual",
                        min_value=2, max_value=max_k, value=best_k_auto, step=1
                    )

                st.markdown("---")

                # --------------------------------------------------
                # 4. PROSES K-MEANS CLUSTERING
                # --------------------------------------------------
                st.header(f"🧩 4. Hasil K-Means Clustering (k = {final_k})")

                kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
                cluster_labels = kmeans_final.fit_predict(data_scaled)
                final_silhouette = silhouette_score(data_scaled, cluster_labels)

                result_df = df.loc[valid_index].copy()
                result_df["Cluster"] = cluster_labels
                st.session_state.clustered_df = result_df

                st.metric("Silhouette Score Akhir", f"{final_silhouette:.4f}")

                st.dataframe(result_df, use_container_width=True)

                st.markdown("---")

                # --------------------------------------------------
                # 5. VISUALISASI CLUSTER
                # --------------------------------------------------
                st.header("📈 5. Visualisasi Cluster")

                tab1, tab2 = st.tabs(["Scatter Plot Cluster", "Karakteristik Tiap Cluster"])

                with tab1:
                    if len(selected_features) == 2:
                        fig3, ax3 = plt.subplots(figsize=(7, 5))
                        scatter = ax3.scatter(
                            data[selected_features[0]], data[selected_features[1]],
                            c=cluster_labels, cmap="tab10", s=60, edgecolor="k", alpha=0.8
                        )
                        ax3.set_xlabel(selected_features[0])
                        ax3.set_ylabel(selected_features[1])
                        ax3.set_title("Visualisasi Cluster (2 Atribut)")
                        legend1 = ax3.legend(*scatter.legend_elements(), title="Cluster")
                        ax3.add_artist(legend1)
                        st.pyplot(fig3)
                    else:
                        pca = PCA(n_components=2, random_state=42)
                        components = pca.fit_transform(data_scaled)
                        fig3, ax3 = plt.subplots(figsize=(7, 5))
                        scatter = ax3.scatter(
                            components[:, 0], components[:, 1],
                            c=cluster_labels, cmap="tab10", s=60, edgecolor="k", alpha=0.8
                        )
                        ax3.set_xlabel("Komponen Utama 1 (PCA)")
                        ax3.set_ylabel("Komponen Utama 2 (PCA)")
                        ax3.set_title("Visualisasi Cluster (Reduksi Dimensi PCA)")
                        legend1 = ax3.legend(*scatter.legend_elements(), title="Cluster")
                        ax3.add_artist(legend1)
                        st.pyplot(fig3)
                        st.caption(
                            f"Karena atribut lebih dari 2, data direduksi menjadi 2 dimensi menggunakan PCA "
                            f"(variansi terjelaskan: {pca.explained_variance_ratio_.sum() * 100:.1f}%)."
                        )

                with tab2:
                    cluster_summary = result_df.groupby("Cluster")[selected_features].mean()
                    st.dataframe(cluster_summary.style.background_gradient(cmap="Blues"), use_container_width=True)

                    fig4, ax4 = plt.subplots(figsize=(9, 5))
                    cluster_summary.plot(kind="bar", ax=ax4)
                    ax4.set_title("Rata-rata Nilai Atribut per Cluster")
                    ax4.set_ylabel("Nilai Rata-rata")
                    ax4.set_xlabel("Cluster")
                    ax4.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                    plt.xticks(rotation=0)
                    st.pyplot(fig4)

                    cluster_counts = result_df["Cluster"].value_counts().sort_index()
                    st.write("**Jumlah anggota per cluster:**")
                    st.bar_chart(cluster_counts)

                st.markdown("---")

                # --------------------------------------------------
                # 6. DOWNLOAD HASIL
                # --------------------------------------------------
                st.header("⬇️ 6. Download Hasil Clustering")

                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Hasil Clustering (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="hasil_clustering_zakat_infak.csv",
                    mime="text/csv"
                )

                summary_buffer = io.StringIO()
                cluster_summary.to_csv(summary_buffer)
                st.download_button(
                    label="Download Ringkasan Karakteristik Cluster (CSV)",
                    data=summary_buffer.getvalue(),
                    file_name="ringkasan_karakteristik_cluster.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Pilih minimal 2 atribut numerik untuk melanjutkan proses clustering.")

else:
    st.info("👈 Silakan upload file data penyaluran (CSV/Excel) melalui sidebar untuk memulai.")
    st.markdown("""
    **Format data yang disarankan:**
    Setiap baris merepresentasikan satu transaksi/periode penyaluran, dengan kolom numerik seperti:
    - `fakir`, `miskin`, `amil`, `mualaf`, `riqob`, `ghorim`, `sabilillah`, `ibnu_sabil`
    - Kolom lain (nama program, periode, dll) boleh ada dan tidak akan ikut dihitung dalam clustering
      kecuali dipilih secara manual.
    """)
