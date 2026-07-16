# ---------------- Distribusi Boxplot per Asnaf ----------------
        st.markdown("### Distribusi Jumlah Mustahik per Asnaf pada Setiap Kategori")
        st.caption(
            "Boxplot berikut menunjukkan sebaran (distribusi) nilai tiap fitur asnaf "
            "(Fakir, Miskin, Amil, Sabilillah, Ibnu Sabil) untuk masing-masing kategori/label klaster."
        )

        # ubah data ke bentuk long/tidy agar bisa di-plot per asnaf x kategori
        df_melt = df_r.melt(
            id_vars=["_cluster_idx"],
            value_vars=FEATURES,
            var_name="Asnaf",
            value_name="Jumlah Mustahik"
        )
        df_melt["Kategori"] = df_melt["_cluster_idx"].map(lambda c: r["labels_txt"][c])

        # urutan tampilan: asnaf sesuai FEATURES, kategori sesuai LABELS_BY_K (Rendah -> Tinggi)
        asnaf_order = FEATURES
        kategori_order = r["labels_txt"]

        fig_box, ax_box = plt.subplots(figsize=(9, 5))
        sns.boxplot(
            data=df_melt,
            x="Asnaf", y="Jumlah Mustahik", hue="Kategori",
            order=asnaf_order, hue_order=kategori_order,
            ax=ax_box
        )
        ax_box.set_title("Distribusi Jumlah Mustahik per Asnaf pada Setiap Kategori")
        ax_box.set_xlabel("Kategori Asnaf")
        ax_box.set_ylabel("Jumlah Mustahik")
        ax_box.legend(title="Kategori", loc="upper right")
        fig_box.tight_layout()
        st.pyplot(fig_box)
