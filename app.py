import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

from ml.ml_utils import (
    NUM_COLS,
    load_data,
    build_features,
    get_model,
    transform_features,
)

st.set_page_config(page_title="ML-Sal", layout="wide")


def check_inconclusive(impact_df, dif_sal, dif_es, dif_hfd):
    if impact_df.empty:
        return False, []

    dominante = impact_df.iloc[0]
    var_dom = dominante["variavel"]

    motivos = []

    if var_dom == "dif_es":
        if (dif_sal > 0 and dif_es > 0) or (dif_sal < 0 and dif_es < 0):
            motivos.append(
                "ES is the dominant variable, but the deviation sign "
                "contradicts the expected inverse relationship."
            )

    if var_dom == "dif_hfd":
        if (dif_sal > 0 and dif_hfd < 0) or (dif_sal < 0 and dif_hfd > 0):
            motivos.append(
                "HFD is the dominant variable, but the deviation sign "
                "contradicts the expected direct relationship."
            )

    return len(motivos) > 0, motivos


def build_impact_df(model, imputer, row_values):
    X_row = pd.DataFrame([row_values], columns=NUM_COLS)
    X_row_num = transform_features(imputer, X_row)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_row_num)

    vals = shap_values.values[0]
    abs_vals = np.abs(vals)
    total = abs_vals.sum() if abs_vals.sum() != 0 else 1.0
    percent = abs_vals / total * 100

    impact_df = pd.DataFrame({
        "variavel": X_row_num.columns,
        "impacto": vals,
        "percent": percent,
        "valor_medido": X_row_num.iloc[0].values,
    })

    impact_df = impact_df[impact_df["percent"] >= 1]
    impact_df = impact_df.sort_values("percent", ascending=False)

    return impact_df, X_row_num


def plot_impact(impact_df, ph_entrada_value):
    if impact_df.empty:
        st.warning("No variables above 1% impact.")
        return

    impact_df_plot = impact_df.copy()
    if ph_entrada_value == 0:
        impact_df_plot = impact_df_plot[
            impact_df_plot["variavel"] != "ph_entrada"
        ].copy()
        if not impact_df_plot.empty:
            total_plot = impact_df_plot["impacto"].abs().sum()
            impact_df_plot["percent"] = (
                impact_df_plot["impacto"].abs() / total_plot * 100
            )

    if impact_df_plot.empty:
        st.warning("All variables removed after filtering.")
        return

    impact_df_plot["impacto_abs"] = impact_df_plot["impacto"].abs()
    impact_df_plot = impact_df_plot.sort_values("impacto_abs")

    impact_df_plot["label_y"] = impact_df_plot.apply(
        lambda r: f"{r['variavel']}\nvalue = {r['valor_medido']:.3g}",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = impact_df_plot["impacto"].apply(
        lambda x: "#d62728" if x > 0 else "#2ca02c"
    )

    bars = ax.barh(
        impact_df_plot["label_y"],
        impact_df_plot["impacto"],
        color=colors,
    )

    ax.axvline(0, color="black", linewidth=0.8)

    for bar, impacto, perc in zip(
        bars,
        impact_df_plot["impacto"],
        impact_df_plot["percent"],
    ):
        y = bar.get_y() + bar.get_height() / 2
        label = f"{impacto:+.3f} ({perc:.1f}%)"

        if perc >= 15:
            ax.text(
                impacto * 0.95,
                y,
                label,
                va="center",
                ha="right" if impacto > 0 else "left",
                color="white",
                fontsize=9,
            )
        else:
            offset = 0.002
            x_text = -offset if impacto > 0 else offset
            ax.text(
                x_text,
                y,
                label,
                va="center",
                ha="right" if impacto > 0 else "left",
                fontsize=9,
                color="black",
            )

    ax.set_xlabel("Local impact on predicted deviation (salt %)")
    st.pyplot(fig)


def validate_float(raw, label, min_v=None, max_v=None):
    if raw is None or raw.strip() == "":
        return None, f"{label}: value cannot be empty."

    value_str = raw.strip().replace(",", ".")
    try:
        value = float(value_str)
    except Exception:
        return None, f"{label}: invalid value."

    if min_v is not None and value < min_v:
        return None, f"{label}: must be ≥ {min_v}."

    if max_v is not None and value > max_v:
        return None, f"{label}: must be ≤ {max_v}."

    return value, None


def validate_int(raw, label, min_v=None, max_v=None):
    if raw is None or raw.strip() == "":
        return None, f"{label}: value cannot be empty."

    try:
        value = int(raw.strip())
    except Exception:
        return None, f"{label}: invalid integer."

    if min_v is not None and value < min_v:
        return None, f"{label}: must be ≥ {min_v}."

    if max_v is not None and value > max_v:
        return None, f"{label}: must be ≤ {max_v}."

    return value, None


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model_cached(df):
    X, y = build_features(df)
    model, imputer = get_model(X, y)
    return model, imputer


st.title("ML-Sal — Local Explanation")

mode = st.radio(
    "Choose a mode",
    ("Use existing record", "Manual input (no DB write)"),
)


df = get_data()

if df.empty:
    st.error("No data available in the database.")
    st.stop()

model, imputer = get_model_cached(df)

if mode == "Use existing record":
    st.subheader("Select a record")

    lotes = sorted(df["lote"].dropna().unique())
    if not lotes:
        st.error("No batches available.")
        st.stop()

    lote = st.selectbox("Batch", lotes)

    df_lote = df[df["lote"] == lote].copy()
    df_lote = df_lote.reset_index()

    st.dataframe(
        df_lote[["index", "data", "cuba", "dif_pct_sal"]],
        use_container_width=True,
    )

    def label_row(r):
        return f"{r['index']} | {r['data']} | Vat {r['cuba']} | diff {r['dif_pct_sal']:.3f}"

    options = df_lote.apply(label_row, axis=1).tolist()
    selected = st.selectbox("Record", options)
    row_idx = int(selected.split("|")[0].strip())

    row = df.loc[row_idx]

    row_values = row[NUM_COLS].values
    impact_df, X_row_num = build_impact_df(model, imputer, row_values)

    dif_sal = row["dif_pct_sal"]
    dif_es = row["dif_es"]
    dif_hfd = row["dif_hfd"]

    st.markdown("---")
    st.write(
        f"**Batch:** {row['lote']} | **Vat:** {row['cuba']} | **Date:** {row['data']}"
    )
    st.write(f"**Observed deviation:** {dif_sal:.3f}")

    inconclusivo, motivos = check_inconclusive(impact_df, dif_sal, dif_es, dif_hfd)

    if inconclusivo:
        st.error("INCONCLUSIVE ANALYSIS")
        for m in motivos:
            st.write(f"- {m}")
    else:
        st.success("COHERENT ANALYSIS")
        plot_impact(impact_df, row["ph_entrada"])

else:
    st.subheader("Manual input (not saved)")

    references = sorted(df["referencia"].dropna().unique())

    with st.form("manual_form"):
        st.markdown("**General data**")
        col1, col2, col3 = st.columns(3)
        with col1:
            date = st.date_input("Date")
        with col2:
            if references:
                reference = st.selectbox("Reference", references)
            else:
                reference = st.text_input("Reference")
        with col3:
            cuba_raw = st.text_input("Vat (integer)")

        lote = st.text_input("Batch")

        st.markdown("**Main variables**")
        col1, col2 = st.columns(2)
        with col1:
            pct_sal_raw = st.text_input("Measured salt % (e.g.: 1.60)")
        with col2:
            dif_pct_sal_raw = st.text_input("Salt % difference (TARGET) (e.g.: -0.040)")

        st.markdown("**Process variables**")
        col1, col2, col3 = st.columns(3)
        with col1:
            dif_es_raw = st.text_input("Dif_ES (e.g.: 0.90)")
            ph_entrada = st.selectbox(
                "Input pH",
                [0, 1],
                format_func=lambda x: "OK" if x == 0 else "NOK",
            )
            densidade_raw = st.text_input("Density (e.g.: 18.8)")
        with col2:
            dif_hfd_raw = st.text_input("Dif_HFD (e.g.: -1.10)")
            ph_salga_raw = st.text_input("Brine pH (e.g.: 5.05)")
            temperatura_raw = st.text_input("Temperature (°C) (e.g.: 10.7)")
        with col3:
            dif_gs_raw = st.text_input("Dif_GS (e.g.: 0.20)")
            min_fora_raw = st.text_input("Minutes out of spec (e.g.: 25)")
            tempo_fora_espec = st.selectbox(
                "Time out of spec",
                [0, 1],
                format_func=lambda x: "In" if x == 0 else "Out",
            )

        submitted = st.form_submit_button("Run analysis")

    if submitted:
        errors = []

        if lote.strip() == "":
            errors.append("Batch cannot be empty.")

        if reference is None or str(reference).strip() == "":
            errors.append("Reference cannot be empty.")

        cuba, err = validate_int(cuba_raw, "Vat", min_v=1)
        if err:
            errors.append(err)

        pct_sal, err = validate_float(pct_sal_raw, "Measured salt %", min_v=0)
        if err:
            errors.append(err)

        dif_pct_sal, err = validate_float(dif_pct_sal_raw, "Salt % difference")
        if err:
            errors.append(err)

        dif_es, err = validate_float(dif_es_raw, "Dif_ES")
        if err:
            errors.append(err)

        dif_hfd, err = validate_float(dif_hfd_raw, "Dif_HFD")
        if err:
            errors.append(err)

        dif_gs, err = validate_float(dif_gs_raw, "Dif_GS")
        if err:
            errors.append(err)

        ph_salga, err = validate_float(ph_salga_raw, "Brine pH", min_v=3.5, max_v=7.5)
        if err:
            errors.append(err)

        densidade, err = validate_float(densidade_raw, "Density", min_v=0)
        if err:
            errors.append(err)

        temperatura, err = validate_float(temperatura_raw, "Temperature (°C)", min_v=-5, max_v=40)
        if err:
            errors.append(err)

        min_fora, err = validate_float(min_fora_raw, "Minutes out of spec", min_v=0)
        if err:
            errors.append(err)

        if errors:
            for e in errors:
                st.error(e)
        else:
            row_values = [
                dif_es,
                dif_hfd,
                dif_gs,
                ph_entrada,
                ph_salga,
                densidade,
                temperatura,
                min_fora,
                tempo_fora_espec,
            ]

            impact_df, X_row_num = build_impact_df(model, imputer, row_values)
            pred = model.predict(X_row_num)[0]

            st.markdown("---")
            st.write(
                f"**Batch:** {lote} | **Vat:** {cuba} | **Date:** {date}"
            )
            st.write(f"**Observed deviation:** {dif_pct_sal:.3f}")
            st.write(f"**Predicted deviation:** {pred:.3f}")

            inconclusivo, motivos = check_inconclusive(
                impact_df, dif_pct_sal, dif_es, dif_hfd
            )

            if inconclusivo:
                st.error("INCONCLUSIVE ANALYSIS")
                for m in motivos:
                    st.write(f"- {m}")
            else:
                st.success("COHERENT ANALYSIS")
                plot_impact(impact_df, ph_entrada)
