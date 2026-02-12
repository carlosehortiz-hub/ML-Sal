import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from ml_utils import load_data, build_features, get_model, transform_features

# ======================================================
# 1. Read data
# ======================================================
df = load_data()

print(f"\n📊 Total available cases: {len(df)}")

# ======================================================
# 2. Choose BATCH
# ======================================================
lote = input("\n🔎 Enter the BATCH to analyze: ").strip()
df_lote = df[df["lote"] == lote].reset_index(drop=True)

if df_lote.empty:
    print(f"\n❌ No records found for batch '{lote}'")
    print("✅ Analysis finished.")
    exit()

# ======================================================
# 3. Choose batch record
# ======================================================
if len(df_lote) == 1:
    linha = df_lote.iloc[0]
else:
    print(f"\n⚠️ Found {len(df_lote)} records for batch {lote}:\n")
    for i, r in df_lote.iterrows():
        print(
            f"[{i}] Date: {r['data']} | "
            f"Tank: {r['cuba']} | "
            f"salt % diff: {r['dif_pct_sal']:.3f}"
        )

    while True:
        escolha = input("\nChoose the record number: ").strip()
        if escolha.isdigit() and int(escolha) in range(len(df_lote)):
            linha = df_lote.iloc[int(escolha)]
            break
        print("❌ Invalid choice.")

# ======================================================
# 4. Extract key variables
# ======================================================
dif_sal = linha["dif_pct_sal"]
dif_es = linha["dif_es"]
dif_hfd = linha["dif_hfd"]

lote = linha["lote"]
cuba = linha["cuba"]
data = linha["data"]

# ======================================================
# 5. Prepare data for ML
# ======================================================
X, y = build_features(df)

model, imputer = get_model(X, y)
X_num = transform_features(imputer, X)

# Global index of the selected row
row_idx = df.index[
    (df["lote"] == linha["lote"]) &
    (df["data"] == linha["data"]) &
    (df["cuba"] == linha["cuba"])
][0]

# ======================================================
# 6. LOCAL SHAP
# ======================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_num)

vals = shap_values.values[row_idx]
abs_vals = np.abs(vals)

total = abs_vals.sum()
percent = abs_vals / total * 100

impact_df = pd.DataFrame({
    "variavel": X_num.columns,
    "impacto": vals,
    "percent": percent,
    "valor_medido": X_num.iloc[row_idx].values,
})

impact_df = impact_df[impact_df["percent"] >= 1]
impact_df = impact_df.sort_values("percent", ascending=False)

# ======================================================
# 7. CHECK ES / HFD DOMINANCE
# ======================================================
dominante = impact_df.iloc[0]
var_dom = dominante["variavel"]

aplicar_validacao = var_dom in ["dif_es", "dif_hfd"]

inconclusivo = False
motivos = []

if aplicar_validacao:
    # ES dominant → inverse relationship
    if var_dom == "dif_es":
        if (dif_sal > 0 and dif_es > 0) or (dif_sal < 0 and dif_es < 0):
            inconclusivo = True
            motivos.append(
                "ES is the dominant variable, but the deviation sign\n"
                "contradicts the expected inverse relationship."
            )

    # HFD dominant → direct relationship
    if var_dom == "dif_hfd":
        if (dif_sal > 0 and dif_hfd < 0) or (dif_sal < 0 and dif_hfd > 0):
            inconclusivo = True
            motivos.append(
                "HFD is the dominant variable, but the deviation sign\n"
                "contradicts the expected direct relationship."
            )

# ======================================================
# 8. OUTPUT
# ======================================================
if inconclusivo:
    print("\n📘 INCONCLUSIVE ANALYSIS")
    print("=" * 75)
    print(f"Batch: {lote} | Tank: {cuba} | Date: {data}")
    print(f"Observed deviation (salt % diff): {dif_sal:.3f}\n")

    for m in motivos:
        print(f"- {m}")

    print("\nConclusion:")
    print("The dominant variable violates physical process assumptions.")
    print("This case should not be interpreted via ML.")
    print("\n❌ No chart was generated.")
    print("✅ Analysis finished.")
    exit()

# ======================================================
# 9. SHAP Chart
# ======================================================
# Remove ph_entrada from the chart when the record value is 0 (OK)
impact_df_plot = impact_df.copy()
if linha.get("ph_entrada") == 0:
    impact_df_plot = impact_df_plot[impact_df_plot["variavel"] != "ph_entrada"].copy()
    if not impact_df_plot.empty:
        total_plot = impact_df_plot["impacto"].abs().sum()
        impact_df_plot["percent"] = impact_df_plot["impacto"].abs() / total_plot * 100

# Sort by absolute impact
impact_df_plot["impacto_abs"] = impact_df_plot["impacto"].abs()
impact_df_plot = impact_df_plot.sort_values("impacto_abs")

# Y-axis labels with measured value
impact_df_plot["label_y"] = impact_df_plot.apply(
    lambda r: f"{r['variavel']}\nvalue = {r['valor_medido']:.3g}",
    axis=1,
)

plt.figure(figsize=(10, 6))

# Colors by impact direction
colors = impact_df_plot["impacto"].apply(
    lambda x: "#d62728" if x > 0 else "#2ca02c"
)

bars = plt.barh(
    impact_df_plot["label_y"],
    impact_df_plot["impacto"],
    color=colors,
)

# Central axis
plt.axvline(0, color="black", linewidth=0.8)

# Annotations
for bar, impacto, perc in zip(
    bars,
    impact_df_plot["impacto"],
    impact_df_plot["percent"],
):
    y = bar.get_y() + bar.get_height() / 2
    label = f"{impacto:+.3f} ({perc:.1f}%)"

    # ≥ 15% → inside the bar
    if perc >= 15:
        plt.text(
            impacto * 0.95,
            y,
            label,
            va="center",
            ha="right" if impacto > 0 else "left",
            color="white",
            fontsize=9,
        )

    # < 15% → near the zero axis, opposite side
    else:
        offset = 0.002
        x_text = -offset if impacto > 0 else offset

        plt.text(
            x_text,
            y,
            label,
            va="center",
            ha="right" if impacto > 0 else "left",
            fontsize=9,
            color="black",
        )

plt.xlabel("Local impact on predicted deviation (salt %)")
plt.title(
    f"Local analysis of salt deviation\n"
    f"Batch: {lote} | Tank: {cuba} | salt % diff: {dif_sal:.3f}",
    fontsize=11,
)

plt.tight_layout()
plt.savefig("shap_local.png", dpi=150)
plt.close()

# ======================================================
# 10. Final interpretation
# ======================================================
print("\n📘 RESULT INTERPRETATION")
print("=" * 75)
print("The analysis was considered COHERENT.")
print("The dominant variable respects the")
print("physical process assumptions, allowing interpretation.")

print("\n📄 File generated: shap_local.png")
print("✅ Analysis completed")
