import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# ======================================================
# 1. Ler dados
# ======================================================
conn = sqlite3.connect("ml_sal.db")
df = pd.read_sql("SELECT * FROM desvios_sal", conn)
conn.close()

df = df.dropna(subset=["dif_pct_sal"]).reset_index(drop=True)

print(f"\n📊 Total de casos disponíveis: {len(df)}")

# ======================================================
# 2. Escolher LOTE
# ======================================================
lote = input("\n🔎 Introduz o LOTE a analisar: ").strip()
df_lote = df[df["lote"] == lote].reset_index(drop=True)

if df_lote.empty:
    print(f"\n❌ Nenhum registo encontrado para o lote '{lote}'")
    print("✅ Análise terminada.")
    exit()

# ======================================================
# 3. Escolher registo do lote
# ======================================================
if len(df_lote) == 1:
    linha = df_lote.iloc[0]
else:
    print(f"\n⚠️ Foram encontrados {len(df_lote)} registos para o lote {lote}:\n")
    for i, r in df_lote.iterrows():
        print(
            f"[{i}] Data: {r['data']} | "
            f"Cuba: {r['cuba']} | "
            f"dif % sal: {r['dif_pct_sal']:.3f}"
        )

    while True:
        escolha = input("\nEscolhe o número do registo: ").strip()
        if escolha.isdigit() and int(escolha) in range(len(df_lote)):
            linha = df_lote.iloc[int(escolha)]
            break
        print("❌ Escolha inválida.")

# ======================================================
# 4. Extrair variáveis chave
# ======================================================
dif_sal = linha["dif_pct_sal"]
dif_es = linha["dif_es"]
dif_hfd = linha["dif_hfd"]

lote = linha["lote"]
cuba = linha["cuba"]
data = linha["data"]

# ======================================================
# 5. Preparar dados para ML
# ======================================================
y = df["dif_pct_sal"]

X = df.drop(columns=[
    "id",
    "data",
    "lote",
    "pct_sal",
    "cuba",
    "referencia",
    "dif_pct_sal"
])

num_cols = [
    "dif_es",
    "dif_hfd",
    "dif_gs",
    "ph_entrada",
    "ph_salga",
    "densidade",
    "temperatura",
    "min_fora",
    "tempo_fora_espec"
]

imputer = SimpleImputer(strategy="median")
X_num = pd.DataFrame(
    imputer.fit_transform(X[num_cols]),
    columns=num_cols
)

# índice global da linha escolhida
row_idx = df.index[
    (df["lote"] == linha["lote"]) &
    (df["data"] == linha["data"]) &
    (df["cuba"] == linha["cuba"])
][0]

# ======================================================
# 6. Modelo
# ======================================================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
model.fit(X_num, y)

# ======================================================
# 7. SHAP LOCAL
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
    "valor_medido": X_num.iloc[row_idx].values
})

impact_df = impact_df[impact_df["percent"] >= 1]
impact_df = impact_df.sort_values("percent", ascending=False)

# ======================================================
# 8. VERIFICAR DOMINÂNCIA DE ES / HFD
# ======================================================
dominante = impact_df.iloc[0]
var_dom = dominante["variavel"]

aplicar_validacao = var_dom in ["dif_es", "dif_hfd"]

inconclusivo = False
motivos = []

if aplicar_validacao:
    # ES dominante → relação inversa
    if var_dom == "dif_es":
        if (dif_sal > 0 and dif_es > 0) or (dif_sal < 0 and dif_es < 0):
            inconclusivo = True
            motivos.append(
                "ES é variável dominante mas o sinal do desvio\n"
                "contraria a relação inversa esperada."
            )

    # HFD dominante → relação direta
    if var_dom == "dif_hfd":
        if (dif_sal > 0 and dif_hfd < 0) or (dif_sal < 0 and dif_hfd > 0):
            inconclusivo = True
            motivos.append(
                "HFD é variável dominante mas o sinal do desvio\n"
                "contraria a relação direta esperada."
            )

# ======================================================
# 9. OUTPUT
# ======================================================
if inconclusivo:
    print("\n📘 ANÁLISE INCONCLUSIVA")
    print("=" * 75)
    print(f"Lote: {lote} | Cuba: {cuba} | Data: {data}")
    print(f"Desvio observado (dif % sal): {dif_sal:.3f}\n")

    for m in motivos:
        print(f"- {m}")

    print("\nConclusão:")
    print("A variável dominante viola pressupostos físicos do processo.")
    print("Este caso não deve ser interpretado via ML.")
    print("\n❌ Nenhum gráfico foi gerado.")
    print("✅ Análise terminada.")
    exit()

# ======================================================
# 10. Gráfico SHAP
# ======================================================
# ======================================================
# 10. Gráfico SHAP (COM valor medido + regra dos 15%)
# ======================================================
# ======================================================
# 10. Gráfico SHAP — impacto DIRECIONAL (versão final)
# ======================================================
# ======================================================
# 10. Gráfico SHAP — impacto DIRECIONAL (legendas sempre visíveis)
# ======================================================
# ======================================================
# Gráfico SHAP — impacto DIRECIONAL (regra 15% + eixo 0)
# ======================================================

# Ordenar por impacto absoluto
impact_df["impacto_abs"] = impact_df["impacto"].abs()
impact_df = impact_df.sort_values("impacto_abs")

# Labels do eixo Y com valor medido
impact_df["label_y"] = impact_df.apply(
    lambda r: f"{r['variavel']}\nvalor = {r['valor_medido']:.3g}",
    axis=1
)

plt.figure(figsize=(10, 6))

# Cores por direção do impacto
colors = impact_df["impacto"].apply(
    lambda x: "#d62728" if x > 0 else "#2ca02c"
)

bars = plt.barh(
    impact_df["label_y"],
    impact_df["impacto"],
    color=colors
)

# Eixo central
plt.axvline(0, color="black", linewidth=0.8)

# Anotações
for bar, impacto, perc in zip(
    bars,
    impact_df["impacto"],
    impact_df["percent"]
):
    y = bar.get_y() + bar.get_height() / 2
    label = f"{impacto:+.3f} ({perc:.1f}%)"

    # =========================
    # ≥ 15% → dentro da barra
    # =========================
    if perc >= 15:
        plt.text(
            impacto * 0.95,
            y,
            label,
            va="center",
            ha="right" if impacto > 0 else "left",
            color="white",
            fontsize=9
        )

    # =========================
    # < 15% → junto ao eixo 0, lado oposto
    # =========================
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
            color="black"
        )

plt.xlabel("Impacto local no desvio previsto (% sal)")
plt.title(
    f"Análise local do desvio de sal\n"
    f"Lote: {lote} | Cuba: {cuba} | dif % sal: {dif_sal:.3f}",
    fontsize=11
)

plt.tight_layout()
plt.savefig("shap_local.png", dpi=150)
plt.close()
# ======================================================
# 11. Interpretação final
# ======================================================
print("\n📘 INTERPRETAÇÃO DO RESULTADO")
print("=" * 75)
print("A análise foi considerada COERENTE.")
print("A variável dominante respeita os pressupostos")
print("físicos do processo, permitindo interpretação.")

print("\n📄 Ficheiro gerado: shap_local.png")
print("✅ Análise concluída")