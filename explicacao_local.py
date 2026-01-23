import sqlite3
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# =========================
# 1. Ler dados
# =========================
conn = sqlite3.connect("ml_sal.db")
df = pd.read_sql("SELECT * FROM desvios_sal", conn)
conn.close()

# =========================
# 2. Limpar dados (target obrigatório)
# =========================
df = df.dropna(subset=["dif_pct_sal"]).reset_index(drop=True)

# =========================
# 3. Informação da matriz final
# =========================
total_linhas = len(df)
print(f"📊 Total de casos disponíveis para análise local: {total_linhas}")
print(f"📌 Índices válidos: 0 até {total_linhas - 1}")

# =========================
# 4. ESCOLHER O CASO A ANALISAR
# =========================
row_id = 1095   # <<< ALTERA APENAS ESTE VALOR

# =========================
# 5. Garantia de segurança
# =========================
if row_id < 0 or row_id >= total_linhas:
    raise ValueError(
        f"row_id inválido ({row_id}). "
        f"Escolhe um valor entre 0 e {total_linhas - 1}"
    )

# =========================
# 6. Guardar info do caso
# =========================
ref = df.iloc[row_id]["referencia"]
cuba = df.iloc[row_id]["cuba"]
lote = df.iloc[row_id]["lote"]
valor_real = df.iloc[row_id]["dif_pct_sal"]

# =========================
# 7. Target
# =========================
y = df["dif_pct_sal"]

# =========================
# 8. Features (modelo causal)
# =========================
X = df.drop(columns=[
    "id",
    "data",
    "lote",
    "pct_sal",
    "cuba",
    "dif_pct_sal"
])

# =========================
# 9. Separar tipos
# =========================
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

cat_cols = [
    "referencia"
]

X_num = X[num_cols]
X_cat = X[cat_cols]

# =========================
# 10. Imputação
# =========================
num_imputer = SimpleImputer(strategy="median")
X_num = pd.DataFrame(
    num_imputer.fit_transform(X_num),
    columns=num_cols
)

cat_imputer = SimpleImputer(strategy="most_frequent")
X_cat = pd.DataFrame(
    cat_imputer.fit_transform(X_cat),
    columns=cat_cols
)

# =========================
# 11. One-Hot Encoding
# =========================
encoder = OneHotEncoder(
    drop="first",
    sparse_output=False
)

X_cat_encoded = pd.DataFrame(
    encoder.fit_transform(X_cat),
    columns=encoder.get_feature_names_out(cat_cols)
)

# =========================
# 12. Dataset final
# =========================
X_final = pd.concat([X_num, X_cat_encoded], axis=1)

# =========================
# 13. Modelo
# =========================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_final, y)

# =========================
# 14. SHAP LOCAL
# =========================
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_final)

# =========================
# 15. Gráfico Waterfall (PNG)
# =========================
plt.figure(figsize=(10, 6))

shap.plots.waterfall(
    shap_values[row_id],
    max_display=10,
    show=False
)

plt.title(
    f"Explicação do desvio de sal\n"
    f"Referência: {ref} | Cuba: {cuba} | Lote: {lote}\n"
    f"Desvio real: {valor_real:.3f}",
    fontsize=11
)

plt.tight_layout()
plt.savefig("shap_local.png", dpi=150)
plt.close()

# =========================
# 16. Output final
# =========================
print("✅ SHAP local gerado com sucesso")
print(f"📄 Ficheiro: shap_local.png")
print(f"📌 Caso analisado → Referência: {ref} | Cuba: {cuba} | Lote: {lote}")
print(f"📈 Desvio real: {valor_real:.3f}")