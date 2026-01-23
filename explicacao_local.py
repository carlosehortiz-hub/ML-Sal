import sqlite3
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# =========================
# Ler dados
# =========================
conn = sqlite3.connect("ml_sal.db")
df = pd.read_sql("SELECT * FROM desvios_sal", conn)
conn.close()

df = df.dropna(subset=["dif_pct_sal"]).reset_index(drop=True)

total_linhas = len(df)
print(f"📊 Total de casos disponíveis: {total_linhas}")
print(f"📌 Índices válidos: 0 até {total_linhas - 1}")

# =========================
# Escolher caso
# =========================
row_id = 1095  # <<< ALTERA APENAS ESTE VALOR

if row_id < 0 or row_id >= total_linhas:
    raise ValueError("row_id inválido")

valor_real = df.iloc[row_id]["dif_pct_sal"]

# =========================
# Target
# =========================
y = df["dif_pct_sal"]

# =========================
# Features (sem referencia)
# =========================
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

# =========================
# Imputação
# =========================
imputer = SimpleImputer(strategy="median")
X_final = pd.DataFrame(
    imputer.fit_transform(X[num_cols]),
    columns=num_cols
)

# =========================
# Modelo
# =========================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_final, y)

# =========================
# SHAP LOCAL
# =========================
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_final)

plt.figure(figsize=(10, 6))
shap.plots.waterfall(
    shap_values[row_id],
    max_display=10,
    show=False
)

plt.title(
    f"Explicação do desvio de sal\n"
    f"Desvio real: {valor_real:.3f}",
    fontsize=11
)

plt.tight_layout()
plt.savefig("shap_local.png", dpi=150)
plt.close()

print("✅ SHAP local (sem referencia) gerado com sucesso")