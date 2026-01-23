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

df = df.dropna(subset=["dif_pct_sal"])

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
# SHAP GLOBAL
# =========================
sample = X_final.sample(
    n=min(300, len(X_final)),
    random_state=42
)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    sample,
    plot_type="bar",
    show=False
)

plt.tight_layout()
plt.savefig("shap_global.png", dpi=150)
plt.close()

print("✅ SHAP global (sem referencia) guardado em shap_global.png")