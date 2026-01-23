import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# =========================
# Ler dados
# =========================
conn = sqlite3.connect("ml_sal.db")
df = pd.read_sql("SELECT * FROM desvios_sal", conn)
conn.close()

# remover linhas sem target
df = df.dropna(subset=["dif_pct_sal"])

# =========================
# Target
# =========================
y = df["dif_pct_sal"]

# =========================
# Features (apenas variáveis de processo)
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

# =========================
# Variáveis numéricas
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

X_num = X[num_cols]

# =========================
# Imputação
# =========================
num_imputer = SimpleImputer(strategy="median")
X_final = pd.DataFrame(
    num_imputer.fit_transform(X_num),
    columns=num_cols
)

# =========================
# Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.2,
    random_state=42
)

print("✅ Dados preparados para ML (sem referencia)")
print("Treino:", X_train.shape)
print("Teste :", X_test.shape)