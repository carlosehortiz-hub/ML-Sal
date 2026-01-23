import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

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
# Features (pct_sal e cuba removidos)
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
# Separar tipos
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
# Imputação
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
# One-Hot Encoding
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
# Dataset final
# =========================
X_final = pd.concat([X_num, X_cat_encoded], axis=1)

# =========================
# Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.2,
    random_state=42
)

print("✅ Dados preparados para ML (modelo causal)")
print("Treino:", X_train.shape)
print("Teste :", X_test.shape)