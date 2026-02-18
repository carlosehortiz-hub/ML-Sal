import sqlite3
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# =========================
# Read data
# =========================
conn = sqlite3.connect("ml_sal.db")
df = pd.read_sql("SELECT * FROM desvios_sal", conn)
conn.close()

# Remove rows without target
df = df.dropna(subset=["dif_pct_sal"])

# =========================
# Target
# =========================
y = df["dif_pct_sal"]

# =========================
# Features (process variables only)
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
# Numeric variables
# =========================
num_cols = [
    "dif_es",
    "dif_gs",
    "ph_entrada",
    "ph_salga",
    "densidade",
    "min_fora",
]

X_num = X[num_cols]

# =========================
# Imputation
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

print("✅ Data prepared for ML (without reference)")
print("Train:", X_train.shape)
print("Test :", X_test.shape)
