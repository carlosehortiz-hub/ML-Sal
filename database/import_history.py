import sqlite3

import pandas as pd

UNIQUE_KEY_COLS = ["lote", "referencia", "cuba"]
UNIQUE_INDEX_NAME = "ux_desvios_sal_lote_ref_cuba"


def normalize_unique_key_columns(df):
    df = df.copy()
    for col in UNIQUE_KEY_COLS:
        if col not in df.columns:
            continue
        if col == "cuba":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return df


df = pd.read_csv("matriz_cloretos.csv", sep=None, engine="python")
df.columns = (
    df.columns
    .str.replace("%", "pct", regex=False)
    .str.replace(" ", "_")
    .str.replace("__", "_")
    .str.lower()
)
df = df.rename(columns={"\ufeffdata": "data"})
df = df.replace(["#VALUE!", "#DIV/0!", "#N/A", "N/A", ""], pd.NA)

df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")

percent_cols = ["pct_sal", "dif_pct_sal", "dif_es", "dif_hfd", "dif_gs"]
for col in percent_cols:
    if col not in df.columns:
        continue
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "ph_entrada_salga" in df.columns:
    df["ph_entrada_salga"] = (
        df["ph_entrada_salga"]
        .astype(str)
        .str.upper()
        .map({"OK": 0, "NOK": 1})
    )

num_cols = [
    "ph_salga",
    "densidade_salga",
    "temperatura_salga",
    "min_fora",
    "tempo_fora_espec",
]
for col in num_cols:
    if col not in df.columns:
        continue
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

conn = sqlite3.connect("ml_sal.db")
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS desvios_sal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data TEXT,
        referencia TEXT,
        cuba INTEGER,
        lote TEXT,
        pct_sal REAL,
        dif_pct_sal REAL,
        dif_es REAL,
        dif_hfd REAL,
        dif_gs REAL,
        ph_entrada INTEGER,
        ph_salga REAL,
        densidade REAL,
        temperatura REAL,
        min_fora REAL,
        tempo_fora_espec INTEGER
    )
    """
)
try:
    cursor.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS {UNIQUE_INDEX_NAME}
        ON desvios_sal (lote, referencia, cuba)
        """
    )
except sqlite3.IntegrityError:
    pass
conn.commit()

df_final = df.rename(columns={
    "ph_entrada_salga": "ph_entrada",
    "densidade_salga": "densidade",
    "temperatura_salga": "temperatura",
})

missing_key_cols = [c for c in UNIQUE_KEY_COLS if c not in df_final.columns]
if missing_key_cols:
    conn.close()
    raise ValueError(
        f"CSV is missing required key columns for uniqueness check: {missing_key_cols}"
    )

df_final = normalize_unique_key_columns(df_final)
invalid_key_rows = int(df_final[UNIQUE_KEY_COLS].isna().any(axis=1).sum())
if invalid_key_rows > 0:
    df_final = df_final[~df_final[UNIQUE_KEY_COLS].isna().any(axis=1)].copy()

before_internal = len(df_final)
df_final = df_final.drop_duplicates(subset=UNIQUE_KEY_COLS, keep="last")
skipped_internal = before_internal - len(df_final)

existing_keys = pd.read_sql(
    "SELECT lote, referencia, cuba FROM desvios_sal",
    conn,
)
existing_keys = normalize_unique_key_columns(existing_keys)
existing_keys = existing_keys.drop_duplicates(subset=UNIQUE_KEY_COLS)
existing_keys["_exists"] = 1

merged = df_final.merge(existing_keys, on=UNIQUE_KEY_COLS, how="left")
skipped_existing = int((merged["_exists"] == 1).sum())
df_to_insert = merged[merged["_exists"].isna()].drop(columns=["_exists"])

inserted_rows = len(df_to_insert)
if inserted_rows > 0:
    df_to_insert.to_sql("desvios_sal", conn, if_exists="append", index=False)
conn.close()

print("✅ History imported successfully!")
print(f"📊 Rows inserted: {inserted_rows}")
print(f"⚠️ Rows skipped (invalid key): {invalid_key_rows}")
print(f"⚠️ Rows skipped (duplicate in file): {skipped_internal}")
print(f"⚠️ Rows skipped (already in DB): {skipped_existing}")

print("\n🔎 Missing values by column:")
if inserted_rows > 0:
    print(df_to_insert.isna().sum())
else:
    print("No new rows inserted.")
