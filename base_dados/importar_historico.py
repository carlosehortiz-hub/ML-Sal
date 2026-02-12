import pandas as pd
import sqlite3

# =========================
# 1. Read CSV
# =========================
df = pd.read_csv(
    "matriz_cloretos.csv",
    sep=None,
    engine="python"
)

# =========================
# 2. Normalize column names
# =========================
df.columns = (
    df.columns
    .str.replace('%', 'pct', regex=False)
    .str.replace(' ', '_')
    .str.replace('__', '_')
    .str.lower()
)

# Remove hidden BOM in the date column
df = df.rename(columns={'\ufeffdata': 'data'})

# =========================
# 3. Clean typical Excel errors
# =========================
df = df.replace(
    ['#VALUE!', '#DIV/0!', '#N/A', 'N/A', ''],
    pd.NA
)

# =========================
# 4. Convert dates
# =========================
df['data'] = pd.to_datetime(
    df['data'],
    dayfirst=True,
    errors='coerce'
)

# =========================
# 5. Convert percentages to float
# =========================
percent_cols = [
    'pct_sal',
    'dif_pct_sal',
    'dif_es',
    'dif_hfd',
    'dif_gs'
]

for col in percent_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace('%', '', regex=False)
        .str.replace(',', '.', regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

# =========================
# 6. Handle pH input (OK / NOK status)
# =========================
df['ph_entrada_salga'] = (
    df['ph_entrada_salga']
    .astype(str)
    .str.upper()
    .map({'OK': 0, 'NOK': 1})
)

# =========================
# 7. Convert remaining numeric columns
# =========================
num_cols = [
    'ph_salga',
    'densidade_salga',
    'temperatura_salga',
    'min_fora',
    'tempo_fora_espec'
]

for col in num_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '.', regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

# =========================
# 8. Connect to SQLite database
# =========================
conn = sqlite3.connect("ml_sal.db")
cursor = conn.cursor()

# =========================
# 9. Create table (if it does not exist)
# =========================
cursor.execute("""
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
""")

conn.commit()

# =========================
# 10. Adjust final names for the DB
# =========================
df_final = df.rename(columns={
    'ph_entrada_salga': 'ph_entrada',
    'densidade_salga': 'densidade',
    'temperatura_salga': 'temperatura'
})

# =========================
# 11. Insert data into the database
# =========================
df_final.to_sql(
    'desvios_sal',
    conn,
    if_exists='append',
    index=False
)

conn.close()

# =========================
# 12. Final messages
# =========================
print("✅ History imported successfully!")
print(f"📊 Total rows imported: {len(df_final)}")

print("\n🔎 Missing values by column:")
print(df_final.isna().sum())
