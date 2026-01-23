import sqlite3
import pandas as pd

# ligar à base
conn = sqlite3.connect("ml_sal.db")

# ler tabela
df = pd.read_sql("SELECT * FROM desvios_sal", conn)
conn.close()

print("📊 Dimensão do dataset:")
print(df.shape)

print("\n📋 Primeiras linhas:")
print(df.head())

print("\n🔎 Tipos de dados:")
print(df.dtypes)

print("\n🎯 Estatísticas do target (dif_pct_sal):")
print(df["dif_pct_sal"].describe())

print("\n❓ Valores em falta:")
print(df.isna().sum())