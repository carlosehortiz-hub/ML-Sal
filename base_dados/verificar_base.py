import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect("ml_sal.db")

# Read table
df = pd.read_sql("SELECT * FROM desvios_sal", conn)
conn.close()

print("📊 Dataset size:")
print(df.shape)

print("\n📋 First rows:")
print(df.head())

print("\n🔎 Data types:")
print(df.dtypes)

print("\n🎯 Target statistics (dif_pct_sal):")
print(df["dif_pct_sal"].describe())

print("\n❓ Missing values:")
print(df.isna().sum())
