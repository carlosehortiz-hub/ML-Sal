import sqlite3
import pandas as pd

# ======================================================
# Connect to the database
# ======================================================
conn = sqlite3.connect("ml_sal.db")

# ======================================================
# Read the latest inserted records
# ======================================================
query = """
SELECT *
FROM desvios_sal
ORDER BY id DESC
LIMIT 5
"""

df = pd.read_sql(query, conn)
conn.close()

# ======================================================
# Show result
# ======================================================
if df.empty:
    print("❌ The table is empty. No records found.")
else:
    print("\n✅ Latest records inserted into the database:")
    print("=" * 60)
    print(df)
