import sqlite3
import pandas as pd

# ======================================================
# Ligar à base de dados
# ======================================================
conn = sqlite3.connect("ml_sal.db")

# ======================================================
# Ler os últimos registos inseridos
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
# Mostrar resultado
# ======================================================
if df.empty:
    print("❌ A tabela está vazia. Nenhum registo encontrado.")
else:
    print("\n✅ Últimos registos inseridos na base de dados:")
    print("=" * 60)
    print(df)