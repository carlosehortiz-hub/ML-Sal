import sqlite3

conn = sqlite3.connect("ml_sal.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS desvios_sal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    data TEXT,
    referencia TEXT,
    cuba INTEGER,
    lote TEXT,

    dif_pct_sal REAL,
    pct_sal REAL,

    ph_entrada REAL,
    ph_salga REAL,
    densidade REAL,
    temperatura REAL,

    observacoes TEXT
);
""")

conn.commit()
conn.close()

print("Table 'desvios_sal' created successfully.")
