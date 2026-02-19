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
);
""")
try:
    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_desvios_sal_lote_ref_cuba
        ON desvios_sal (lote, referencia, cuba)
        """
    )
except sqlite3.IntegrityError:
    pass

conn.commit()
conn.close()

print("Table 'desvios_sal' created successfully.")
