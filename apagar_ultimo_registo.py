import sqlite3

conn = sqlite3.connect("ml_sal.db")
cursor = conn.cursor()

# Ver último registo
cursor.execute("""
SELECT id, data, referencia, lote, dif_pct_sal
FROM desvios_sal
ORDER BY id DESC
LIMIT 1
""")

registo = cursor.fetchone()

if registo is None:
    print("❌ Não existem registos para apagar.")
    conn.close()
    exit()

print("\n⚠️ ÚLTIMO REGISTO INSERIDO:")
print(f"ID        : {registo[0]}")
print(f"Data      : {registo[1]}")
print(f"Referência: {registo[2]}")
print(f"Lote      : {registo[3]}")
print(f"Desvio    : {registo[4]}")

confirmar = input("\nConfirmar apagamento deste registo? (s/n): ").strip().lower()

if confirmar == "s":
    cursor.execute(
        "DELETE FROM desvios_sal WHERE id = ?",
        (registo[0],)
    )
    conn.commit()
    print("\n✅ Registo apagado com sucesso.")
else:
    print("\n❌ Operação cancelada.")

conn.close()