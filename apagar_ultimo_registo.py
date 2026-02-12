import sqlite3

conn = sqlite3.connect("ml_sal.db")
cursor = conn.cursor()

# View last record
cursor.execute("""
SELECT id, data, referencia, lote, dif_pct_sal
FROM desvios_sal
ORDER BY id DESC
LIMIT 1
""")

registo = cursor.fetchone()

if registo is None:
    print("❌ There are no records to delete.")
    conn.close()
    exit()

print("\n⚠️ LAST INSERTED RECORD:")
print(f"ID        : {registo[0]}")
print(f"Date      : {registo[1]}")
print(f"Reference : {registo[2]}")
print(f"Batch     : {registo[3]}")
print(f"Deviation : {registo[4]}")

confirmar = input("\nConfirm deletion of this record? (y/n): ").strip().lower()

if confirmar == "y":
    cursor.execute(
        "DELETE FROM desvios_sal WHERE id = ?",
        (registo[0],)
    )
    conn.commit()
    print("\n✅ Record deleted successfully.")
else:
    print("\n❌ Operation canceled.")

conn.close()
