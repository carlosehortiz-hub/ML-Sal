import sqlite3
from datetime import datetime
import sys

# ======================================================
# FULLY SAFE INPUT FUNCTIONS
# ======================================================

def input_float(mensagem, exemplo=None, minimo=None, maximo=None):
    while True:
        texto = mensagem
        if exemplo:
            texto += f" (e.g.: {exemplo})"
        texto += " → use dot or comma: "

        valor_str = input(texto)

        if valor_str is None:
            print("\n❌ Empty input.\n")
            continue

        valor_str = valor_str.strip()
        if valor_str == "":
            print("\n❌ Value cannot be empty.\n")
            continue

        valor_str = valor_str.replace(",", ".")

        try:
            valor = float(valor_str)
        except Exception:
            print(
                "\n❌ Invalid value."
                "\n➡️ Enter numbers only."
                "\n➡️ Valid examples: 1.60 | 1,60 | -0.045\n"
            )
            continue

        if minimo is not None and valor < minimo:
            print(f"\n❌ Invalid value. Must be ≥ {minimo}.\n")
            continue

        if maximo is not None and valor > maximo:
            print(f"\n❌ Invalid value. Must be ≤ {maximo}.\n")
            continue

        return valor


def input_int(mensagem, exemplo=None, minimo=None, maximo=None):
    while True:
        texto = mensagem
        if exemplo:
            texto += f" (e.g.: {exemplo})"
        texto += ": "

        valor_str = input(texto)

        if valor_str is None:
            print("\n❌ Empty input.\n")
            continue

        valor_str = valor_str.strip()
        if valor_str == "":
            print("\n❌ Value cannot be empty.\n")
            continue

        try:
            valor = int(valor_str)
        except Exception:
            print(
                "\n❌ Invalid value."
                "\n➡️ Enter an integer only.\n"
            )
            continue

        if minimo is not None and valor < minimo:
            print(f"\n❌ Invalid value. Must be ≥ {minimo}.\n")
            continue

        if maximo is not None and valor > maximo:
            print(f"\n❌ Invalid value. Must be ≤ {maximo}.\n")
            continue

        return valor


def input_data():
    while True:
        data = input("Date (YYYY-MM-DD) [Enter = today]: ")

        if data is None:
            print("\n❌ Invalid input.\n")
            continue

        data = data.strip()

        if data == "":
            return datetime.today().strftime("%Y-%m-%d")

        try:
            datetime.strptime(data, "%Y-%m-%d")
            return data
        except Exception:
            print(
                "\n❌ Invalid date."
                "\n➡️ Use YYYY-MM-DD format.\n"
            )


def input_choice(mensagem, opcoes, descricao=None):
    while True:
        if descricao:
            print(descricao)

        valor = input(f"{mensagem} {opcoes}: ")

        if valor is None:
            print("\n❌ Invalid input.\n")
            continue

        valor = valor.strip()

        if valor in opcoes:
            return int(valor)

        print(
            f"\n❌ Invalid value."
            f"\n➡️ Choose one of the options {opcoes}.\n"
        )


def input_referencia(cursor):
    cursor.execute(
        "SELECT DISTINCT referencia FROM desvios_sal ORDER BY referencia"
    )
    refs = [r[0] for r in cursor.fetchall()]

    if not refs:
        print("\n❌ There are no references in the database.")
        sys.exit(1)

    print("\n📚 Valid references:")
    for r in refs:
        print(f" - {r}")

    while True:
        ref = input("\nReference (copy exactly from the list): ")

        if ref is None:
            print("\n❌ Invalid input.\n")
            continue

        ref = ref.strip()

        if ref == "":
            print("\n❌ Reference cannot be empty.\n")
            continue

        if ref in refs:
            return ref

        print(
            "\n❌ Invalid reference."
            "\n➡️ It must match exactly one of the listed references."
            "\n➡️ Watch out for spaces, uppercase letters, and accents.\n"
        )


# ======================================================
# DATABASE CONNECTION
# ======================================================

conn = sqlite3.connect("ml_sal.db")
cursor = conn.cursor()

print("\n📥 MANUAL INSERT OF NEW DEVIATION")
print("=" * 60)

# ======================================================
# GENERAL DATA
# ======================================================

data = input_data()
referencia = input_referencia(cursor)

cuba = input_int(
    "Vat (integer)",
    exemplo="12",
    minimo=1
)

lote = ""
while lote == "":
    lote = input("Batch (free text): ")
    if lote is None:
        lote = ""
    lote = lote.strip()
    if lote == "":
        print("\n❌ Batch cannot be empty.\n")

# ======================================================
# MAIN VARIABLES
# ======================================================

pct_sal = input_float("Measured salt %", exemplo="1.60", minimo=0)
dif_pct_sal = input_float("Salt % difference (TARGET)", exemplo="-0.040")

# ======================================================
# PROCESS VARIABLES
# ======================================================

dif_es = input_float("Dif_ES", exemplo="0.90")
dif_hfd = input_float("Dif_HFD", exemplo="-1.10")
dif_gs = input_float("Dif_GS", exemplo="0.20")

ph_entrada = input_choice(
    "Input pH",
    ["0", "1"],
    descricao="0 = OK | 1 = NOK"
)

ph_salga = input_float(
    "Brine pH",
    exemplo="5.05",
    minimo=3.5,
    maximo=7.5
)

densidade = input_float("Density", exemplo="18.8", minimo=0)
temperatura = input_float(
    "Temperature (°C)",
    exemplo="10.7",
    minimo=-5,
    maximo=40
)

min_fora = input_float(
    "Minutes out of spec",
    exemplo="25"
)

tempo_fora_espec = input_choice(
    "Time out of spec",
    ["0", "1"],
    descricao="0 = In | 1 = Out"
)

# ======================================================
# FINAL CONFIRMATION
# ======================================================

print("\n📋 DATA SUMMARY")
print("-" * 60)

print(f"Date               : {data}")
print(f"Reference          : {referencia}")
print(f"Vat                : {cuba}")
print(f"Batch              : {lote}")
print(f"Salt %             : {pct_sal}")
print(f"Salt % diff (target): {dif_pct_sal}")
print(f"Dif_ES             : {dif_es}")
print(f"Dif_HFD            : {dif_hfd}")
print(f"Dif_GS             : {dif_gs}")
print(f"Input pH (0/1)     : {ph_entrada}")
print(f"Brine pH           : {ph_salga}")
print(f"Density            : {densidade}")
print(f"Temperature        : {temperatura}")
print(f"Minutes out        : {min_fora}")
print(f"Time out spec      : {tempo_fora_espec}")

confirmar = input("\nConfirm insertion? (y/n): ")
if confirmar is None or confirmar.strip().lower() != "y":
    print("\n❌ Insertion canceled.")
    conn.close()
    sys.exit(0)

# ======================================================
# INSERT INTO DATABASE
# ======================================================

cursor.execute(
    """
    INSERT INTO desvios_sal (
        data, referencia, cuba, lote,
        pct_sal, dif_pct_sal,
        dif_es, dif_hfd, dif_gs,
        ph_entrada, ph_salga,
        densidade, temperatura,
        min_fora, tempo_fora_espec
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
        data, referencia, cuba, lote,
        pct_sal, dif_pct_sal,
        dif_es, dif_hfd, dif_gs,
        ph_entrada, ph_salga,
        densidade, temperatura,
        min_fora, tempo_fora_espec
    )
)

conn.commit()
conn.close()

print("\n✅ New deviation inserted successfully!")
