import sqlite3
from datetime import datetime
import sys

# ======================================================
# FUNÇÕES DE INPUT TOTALMENTE SEGURAS
# ======================================================

def input_float(mensagem, exemplo=None, minimo=None, maximo=None):
    while True:
        texto = mensagem
        if exemplo:
            texto += f" (ex: {exemplo})"
        texto += " → usa ponto ou vírgula: "

        valor_str = input(texto)

        if valor_str is None:
            print("\n❌ Entrada vazia.\n")
            continue

        valor_str = valor_str.strip()
        if valor_str == "":
            print("\n❌ Valor não pode estar vazio.\n")
            continue

        valor_str = valor_str.replace(",", ".")

        try:
            valor = float(valor_str)
        except Exception:
            print(
                "\n❌ Valor inválido."
                "\n➡️ Introduz apenas números."
                "\n➡️ Exemplos válidos: 1.60 | 1,60 | -0.045\n"
            )
            continue

        if minimo is not None and valor < minimo:
            print(f"\n❌ Valor inválido. Deve ser ≥ {minimo}.\n")
            continue

        if maximo is not None and valor > maximo:
            print(f"\n❌ Valor inválido. Deve ser ≤ {maximo}.\n")
            continue

        return valor


def input_int(mensagem, exemplo=None, minimo=None, maximo=None):
    while True:
        texto = mensagem
        if exemplo:
            texto += f" (ex: {exemplo})"
        texto += ": "

        valor_str = input(texto)

        if valor_str is None:
            print("\n❌ Entrada vazia.\n")
            continue

        valor_str = valor_str.strip()
        if valor_str == "":
            print("\n❌ Valor não pode estar vazio.\n")
            continue

        try:
            valor = int(valor_str)
        except Exception:
            print(
                "\n❌ Valor inválido."
                "\n➡️ Introduz apenas um número inteiro.\n"
            )
            continue

        if minimo is not None and valor < minimo:
            print(f"\n❌ Valor inválido. Deve ser ≥ {minimo}.\n")
            continue

        if maximo is not None and valor > maximo:
            print(f"\n❌ Valor inválido. Deve ser ≤ {maximo}.\n")
            continue

        return valor


def input_data():
    while True:
        data = input("Data (YYYY-MM-DD) [Enter = hoje]: ")

        if data is None:
            print("\n❌ Entrada inválida.\n")
            continue

        data = data.strip()

        if data == "":
            return datetime.today().strftime("%Y-%m-%d")

        try:
            datetime.strptime(data, "%Y-%m-%d")
            return data
        except Exception:
            print(
                "\n❌ Data inválida."
                "\n➡️ Usa o formato YYYY-MM-DD.\n"
            )


def input_choice(mensagem, opcoes, descricao=None):
    while True:
        if descricao:
            print(descricao)

        valor = input(f"{mensagem} {opcoes}: ")

        if valor is None:
            print("\n❌ Entrada inválida.\n")
            continue

        valor = valor.strip()

        if valor in opcoes:
            return int(valor)

        print(
            f"\n❌ Valor inválido."
            f"\n➡️ Escolhe uma das opções {opcoes}.\n"
        )


def input_referencia(cursor):
    cursor.execute(
        "SELECT DISTINCT referencia FROM desvios_sal ORDER BY referencia"
    )
    refs = [r[0] for r in cursor.fetchall()]

    if not refs:
        print("\n❌ Não existem referências na base de dados.")
        sys.exit(1)

    print("\n📚 Referências válidas:")
    for r in refs:
        print(f" - {r}")

    while True:
        ref = input("\nReferência (copiar exatamente da lista): ")

        if ref is None:
            print("\n❌ Entrada inválida.\n")
            continue

        ref = ref.strip()

        if ref == "":
            print("\n❌ Referência não pode estar vazia.\n")
            continue

        if ref in refs:
            return ref

        print(
            "\n❌ Referência inválida."
            "\n➡️ Deve coincidir exatamente com uma das referências listadas."
            "\n➡️ Atenção a espaços, maiúsculas e acentos.\n"
        )


# ======================================================
# LIGAÇÃO À BASE DE DADOS
# ======================================================

conn = sqlite3.connect("ml_sal.db")
cursor = conn.cursor()

print("\n📥 INSERÇÃO MANUAL DE NOVO DESVIO")
print("=" * 60)

# ======================================================
# DADOS GERAIS
# ======================================================

data = input_data()
referencia = input_referencia(cursor)

cuba = input_int(
    "Cuba (inteiro)",
    exemplo="12",
    minimo=1
)

lote = ""
while lote == "":
    lote = input("Lote (texto livre): ")
    if lote is None:
        lote = ""
    lote = lote.strip()
    if lote == "":
        print("\n❌ Lote não pode estar vazio.\n")

# ======================================================
# VARIÁVEIS PRINCIPAIS
# ======================================================

pct_sal = input_float("% Sal medido", exemplo="1.60", minimo=0)
dif_pct_sal = input_float("Diferença % sal (TARGET)", exemplo="-0.040")

# ======================================================
# VARIÁVEIS DE PROCESSO
# ======================================================

dif_es = input_float("Dif_ES", exemplo="0.90")
dif_hfd = input_float("Dif_HFD", exemplo="-1.10")
dif_gs = input_float("Dif_GS", exemplo="0.20")

ph_entrada = input_choice(
    "pH entrada",
    ["0", "1"],
    descricao="0 = NOK | 1 = OK"
)

ph_salga = input_float(
    "pH salga",
    exemplo="5.05",
    minimo=3.5,
    maximo=7.5
)

densidade = input_float("Densidade", exemplo="18.8", minimo=0)
temperatura = input_float(
    "Temperatura (°C)",
    exemplo="10.7",
    minimo=-5,
    maximo=40
)

min_fora = input_float(
    "Minutos fora de especificação",
    exemplo="25",
    minimo=0
)

tempo_fora_espec = input_choice(
    "Tempo fora de especificação",
    ["0", "1"],
    descricao="0 = Dentro | 1 = Fora"
)

# ======================================================
# CONFIRMAÇÃO FINAL
# ======================================================

print("\n📋 RESUMO DOS DADOS")
print("-" * 60)

print(f"Data               : {data}")
print(f"Referência         : {referencia}")
print(f"Cuba               : {cuba}")
print(f"Lote               : {lote}")
print(f"% Sal              : {pct_sal}")
print(f"Dif % Sal (target) : {dif_pct_sal}")
print(f"Dif_ES             : {dif_es}")
print(f"Dif_HFD            : {dif_hfd}")
print(f"Dif_GS             : {dif_gs}")
print(f"pH entrada (0/1)   : {ph_entrada}")
print(f"pH salga           : {ph_salga}")
print(f"Densidade          : {densidade}")
print(f"Temperatura        : {temperatura}")
print(f"Min fora           : {min_fora}")
print(f"Tempo fora espec   : {tempo_fora_espec}")

confirmar = input("\nConfirmar inserção? (s/n): ")
if confirmar is None or confirmar.strip().lower() != "s":
    print("\n❌ Inserção cancelada.")
    conn.close()
    sys.exit(0)

# ======================================================
# INSERÇÃO NA BASE DE DADOS
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

print("\n✅ Novo desvio inserido com sucesso!")