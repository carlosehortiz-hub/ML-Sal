import sqlite3
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# ======================================================
# 1. Ler dados
# ======================================================
conn = sqlite3.connect("ml_sal.db")
df = pd.read_sql("SELECT * FROM desvios_sal", conn)
conn.close()

df = df.dropna(subset=["dif_pct_sal"]).reset_index(drop=True)

# ======================================================
# 2. Escolher caso base
# ======================================================
print(f"\n📊 Total de casos disponíveis: {len(df)}")
row_id = int(input("Escolhe o row_id do caso base: "))

if row_id < 0 or row_id >= len(df):
    raise ValueError("row_id inválido")

# ======================================================
# 3. Informação do caso
# ======================================================
ref = df.iloc[row_id]["referencia"]
cuba = df.iloc[row_id]["cuba"]
lote = df.iloc[row_id]["lote"]
desvio_real = df.iloc[row_id]["dif_pct_sal"]

# ======================================================
# 4. Target
# ======================================================
y = df["dif_pct_sal"]

# ======================================================
# 5. Features (modelo causal)
# ======================================================
X = df.drop(columns=[
    "id",
    "data",
    "lote",
    "pct_sal",
    "cuba",
    "dif_pct_sal"
])

# ======================================================
# 6. Separar tipos
# ======================================================
num_cols = [
    "dif_es",
    "dif_hfd",
    "dif_gs",
    "ph_entrada",
    "ph_salga",
    "densidade",
    "temperatura",
    "min_fora",
    "tempo_fora_espec"
]

cat_cols = ["referencia"]

X_num = X[num_cols]
X_cat = X[cat_cols]

# ======================================================
# 7. Imputação
# ======================================================
num_imputer = SimpleImputer(strategy="median")
X_num = pd.DataFrame(
    num_imputer.fit_transform(X_num),
    columns=num_cols
)

cat_imputer = SimpleImputer(strategy="most_frequent")
X_cat = pd.DataFrame(
    cat_imputer.fit_transform(X_cat),
    columns=cat_cols
)

# ======================================================
# 8. One-Hot Encoding
# ======================================================
encoder = OneHotEncoder(drop="first", sparse_output=False)

X_cat_encoded = pd.DataFrame(
    encoder.fit_transform(X_cat),
    columns=encoder.get_feature_names_out(cat_cols)
)

# ======================================================
# 9. Dataset final
# ======================================================
X_final = pd.concat([X_num, X_cat_encoded], axis=1)

# ======================================================
# 10. Modelo
# ======================================================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_final, y)

# ======================================================
# 11. Caso base
# ======================================================
x_base = X_final.iloc[row_id].copy()
desvio_previsto_base = model.predict(
    pd.DataFrame([x_base])
)[0]

print("\n" + "=" * 70)
print("📌 CASO BASE")
print("=" * 70)
print(f"Referência              : {ref}")
print(f"Cuba                    : {cuba}")
print(f"Lote                    : {lote}")
print(f"Desvio REAL medido      : {desvio_real:.3f}")
print(f"Desvio PREVISTO (base)  : {desvio_previsto_base:.3f}")

# ======================================================
# 12. Loop de simulações
# ======================================================
variaveis_simulaveis = list(x_base.index)

while True:
    print("\n" + "=" * 70)
    print("🔬 NOVA SIMULAÇÃO WHAT-IF")
    print("=" * 70)

    x_sim = x_base.copy()
    alteracoes = {}

    print("\nVariáveis disponíveis:")
    for v in variaveis_simulaveis:
        print(f" - {v}")

    while True:
        var = input("\nQual variável queres simular? (ou 'fim'): ").strip()

        if var.lower() == "fim":
            break

        if var not in x_sim.index:
            print("❌ Variável inválida")
            continue

        valor_atual = x_sim[var]
        print(f"Valor atual usado pelo modelo: {valor_atual}")

        try:
            novo_valor = float(input("Novo valor: "))
        except ValueError:
            print("❌ Valor inválido")
            continue

        x_sim[var] = novo_valor
        alteracoes[var] = (valor_atual, novo_valor)

        mais = input("Queres alterar mais variáveis nesta simulação? (s/n): ")
        if mais.lower() != "s":
            break

    if not alteracoes:
        print("⚠️ Nenhuma variável alterada. Simulação ignorada.")
    else:
        desvio_sim = model.predict(pd.DataFrame([x_sim]))[0]
        impacto = desvio_sim - desvio_previsto_base

        print("\n📊 RESULTADO DA SIMULAÇÃO")
        print("-" * 40)

        for var, (antigo, novo) in alteracoes.items():
            print(f"{var}: {antigo} → {novo}")

        print(f"\nDesvio PREVISTO (simulado): {desvio_sim:.3f}")
        print(f"Impacto vs BASE           : {impacto:+.3f}")

    repetir = input("\nQueres fazer outra simulação? (s/n): ")
    if repetir.lower() != "s":
        print("\n✅ Simulações terminadas.")
        break