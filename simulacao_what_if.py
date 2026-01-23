import sqlite3
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# =========================
# Ler dados
# =========================
conn = sqlite3.connect("ml_sal.db")
df = pd.read_sql("SELECT * FROM desvios_sal", conn)
conn.close()

df = df.dropna(subset=["dif_pct_sal"]).reset_index(drop=True)

print(f"\n📊 Total de casos disponíveis: {len(df)}")
row_id = int(input("Escolhe o row_id do caso base: "))

if row_id < 0 or row_id >= len(df):
    raise ValueError("row_id inválido")

desvio_real = df.iloc[row_id]["dif_pct_sal"]

# =========================
# Target
# =========================
y = df["dif_pct_sal"]

# =========================
# Features (sem referencia)
# =========================
X = df.drop(columns=[
    "id",
    "data",
    "lote",
    "pct_sal",
    "cuba",
    "referencia",
    "dif_pct_sal"
])

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

# =========================
# Imputação
# =========================
imputer = SimpleImputer(strategy="median")
X_final = pd.DataFrame(
    imputer.fit_transform(X[num_cols]),
    columns=num_cols
)

# =========================
# Modelo
# =========================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_final, y)

# =========================
# Caso base
# =========================
x_base = X_final.iloc[row_id].copy()
desvio_base = model.predict(pd.DataFrame([x_base]))[0]

print("\n📌 CASO BASE")
print(f"Desvio REAL      : {desvio_real:.3f}")
print(f"Desvio PREVISTO  : {desvio_base:.3f}")

# =========================
# Loop interativo
# =========================
variaveis = list(x_base.index)

while True:
    x_sim = x_base.copy()
    alteracoes = {}

    print("\nVariáveis disponíveis:")
    for v in variaveis:
        print(f" - {v}")

    while True:
        var = input("\nVariável a simular (ou 'fim'): ").strip()
        if var.lower() == "fim":
            break
        if var not in x_sim.index:
            print("❌ Variável inválida")
            continue

        atual = x_sim[var]
        novo = float(input("Novo valor: "))
        x_sim[var] = novo
        alteracoes[var] = (atual, novo)

        if input("Alterar mais variáveis? (s/n): ").lower() != "s":
            break

    if alteracoes:
        desvio_sim = model.predict(pd.DataFrame([x_sim]))[0]
        impacto = desvio_sim - desvio_base

        print("\n📊 RESULTADO")
        for v, (a, n) in alteracoes.items():
            print(f"{v}: {a} → {n}")
        print(f"Desvio previsto: {desvio_sim:.3f}")
        print(f"Impacto vs base: {impacto:+.3f}")

    if input("\nNova simulação? (s/n): ").lower() != "s":
        break