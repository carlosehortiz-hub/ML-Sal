import pandas as pd

from ml.ml_utils import load_data, build_features, get_model, transform_features

# =========================
# Read data
# =========================
df = load_data()

print(f"\n📊 Total available cases: {len(df)}")
row_id = int(input("Choose the row_id for the base case: "))

if row_id < 0 or row_id >= len(df):
    raise ValueError("Invalid row_id")

desvio_real = df.iloc[row_id]["dif_pct_sal"]

# =========================
# Features (without reference)
# =========================
X, y = build_features(df)

# =========================
# Model (cached)
# =========================
model, imputer = get_model(X, y)
X_num = transform_features(imputer, X)

# =========================
# Base case
# =========================
x_base = X_num.iloc[row_id].copy()
desvio_base = model.predict(pd.DataFrame([x_base]))[0]

print("\n📌 BASE CASE")
print(f"REAL deviation   : {desvio_real:.3f}")
print(f"PREDICTED deviation: {desvio_base:.3f}")

# =========================
# Interactive loop
# =========================
variaveis = list(x_base.index)

while True:
    x_sim = x_base.copy()
    alteracoes = {}

    print("\nAvailable variables:")
    for v in variaveis:
        print(f" - {v}")

    while True:
        var = input("\nVariable to simulate (or 'end'): ").strip()
        if var.lower() == "end":
            break
        if var not in x_sim.index:
            print("❌ Invalid variable")
            continue

        atual = x_sim[var]
        novo = float(input("New value: "))
        x_sim[var] = novo
        alteracoes[var] = (atual, novo)

        if input("Change more variables? (y/n): ").lower() != "y":
            break

    if alteracoes:
        desvio_sim = model.predict(pd.DataFrame([x_sim]))[0]
        impacto = desvio_sim - desvio_base

        print("\n📊 RESULT")
        for v, (a, n) in alteracoes.items():
            print(f"{v}: {a} → {n}")
        print(f"Predicted deviation: {desvio_sim:.3f}")
        print(f"Impact vs base: {impacto:+.3f}")

    if input("\nNew simulation? (y/n): ").lower() != "y":
        break
