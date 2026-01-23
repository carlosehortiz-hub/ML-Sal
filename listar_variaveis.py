import pandas as pd

df = pd.read_csv("matriz_cloretos.csv", sep=None, engine="python")

print("Variáveis no ficheiro matriz_cloretos.csv:\n")
for col in df.columns:
    print(col)