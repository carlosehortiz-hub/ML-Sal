import pandas as pd

df = pd.read_csv("matriz_cloretos.csv", sep=None, engine="python")

print("Variables in matriz_cloretos.csv:\n")
for col in df.columns:
    print(col)
