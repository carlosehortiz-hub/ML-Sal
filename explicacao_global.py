import pandas as pd
import shap
import matplotlib.pyplot as plt

from ml_utils import load_data, build_features, get_model, transform_features

# =========================
# Read data
# =========================
df = load_data()

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
# SHAP GLOBAL
# =========================
sample = X_num.sample(
    n=min(300, len(X_num)),
    random_state=42,
)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    sample,
    plot_type="bar",
    show=False,
)

plt.tight_layout()
plt.savefig("shap_global.png", dpi=150)
plt.close()

print("✅ Global SHAP (without reference) saved to shap_global.png")
