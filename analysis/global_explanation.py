import os
import sys
from pathlib import Path
import shap
import matplotlib.pyplot as plt

# Ensure project root is on sys.path when running from subdirectories.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.ml_utils import load_data, build_features, get_model, transform_features, OUTPUTS_DIR

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

background = X_num.sample(
    n=min(200, len(X_num)),
    random_state=42,
)
explainer = shap.Explainer(model, background)
shap_values = explainer(sample)

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    sample,
    plot_type="bar",
    show=False,
)

plt.tight_layout()
os.makedirs(OUTPUTS_DIR, exist_ok=True)
shap_global_path = os.path.join(OUTPUTS_DIR, "shap_global.png")
plt.savefig(shap_global_path, dpi=150)
plt.close()

print(f"✅ Global SHAP (without reference) saved to {shap_global_path}")
