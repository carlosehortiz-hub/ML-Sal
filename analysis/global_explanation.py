import os
import sys
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is on sys.path when running from subdirectories.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.ml_utils import load_data, build_features, get_model, transform_features, OUTPUTS_DIR


def _sample_rows(df_like, max_rows, random_state=42):
    if hasattr(df_like, "sample"):
        n = min(max_rows, len(df_like))
        return df_like.sample(n=n, random_state=random_state)
    return df_like[:max_rows]


def compute_shap_values(model, background, X_eval, nsamples=200):
    feature_cols = list(background.columns) if hasattr(background, "columns") else None

    def predict_fn(data):
        data_in = data
        if feature_cols is not None and not hasattr(data_in, "columns"):
            data_in = pd.DataFrame(data_in, columns=feature_cols)
        return model.predict(data_in)

    try:
        explainer = shap.Explainer(predict_fn, background)
        shap_values = explainer(X_eval)
        values = getattr(shap_values, "values", shap_values)
        return np.asarray(values), "explainer"
    except Exception:
        bg_small = _sample_rows(background, 80)
        kernel = shap.KernelExplainer(predict_fn, bg_small)
        values = kernel.shap_values(X_eval, nsamples=nsamples)
        if isinstance(values, list):
            values = values[0]
        return np.asarray(values), "kernel"

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
shap_values, shap_mode = compute_shap_values(
    model,
    background,
    sample,
)
if shap_mode != "explainer":
    print("ℹ️ Using SHAP compatibility mode (KernelExplainer).")

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
