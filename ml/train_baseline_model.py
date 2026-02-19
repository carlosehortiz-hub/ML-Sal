import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning

# Ensure project root is on sys.path when running from subdirectories.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.ml_utils import (
    load_data,
    build_features,
    filter_training_rows,
    NUM_COLS,
    TARGET_COL,
    MODEL_PATH,
    OUTPUTS_DIR,
    _save_model,
)

# =========================
# Read data
# =========================
df = load_data()
if df.empty:
    print("❌ No data available in the database.")
    raise SystemExit(1)


def _available_refs(df_in):
    if "referencia" not in df_in.columns:
        return []
    refs = []
    for value in df_in["referencia"].dropna().tolist():
        ref = str(value).strip()
        if ref and ref not in refs:
            refs.append(ref)
    refs.sort()
    return refs


def _reference_counts_for_training(df_in):
    if "referencia" not in df_in.columns:
        return {}
    df_count = df_in.copy()
    if "densidade" in df_count.columns:
        df_count = df_count[df_count["densidade"] != 0]
    refs_series = df_count["referencia"].dropna().astype(str).str.strip()
    refs_series = refs_series[refs_series != ""]
    return refs_series.value_counts().to_dict()


def _parse_refs_input(raw, refs):
    raw = (raw or "").strip()
    if raw == "":
        return None, "no references provided"

    for ref in refs:
        if ref.lower() == raw.lower():
            return [ref], None

    delimiter = ";" if ";" in raw else ","
    tokens = [t.strip() for t in raw.split(delimiter) if t.strip()]
    if not tokens:
        return None, "no references provided"

    chosen = []
    for token in tokens:
        selected = None
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(refs):
                selected = refs[idx - 1]
            else:
                return None, f"index out of range: {token}"
        else:
            for ref in refs:
                if ref.lower() == token.lower():
                    selected = ref
                    break
            if selected is None:
                return None, f"unknown reference: {token}"

        if selected not in chosen:
            chosen.append(selected)

    return chosen, None


selected_refs = []
refs = _available_refs(df)
ref_counts = _reference_counts_for_training(df)
total_count = int(sum(ref_counts.values()))
env_refs = os.environ.get("ML_SAL_REFERENCIAS", "").strip()

if env_refs and refs:
    if env_refs.lower() not in {"all", "*"}:
        parsed, err = _parse_refs_input(env_refs, refs)
        if err:
            print(f"❌ Invalid ML_SAL_REFERENCIAS: {err}")
            raise SystemExit(1)
        selected_refs = parsed
elif refs:
    print("\n🧪 Training scope by reference")
    print(f"1. All references (default, n={total_count})")
    print("2. One reference")
    print("3. Multiple references")
    while True:
        scope = input("Choice [1-3, default 1]: ").strip()
        if scope in {"", "1"}:
            break
        if scope == "2":
            print("\nAvailable references:")
            for idx, ref in enumerate(refs, start=1):
                print(f"{idx}. {ref} (n={ref_counts.get(ref, 0)})")
            raw = input("Reference (name or number): ").strip()
            parsed, err = _parse_refs_input(raw, refs)
            if err:
                print(f"❌ Invalid selection: {err}")
                continue
            selected_refs = [parsed[0]]
            break
        if scope == "3":
            print("\nAvailable references:")
            for idx, ref in enumerate(refs, start=1):
                print(f"{idx}. {ref} (n={ref_counts.get(ref, 0)})")
            raw = input(
                "References (numbers comma-separated, or names semicolon-separated): "
            ).strip()
            parsed, err = _parse_refs_input(raw, refs)
            if err:
                print(f"❌ Invalid selection: {err}")
                continue
            selected_refs = parsed
            break
        print("❌ Invalid option.")

if selected_refs:
    df = df[df["referencia"].astype(str).str.strip().isin(selected_refs)].reset_index(drop=True)
    print(f"🎯 Training scope: {len(selected_refs)} reference(s): {', '.join(selected_refs)}")
else:
    print("🎯 Training scope: all references.")

if df.empty:
    print("❌ No rows available after reference filter.")
    raise SystemExit(1)

# =========================
# Target
# =========================
X, y = build_features(df)
X, y, removed_rows = filter_training_rows(X, y)
if removed_rows > 0:
    print(f"⚠️ Ignoring {removed_rows} rows with densidade = 0 for training/evaluation.")

if len(X) < 2:
    print("❌ Not enough rows after filtering densidade = 0.")
    raise SystemExit(1)

# =========================
# Features (without reference)
# =========================
X_num = X[NUM_COLS]

# =========================
# Imputation
# =========================
imputer = SimpleImputer(strategy="median")
X_final = pd.DataFrame(
    imputer.fit_transform(X_num),
    columns=NUM_COLS
)

n_samples = len(X_final)
poly_min_samples = int(os.environ.get("ML_SAL_MIN_SAMPLES_POLY", "120"))
use_poly_models = n_samples >= poly_min_samples
if not use_poly_models:
    print(
        f"⚠️ Polynomial models disabled (n={n_samples} < {poly_min_samples}) "
        "to avoid numeric instability."
    )

# =========================
# Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Model
# =========================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test)

if len(y_test) >= 2:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        r2 = r2_score(y_test, y_pred)
else:
    r2 = float("nan")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
y_std = y_test.std()
rmse_rel = rmse / y_std if y_std != 0 else float("nan")

print("📈 Explainable model (without reference)")
print(f"R²   = {r2:.3f}")
print(f"RMSE = {rmse:.4f}")
print(f"RMSE/STD(y) = {rmse_rel:.3f}")


def evaluate_model(name, estimator, X_tr, y_tr, X_te, y_te, y_std_value):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        with np.errstate(all="ignore"):
            estimator.fit(X_tr, y_tr)
            preds = estimator.predict(X_te)

    if not np.isfinite(preds).all():
        return None

    if len(y_te) >= 2:
        r2_v = r2_score(y_te, preds)
    else:
        r2_v = float("nan")
    rmse_v = np.sqrt(mean_squared_error(y_te, preds))
    rmse_rel_v = rmse_v / y_std_value if y_std_value != 0 else float("nan")
    runtime_warnings = sum(
        1 for w in caught
        if issubclass(w.category, RuntimeWarning)
    )
    return (name, r2_v, rmse_v, rmse_rel_v, runtime_warnings)


models = [
    ("DummyMean", DummyRegressor(strategy="mean")),
    ("Linear", LinearRegression()),
    ("Ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    ("RandomForest", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )),
    ("ExtraTrees", ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )),
    ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
]
if use_poly_models:
    models.insert(3, ("Interactions_Ridge", make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
        StandardScaler(),
        Ridge(alpha=1.0),
    )))
    models.insert(4, ("Poly2_Ridge", make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1.0),
    )))

print("\n📊 Model comparison (same train/test split)")
split_metrics = []
for name, est in models:
    m = evaluate_model(name, est, X_train, y_train, X_test, y_test, y_std)
    if m is None:
        print(f"{name:<16} skipped (non-finite predictions)")
        continue
    warn_note = "  [numeric warnings]" if m[4] > 0 else ""
    print(f"{m[0]:<16} R²={m[1]:.3f}  RMSE={m[2]:.4f}  RMSE/STD={m[3]:.3f}{warn_note}")
    split_metrics.append({
        "name": m[0],
        "r2": m[1],
        "rmse": m[2],
        "rmse_rel": m[3],
    })

# =========================
# Cross-validation (more stable estimate)
# =========================
cv_metrics = []
if n_samples < 2:
    print("\n⚠️  Not enough samples for cross-validation.")
else:
    def _recommended_n_splits(n):
        if n < 30:
            return 2
        if n < 80:
            return 3
        if n < 150:
            return 4
        return 5

    n_splits = min(_recommended_n_splits(n_samples), n_samples)
    n_splits = max(2, n_splits)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    include_cv_r2 = n_samples >= 4
    scoring = {"mse": "neg_mean_squared_error"}
    if include_cv_r2:
        scoring["r2"] = "r2"
    else:
        print("⚠️ R² omitted in CV (too few samples per fold).")
    y_std_all = y.std()

    print(f"\n📈 Cross-validation (KFold, n_splits={n_splits})")
    for name, est in models:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            with np.errstate(all="ignore"):
                try:
                    cv_results = cross_validate(est, X_final, y, cv=cv, scoring=scoring)
                except Exception as exc:
                    print(f"{name:<16} skipped in CV ({exc})")
                    continue

        if include_cv_r2:
            r2_mean = cv_results["test_r2"].mean()
            r2_std = cv_results["test_r2"].std()
        else:
            r2_mean = float("nan")
            r2_std = float("nan")
        rmse_vals = np.sqrt(-cv_results["test_mse"])
        rmse_mean = rmse_vals.mean()
        rmse_std = rmse_vals.std()
        rmse_rel_mean = rmse_mean / y_std_all if y_std_all != 0 else float("nan")
        runtime_warnings = sum(
            1 for w in caught
            if issubclass(w.category, RuntimeWarning)
        )
        warn_note = "  [numeric warnings]" if runtime_warnings > 0 else ""
        print(
            f"{name:<16} R²={r2_mean:.3f}±{r2_std:.3f}  "
            f"RMSE={rmse_mean:.4f}±{rmse_std:.4f}  "
            f"RMSE/STD={rmse_rel_mean:.3f}{warn_note}"
        )
        cv_metrics.append({
            "name": name,
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "rmse_rel_mean": rmse_rel_mean,
        })

# =========================
# Top interaction coefficients (Ridge with interactions)
# =========================
if use_poly_models:
    print("\n🔎 Top interaction coefficients (Interactions_Ridge)")
    interactions_model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            interactions_model.fit(X_final, y)
    poly = interactions_model.named_steps["polynomialfeatures"]
    ridge = interactions_model.named_steps["ridge"]

    try:
        feature_names = poly.get_feature_names_out(NUM_COLS)
    except AttributeError:
        feature_names = poly.get_feature_names(NUM_COLS)

    coefs = ridge.coef_.ravel()
    interaction_terms = []
    for name, coef in zip(feature_names, coefs):
        if " " in name:
            interaction_terms.append((name.replace(" ", " * "), float(coef)))

    interaction_terms.sort(key=lambda x: abs(x[1]), reverse=True)
    if interaction_terms:
        for name, coef in interaction_terms[:10]:
            print(f"{name:<30} coef={coef:+.4f}")
    else:
        print("No interaction terms found.")
else:
    print("\n🔎 Top interaction coefficients (Interactions_Ridge)")
    print("Skipped (polynomial models disabled for small sample size).")

# =========================
# Scatter + correlation per feature
# =========================
print("\n🔎 Feature correlations with target (Pearson)")
df_corr = X_final.copy()
df_corr[TARGET_COL] = y.values
corr_series = df_corr.corr(numeric_only=True)[TARGET_COL].drop(TARGET_COL)
corr_series = corr_series.sort_values(ascending=False)
print(corr_series)

os.makedirs(OUTPUTS_DIR, exist_ok=True)
cols = 2
rows = int(np.ceil(len(NUM_COLS) / cols)) if len(NUM_COLS) > 0 else 1
fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
axes = np.array(axes).reshape(-1)

for i, feature in enumerate(NUM_COLS):
    ax = axes[i]
    ax.scatter(df_corr[feature], df_corr[TARGET_COL], s=14, alpha=0.6)
    r = corr_series.get(feature, float("nan"))
    ax.set_title(f"{feature} vs {TARGET_COL} (r={r:.3f})")
    ax.set_xlabel(feature)
    ax.set_ylabel(TARGET_COL)

# Hide unused axes
for j in range(len(NUM_COLS), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
scatter_path = os.path.join(OUTPUTS_DIR, "feature_scatter.png")
plt.savefig(scatter_path, dpi=150)
plt.close()
print(f"\n📄 Scatter plot saved to {scatter_path}")

# =========================
# Save selected model
# =========================
def _resolve_model_choice(raw, model_names):
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(model_names):
            return model_names[idx - 1]

    raw_lower = raw.lower()
    lower_map = {name.lower(): name for name in model_names}
    if raw_lower in lower_map:
        return lower_map[raw_lower]

    compact_map = {
        name.lower().replace("_", "").replace(" ", ""): name
        for name in model_names
    }
    raw_compact = raw_lower.replace("_", "").replace(" ", "")
    return compact_map.get(raw_compact)


def _pick_model_name(model_names, default_name):
    env_choice = os.environ.get("ML_SAL_MODEL")
    if env_choice:
        resolved = _resolve_model_choice(env_choice, model_names)
        if resolved:
            return resolved
        print(f"⚠️ Invalid ML_SAL_MODEL='{env_choice}'. Using {default_name}.")
        return default_name

    print("\n💾 Choose a model to save")
    for idx, name in enumerate(model_names, start=1):
        marker = " (default)" if name == default_name else ""
        print(f"{idx}. {name}{marker}")

    raw = input(
        f"Select model [1-{len(model_names)}] or name (default: {default_name}): "
    ).strip()

    if raw == "":
        return default_name

    resolved = _resolve_model_choice(raw, model_names)
    if resolved:
        return resolved

    print(f"⚠️ Invalid choice '{raw}'. Using {default_name}.")
    return default_name


model_dict = {name: est for name, est in models if name != "DummyMean"}
model_names = list(model_dict.keys())
if not split_metrics:
    print("\n❌ No stable model metrics available to choose/save.")
    raise SystemExit(1)
split_best = min(
    [m for m in split_metrics if m["name"] in model_dict],
    key=lambda m: m["rmse"],
)
cv_best = None
cv_candidates = [m for m in cv_metrics if m["name"] in model_dict]
if cv_candidates:
    cv_best = min(
        cv_candidates,
        key=lambda m: m["rmse_mean"],
    )

recommended_model = cv_best["name"] if cv_best is not None else split_best["name"]
print("\n🏆 Most precise model")
print(
    f"Holdout best: {split_best['name']} "
    f"(RMSE={split_best['rmse']:.4f}, R²={split_best['r2']:.3f})"
)
if cv_best is not None:
    print(
        f"CV best     : {cv_best['name']} "
        f"(RMSE={cv_best['rmse_mean']:.4f}, R²={cv_best['r2_mean']:.3f})"
    )
print(f"Recommended to save: {recommended_model}")

default_model = recommended_model if recommended_model in model_dict else model_names[0]
selected_name = _pick_model_name(model_names, default_model)
selected_estimator = clone(model_dict[selected_name])
selected_estimator.fit(X_final, y)
_save_model(
    selected_estimator,
    imputer,
    MODEL_PATH,
    model_name=selected_name,
    target_col=TARGET_COL,
)
print(f"\n✅ Saved model '{selected_name}' to {MODEL_PATH}")
