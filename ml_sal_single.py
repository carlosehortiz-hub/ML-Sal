"""
Single-file bundle for ML-Sal.

Usage:
  - CLI menu:    python ml_sal_single.py
  - Streamlit:   streamlit run ml_sal_single.py -- --streamlit

Optional environment variables:
  - ML_SAL_DB_PATH
  - ML_SAL_CSV_PATH
  - ML_SAL_OUTPUTS_DIR
  - ML_SAL_MODELS_DIR
  - ML_SAL_MODEL (model name for non-interactive training)
  - ML_SAL_AUTO_INSTALL (default 1; set to 0 to disable auto-install)
  - ML_SAL_PIP_ARGS (extra arguments passed to pip)
"""

import os
import sys
import sqlite3
import subprocess
import importlib
from datetime import datetime

pd = None
np = None
joblib = None
SimpleImputer = None
RandomForestRegressor = None
train_test_split = None
r2_score = None
mean_squared_error = None


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_DB_PATH = os.environ.get("ML_SAL_DB_PATH", os.path.join(BASE_DIR, "ml_sal.db"))
DEFAULT_CSV_PATH = os.environ.get("ML_SAL_CSV_PATH", os.path.join(BASE_DIR, "matriz_cloretos.csv"))
OUTPUTS_DIR = os.environ.get("ML_SAL_OUTPUTS_DIR", os.path.join(BASE_DIR, "outputs"))
MODELS_DIR = os.environ.get("ML_SAL_MODELS_DIR", os.path.join(BASE_DIR, "models"))
MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")

TARGET_COL = "dif_pct_sal"
DEVIATION_COL = "dif_pct_sal"

NUM_COLS = [
    "dif_es",
    "dif_gs",
    "ph_entrada",
    "densidade",
    "min_fora",
]

DROP_COLS = [
    "id",
    "data",
    "lote",
    "pct_sal",
    "cuba",
    "referencia",
    "dif_pct_sal",
]


def _resolve_path(path, default):
    return default if path is None else path


_CORE_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("joblib", "joblib"),
    ("sklearn", "scikit-learn"),
]

_CHECKED_IMPORTS = set()
_CORE_READY = False


def _auto_install_enabled():
    value = os.environ.get("ML_SAL_AUTO_INSTALL", "1").strip().lower()
    return value not in {"0", "false", "no"}


def _pip_install(packages):
    if not packages:
        return
    if not _auto_install_enabled():
        missing = ", ".join(packages)
        raise ImportError(
            f"Missing packages: {missing}. Auto-install disabled. "
            "Set ML_SAL_AUTO_INSTALL=1 to allow install."
        )

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    extra = os.environ.get("ML_SAL_PIP_ARGS")
    if extra:
        cmd += extra.split()

    print(f"Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        raise ImportError(
            "Failed to install required packages. "
            "Check your network connection and pip configuration."
        ) from exc


def ensure_packages(packages):
    missing = []
    for import_name, pip_name in packages:
        if import_name in _CHECKED_IMPORTS:
            continue
        try:
            importlib.import_module(import_name)
            _CHECKED_IMPORTS.add(import_name)
        except Exception:
            missing.append((import_name, pip_name))

    if missing:
        _pip_install([pip_name for _, pip_name in missing])
        for import_name, _ in missing:
            importlib.import_module(import_name)
            _CHECKED_IMPORTS.add(import_name)


def _load_core_packages():
    global pd, np, joblib
    global SimpleImputer, RandomForestRegressor
    global train_test_split, r2_score, mean_squared_error

    import numpy as _np
    import pandas as _pd
    import joblib as _joblib
    from sklearn.impute import SimpleImputer as _SimpleImputer
    from sklearn.ensemble import RandomForestRegressor as _RandomForestRegressor
    from sklearn.model_selection import train_test_split as _train_test_split
    from sklearn.metrics import r2_score as _r2_score, mean_squared_error as _mean_squared_error

    np = _np
    pd = _pd
    joblib = _joblib
    SimpleImputer = _SimpleImputer
    RandomForestRegressor = _RandomForestRegressor
    train_test_split = _train_test_split
    r2_score = _r2_score
    mean_squared_error = _mean_squared_error


def ensure_core_dependencies():
    global _CORE_READY
    if _CORE_READY:
        return
    ensure_packages(_CORE_PACKAGES)
    _load_core_packages()
    _CORE_READY = True


def import_optional(module_name, pip_name, check_import=None):
    ensure_packages([(check_import or module_name, pip_name)])
    return importlib.import_module(module_name)


# =========================
# ML utilities (from ml/ml_utils.py)
# =========================

def load_data(db_path=None):
    ensure_core_dependencies()
    db_path = _resolve_path(db_path, DEFAULT_DB_PATH)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM desvios_sal", conn)
    conn.close()

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return df


def build_features(df):
    y = df[TARGET_COL]
    X = df.drop(columns=DROP_COLS)
    return X, y


def filter_training_rows(X, y):
    if "densidade" not in X.columns:
        return X.copy(), y.copy(), 0

    mask = X["densidade"] != 0
    removed = int((~mask).sum())
    return X.loc[mask].copy(), y.loc[mask].copy(), removed


def fit_model(X, y, n_estimators=300, random_state=42):
    ensure_core_dependencies()
    X_train, y_train, _ = filter_training_rows(X, y)
    if X_train.empty:
        raise ValueError("No rows left after filtering densidade = 0.")

    imputer = SimpleImputer(strategy="median")
    X_num = pd.DataFrame(
        imputer.fit_transform(X_train[NUM_COLS]),
        columns=NUM_COLS,
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_num, y_train)
    return model, imputer, X_num


def _load_model(path):
    ensure_core_dependencies()
    if not os.path.exists(path):
        return None
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        return None
    if "model" not in payload or "imputer" not in payload:
        return None
    if "target_col" not in payload or payload.get("target_col") != TARGET_COL:
        return None
    return payload


def _save_model(model, imputer, path, model_name=None, target_col=None):
    ensure_core_dependencies()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "num_cols": NUM_COLS,
            "model_name": model_name,
            "target_col": target_col or TARGET_COL,
        },
        path,
    )


def _get_auto_train_model_candidates():
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Linear": LinearRegression(),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }


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


def _pick_auto_train_model(model_names, default_name):
    env_choice = os.environ.get("ML_SAL_MODEL")
    if env_choice:
        resolved = _resolve_model_choice(env_choice, model_names)
        if resolved:
            return resolved
        print(f"Invalid ML_SAL_MODEL='{env_choice}'. Using {default_name}.")
        return default_name

    can_prompt = sys.stdin is not None and sys.stdin.isatty() and "--streamlit" not in sys.argv
    if not can_prompt:
        return default_name

    print("\nNo trained model found. Choose a model to train now:")
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

    print(f"Invalid choice '{raw}'. Using {default_name}.")
    return default_name


def get_model(X, y, model_path=MODEL_PATH):
    ensure_core_dependencies()
    retrain = os.environ.get("RETRAIN_MODEL") == "1"
    payload = None if retrain else _load_model(model_path)
    if payload is not None and payload.get("num_cols") == NUM_COLS:
        return payload["model"], payload["imputer"]

    X_train, y_train, removed_rows = filter_training_rows(X, y)
    if removed_rows > 0:
        print(f"Ignoring {removed_rows} rows with densidade = 0 for training.")
    if X_train.empty:
        raise ValueError("No rows left after filtering densidade = 0.")

    imputer = SimpleImputer(strategy="median")
    X_num = pd.DataFrame(
        imputer.fit_transform(X_train[NUM_COLS]),
        columns=NUM_COLS,
    )

    model_dict = _get_auto_train_model_candidates()
    model_names = list(model_dict.keys())
    default_model = "RandomForest" if "RandomForest" in model_dict else model_names[0]
    selected_name = _pick_auto_train_model(model_names, default_model)

    model = model_dict[selected_name]
    model.fit(X_num, y_train)
    _save_model(model, imputer, model_path, model_name=selected_name)
    print(f"Auto-trained and saved model '{selected_name}' to {model_path}")
    return model, imputer


def transform_features(imputer, X):
    ensure_core_dependencies()
    return pd.DataFrame(
        imputer.transform(X[NUM_COLS]),
        columns=NUM_COLS,
    )


def data_quality_report(df):
    print("\nDATA QUALITY REPORT")
    print("=" * 70)

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("OK: no missing values detected.")
    else:
        print("Missing values by column:")
        print(missing)

    print("\nRule checks")
    rules = {
        "ph_entrada": (0, 1),
        "tempo_fora_espec": (0, 1),
        "ph_salga": (3.5, 7.5),
        "temperatura": (-5, 40),
        "densidade": (0, None),
        "min_fora": (0, None),
    }

    for col, (min_v, max_v) in rules.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        below = (series < min_v).sum() if min_v is not None else 0
        above = (series > max_v).sum() if max_v is not None else 0

        if below == 0 and above == 0:
            print(f"OK: {col} within expected range")
        else:
            print(f"WARNING: {col} {below} below, {above} above")

    print("\nOutliers (IQR method)")
    numeric_cols = [c for c in NUM_COLS + [TARGET_COL] if c in df.columns]
    outlier_rows = 0

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        outlier_rows += int(outliers > 0)
        print(f"- {col}: {outliers} outliers")

    if outlier_rows == 0:
        print("OK: no IQR outliers detected.")


def _sample_rows(df_like, max_rows, random_state=42):
    if hasattr(df_like, "sample"):
        n = min(max_rows, len(df_like))
        return df_like.sample(n=n, random_state=random_state)
    return df_like[:max_rows]


def compute_shap_values(shap_module, model, background, X_eval, nsamples=200):
    feature_cols = list(background.columns) if hasattr(background, "columns") else None

    def predict_fn(data):
        data_in = data
        if feature_cols is not None and not hasattr(data_in, "columns"):
            data_in = pd.DataFrame(data_in, columns=feature_cols)
        return model.predict(data_in)

    try:
        explainer = shap_module.Explainer(predict_fn, background)
        shap_values = explainer(X_eval)
        values = getattr(shap_values, "values", shap_values)
        return np.asarray(values), "explainer"
    except Exception as explainer_error:
        bg_small = _sample_rows(background, 80)
        try:
            kernel = shap_module.KernelExplainer(predict_fn, bg_small)
            values = kernel.shap_values(X_eval, nsamples=nsamples)
            if isinstance(values, list):
                values = values[0]
            return np.asarray(values), "kernel"
        except Exception as kernel_error:
            raise RuntimeError(
                "SHAP failed in both modes. "
                f"Explainer error: {explainer_error} | "
                f"KernelExplainer error: {kernel_error}"
            )


# =========================
# Database and utilities
# =========================

def import_history(csv_path=None, db_path=None):
    ensure_core_dependencies()
    csv_path = _resolve_path(csv_path, DEFAULT_CSV_PATH)
    db_path = _resolve_path(db_path, DEFAULT_DB_PATH)

    df = pd.read_csv(csv_path, sep=None, engine="python")

    df.columns = (
        df.columns
        .str.replace("%", "pct", regex=False)
        .str.replace(" ", "_")
        .str.replace("__", "_")
        .str.lower()
    )

    df = df.rename(columns={"\ufeffdata": "data"})
    df = df.replace(["#VALUE!", "#DIV/0!", "#N/A", "N/A", ""], pd.NA)

    df["data"] = pd.to_datetime(
        df["data"],
        dayfirst=True,
        errors="coerce",
    )

    percent_cols = ["pct_sal", "dif_pct_sal", "dif_es", "dif_hfd", "dif_gs"]
    for col in percent_cols:
        if col not in df.columns:
            continue
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "ph_entrada_salga" in df.columns:
        df["ph_entrada_salga"] = (
            df["ph_entrada_salga"]
            .astype(str)
            .str.upper()
            .map({"OK": 0, "NOK": 1})
        )

    num_cols = [
        "ph_salga",
        "densidade_salga",
        "temperatura_salga",
        "min_fora",
        "tempo_fora_espec",
    ]
    for col in num_cols:
        if col not in df.columns:
            continue
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS desvios_sal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT,
            referencia TEXT,
            cuba INTEGER,
            lote TEXT,
            pct_sal REAL,
            dif_pct_sal REAL,
            dif_es REAL,
            dif_hfd REAL,
            dif_gs REAL,
            ph_entrada INTEGER,
            ph_salga REAL,
            densidade REAL,
            temperatura REAL,
            min_fora REAL,
            tempo_fora_espec INTEGER
        )
        """
    )
    conn.commit()

    df_final = df.rename(columns={
        "ph_entrada_salga": "ph_entrada",
        "densidade_salga": "densidade",
        "temperatura_salga": "temperatura",
    })

    df_final.to_sql(
        "desvios_sal",
        conn,
        if_exists="append",
        index=False,
    )
    conn.close()

    print("History imported successfully.")
    print(f"Total rows imported: {len(df_final)}")
    print("\nMissing values by column:")
    print(df_final.isna().sum())


def verify_database(db_path=None):
    ensure_core_dependencies()
    db_path = _resolve_path(db_path, DEFAULT_DB_PATH)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM desvios_sal", conn)
    conn.close()

    print("Dataset size:")
    print(df.shape)
    print("\nFirst rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print(f"\nTarget statistics ({TARGET_COL}):")
    if TARGET_COL in df.columns:
        print(df[TARGET_COL].describe())
    else:
        print(f"Column not found: {TARGET_COL}")
    print("\nMissing values:")
    print(df.isna().sum())


def verify_inserts(db_path=None):
    ensure_core_dependencies()
    db_path = _resolve_path(db_path, DEFAULT_DB_PATH)
    conn = sqlite3.connect(db_path)
    query = """
    SELECT *
    FROM desvios_sal
    ORDER BY id DESC
    LIMIT 5
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print("The table is empty. No records found.")
    else:
        print("\nLatest records inserted into the database:")
        print("=" * 60)
        print(df)


def delete_last_record(db_path=None):
    db_path = _resolve_path(db_path, DEFAULT_DB_PATH)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, data, referencia, lote, dif_pct_sal
        FROM desvios_sal
        ORDER BY id DESC
        LIMIT 1
        """
    )
    registo = cursor.fetchone()

    if registo is None:
        print("There are no records to delete.")
        conn.close()
        return

    print("\nLAST INSERTED RECORD:")
    print(f"ID        : {registo[0]}")
    print(f"Date      : {registo[1]}")
    print(f"Reference : {registo[2]}")
    print(f"Batch     : {registo[3]}")
    print(f"Deviation : {registo[4]}")

    confirmar = input("\nConfirm deletion of this record? (y/n): ").strip().lower()
    if confirmar == "y":
        cursor.execute(
            "DELETE FROM desvios_sal WHERE id = ?",
            (registo[0],),
        )
        conn.commit()
        print("\nRecord deleted successfully.")
    else:
        print("\nOperation canceled.")

    conn.close()


def create_table(db_path=None):
    db_path = _resolve_path(db_path, DEFAULT_DB_PATH)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS desvios_sal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT,
            referencia TEXT,
            cuba INTEGER,
            lote TEXT,
            dif_pct_sal REAL,
            pct_sal REAL,
            ph_entrada REAL,
            ph_salga REAL,
            densidade REAL,
            temperatura REAL,
            observacoes TEXT
        );
        """
    )
    conn.commit()
    conn.close()
    print("Table 'desvios_sal' created successfully.")


def list_variables(csv_path=None):
    ensure_core_dependencies()
    csv_path = _resolve_path(csv_path, DEFAULT_CSV_PATH)
    df = pd.read_csv(csv_path, sep=None, engine="python")
    print("Variables in matriz_cloretos.csv:\n")
    for col in df.columns:
        print(col)


def pearson_heatmap_analysis(csv_path=None, out_path=None, db_path=None, source=None):
    ensure_core_dependencies()
    csv_path = _resolve_path(csv_path, DEFAULT_CSV_PATH)
    db_path = _resolve_path(db_path, DEFAULT_DB_PATH)
    out_path = _resolve_path(out_path, os.path.join(OUTPUTS_DIR, "pearson_heatmap.png"))

    source_norm = (source or "").strip().lower()
    if source_norm not in {"db", "csv"}:
        default_source = "db" if os.path.exists(db_path) else "csv"
        raw = input(
            f"Heatmap source [db/csv] (default: {default_source}): "
        ).strip().lower()
        source_norm = raw if raw in {"db", "csv"} else default_source

    if source_norm == "db":
        if not os.path.exists(db_path):
            print(f"DB not found: {db_path}")
            return
        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql("SELECT * FROM desvios_sal", conn)
        except Exception as exc:
            print(f"Could not read table 'desvios_sal' from DB: {exc}")
            conn.close()
            return
        conn.close()
    else:
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path}")
            return

        df = pd.read_csv(csv_path, sep=None, engine="python")
        df.columns = (
            df.columns
            .str.replace("%", "pct", regex=False)
            .str.replace(" ", "_")
            .str.replace("__", "_")
            .str.lower()
        )
        df = df.rename(columns={"\ufeffdata": "data"})
        df = df.replace(["#VALUE!", "#DIV/0!", "#N/A", "N/A", ""], pd.NA)

        if "data" in df.columns:
            df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")

        percent_cols = ["pct_sal", "dif_pct_sal", "dif_es", "dif_hfd", "dif_gs"]
        for col in percent_cols:
            if col not in df.columns:
                continue
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "ph_entrada_salga" in df.columns:
            df["ph_entrada_salga"] = (
                df["ph_entrada_salga"]
                .astype(str)
                .str.upper()
                .map({"OK": 0, "NOK": 1})
            )

        num_cols = [
            "ph_salga",
            "densidade_salga",
            "temperatura_salga",
            "min_fora",
            "tempo_fora_espec",
        ]
        for col in num_cols:
            if col not in df.columns:
                continue
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.rename(columns={
            "ph_entrada_salga": "ph_entrada",
            "densidade_salga": "densidade",
            "temperatura_salga": "temperatura",
        })

    df_num = df.select_dtypes(include="number")
    if "id" in df_num.columns:
        df_num = df_num.drop(columns=["id"])
    if df_num.empty:
        print("No numeric columns found for correlation.")
        return

    corr = df_num.corr(method="pearson")
    print("\nNumeric columns used in correlation:")
    print(", ".join(df_num.columns))

    target_default = TARGET_COL if TARGET_COL in corr.columns else "dif_pct_sal"
    target_col = input(
        f"\nTarget column for exclusions [default: {target_default}]: "
    ).strip()
    if target_col == "":
        target_col = target_default
    if target_col not in corr.columns:
        print(f"Target '{target_col}' not found. Continuing without target-based ranking.")

    def read_float(prompt, default):
        raw = input(f"{prompt} [default: {default}]: ").strip()
        if raw == "":
            return float(default)
        try:
            return float(raw.replace(",", "."))
        except Exception:
            print(f"Invalid value '{raw}'. Using default {default}.")
            return float(default)

    def read_int(prompt, default):
        raw = input(f"{prompt} [default: {default}]: ").strip()
        if raw == "":
            return int(default)
        try:
            return int(raw)
        except Exception:
            print(f"Invalid value '{raw}'. Using default {default}.")
            return int(default)

    min_target = read_float("Min abs corr with target to keep", 0.05)
    max_pair = read_float("Pairwise collinearity threshold", 0.90)
    top_pairs = read_int("Top correlated pairs to print", 20)

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = []
    for row in upper.index:
        for col in upper.columns:
            val = upper.loc[row, col]
            if pd.notna(val) and abs(val) >= max_pair:
                pairs.append((row, col, float(val)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    target_corr = None
    low_target = []
    if target_col in corr.columns:
        target_corr = corr[target_col].drop(labels=[target_col]).abs().sort_values()
        low_target = target_corr[target_corr < min_target].index.tolist()

    collinear_drop = []
    for a, b, _ in pairs:
        if target_corr is not None:
            a_score = abs(corr.loc[a, target_col]) if a in corr.index else 0
            b_score = abs(corr.loc[b, target_col]) if b in corr.index else 0
            drop = a if a_score < b_score else b
        else:
            drop = b
        collinear_drop.append(drop)

    suggested = sorted(set(low_target + collinear_drop))

    if target_corr is not None:
        print(f"\nCorrelation with target '{target_col}' (abs sorted):")
        print(target_corr.sort_values(ascending=False))
    else:
        print(f"\nTarget '{target_col}' not found in numeric columns.")

    print(f"\nTop correlated pairs (abs >= {max_pair}):")
    if pairs:
        for a, b, v in pairs[:top_pairs]:
            print(f"{a} <-> {b}: {v:+.3f}")
    else:
        print("No pairs above threshold.")

    print(f"\nLow correlation with target (abs < {min_target}):")
    print(low_target if low_target else "None")

    print(f"\nSuggested drops due to collinearity (>= {max_pair}):")
    print(sorted(set(collinear_drop)) if collinear_drop else "None")

    manual = input("\nManual exclusions (comma-separated, optional): ").strip()
    manual_exclude = [c.strip() for c in manual.split(",") if c.strip()]
    final_exclude = sorted(set(suggested + manual_exclude))

    print("\nSuggested exclusions (combined):")
    print(final_exclude if final_exclude else "None")

    write_filtered = input(
        "Write filtered CSV? Enter path or press Enter to skip: "
    ).strip()
    if write_filtered:
        keep_cols = [c for c in df.columns if c not in final_exclude]
        df_filtered = df[keep_cols].copy()
        out_dir = os.path.dirname(write_filtered)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_filtered.to_csv(write_filtered, index=False)
        print(f"Filtered CSV written to: {write_filtered}")

    try:
        sns = import_optional("seaborn", "seaborn")
        plt = import_optional("matplotlib.pyplot", "matplotlib", check_import="matplotlib")
    except Exception as exc:
        print(str(exc))
        print("Install seaborn/matplotlib to generate the heatmap image.")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPearson heatmap saved to: {out_path}")


# =========================
# Manual insert helpers (from utilities/insert_deviation_manual.py)
# =========================

def input_float(mensagem, exemplo=None, minimo=None, maximo=None):
    while True:
        texto = mensagem
        if exemplo:
            texto += f" (e.g.: {exemplo})"
        texto += " -> use dot or comma: "

        valor_str = input(texto)
        if valor_str is None:
            print("\nEmpty input.\n")
            continue

        valor_str = valor_str.strip()
        if valor_str == "":
            print("\nValue cannot be empty.\n")
            continue

        valor_str = valor_str.replace(",", ".")
        try:
            valor = float(valor_str)
        except Exception:
            print(
                "\nInvalid value."
                "\nEnter numbers only."
                "\nValid examples: 1.60 | 1,60 | -0.045\n"
            )
            continue

        if minimo is not None and valor < minimo:
            print(f"\nInvalid value. Must be >= {minimo}.\n")
            continue

        if maximo is not None and valor > maximo:
            print(f"\nInvalid value. Must be <= {maximo}.\n")
            continue

        return valor


def input_int(mensagem, exemplo=None, minimo=None, maximo=None):
    while True:
        texto = mensagem
        if exemplo:
            texto += f" (e.g.: {exemplo})"
        texto += ": "

        valor_str = input(texto)
        if valor_str is None:
            print("\nEmpty input.\n")
            continue

        valor_str = valor_str.strip()
        if valor_str == "":
            print("\nValue cannot be empty.\n")
            continue

        try:
            valor = int(valor_str)
        except Exception:
            print("\nInvalid value. Enter an integer only.\n")
            continue

        if minimo is not None and valor < minimo:
            print(f"\nInvalid value. Must be >= {minimo}.\n")
            continue

        if maximo is not None and valor > maximo:
            print(f"\nInvalid value. Must be <= {maximo}.\n")
            continue

        return valor


def input_data():
    while True:
        data = input("Date (YYYY-MM-DD) [Enter = today]: ")
        if data is None:
            print("\nInvalid input.\n")
            continue

        data = data.strip()
        if data == "":
            return datetime.today().strftime("%Y-%m-%d")

        try:
            datetime.strptime(data, "%Y-%m-%d")
            return data
        except Exception:
            print("\nInvalid date. Use YYYY-MM-DD format.\n")


def input_choice(mensagem, opcoes, descricao=None):
    while True:
        if descricao:
            print(descricao)
        valor = input(f"{mensagem} {opcoes}: ")
        if valor is None:
            print("\nInvalid input.\n")
            continue

        valor = valor.strip()
        if valor in opcoes:
            return int(valor)

        print(f"\nInvalid value. Choose one of the options {opcoes}.\n")


def input_referencia(cursor):
    cursor.execute("SELECT DISTINCT referencia FROM desvios_sal ORDER BY referencia")
    refs = [r[0] for r in cursor.fetchall()]

    if not refs:
        print("\nThere are no references in the database.")
        sys.exit(1)

    print("\nValid references:")
    for r in refs:
        print(f" - {r}")

    while True:
        ref = input("\nReference (copy exactly from the list): ")
        if ref is None:
            print("\nInvalid input.\n")
            continue

        ref = ref.strip()
        if ref == "":
            print("\nReference cannot be empty.\n")
            continue

        if ref in refs:
            return ref

        print(
            "\nInvalid reference."
            "\nIt must match exactly one of the listed references."
            "\nWatch out for spaces, uppercase letters, and accents.\n"
        )


def insert_deviation_manual(db_path=None):
    db_path = _resolve_path(db_path, DEFAULT_DB_PATH)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\nMANUAL INSERT OF NEW DEVIATION")
    print("=" * 60)

    data = input_data()
    referencia = input_referencia(cursor)

    cuba = input_int("Vat (integer)", exemplo="12", minimo=1)

    lote = ""
    while lote == "":
        lote = input("Batch (free text): ")
        if lote is None:
            lote = ""
        lote = lote.strip()
        if lote == "":
            print("\nBatch cannot be empty.\n")

    pct_sal = input_float("Measured salt %", exemplo="1.60", minimo=0)
    dif_pct_sal = input_float("Salt % difference", exemplo="-0.040")

    dif_es = input_float("Dif_ES", exemplo="0.90")
    dif_hfd = input_float("Dif_HFD", exemplo="-1.10")
    dif_gs = input_float("Dif_GS", exemplo="0.20")

    ph_entrada = input_choice(
        "Input pH",
        ["0", "1"],
        descricao="0 = OK | 1 = NOK",
    )

    ph_salga = input_float("Brine pH", exemplo="5.05", minimo=3.5, maximo=7.5)
    densidade = input_float("Density", exemplo="18.8", minimo=0)
    temperatura = input_float("Temperature (C)", exemplo="10.7", minimo=-5, maximo=40)
    min_fora = input_float("Minutes out of spec", exemplo="25")
    tempo_fora_espec = input_choice(
        "Time out of spec",
        ["0", "1"],
        descricao="0 = In | 1 = Out",
    )

    print("\nDATA SUMMARY")
    print("-" * 60)
    print(f"Date               : {data}")
    print(f"Reference          : {referencia}")
    print(f"Vat                : {cuba}")
    print(f"Batch              : {lote}")
    print(f"Salt %             : {pct_sal}")
    print(f"Salt % diff         : {dif_pct_sal}")
    print(f"Dif_ES             : {dif_es}")
    print(f"Dif_HFD            : {dif_hfd}")
    print(f"Dif_GS             : {dif_gs}")
    print(f"Input pH (0/1)     : {ph_entrada}")
    print(f"Brine pH           : {ph_salga}")
    print(f"Density            : {densidade}")
    print(f"Temperature        : {temperatura}")
    print(f"Minutes out        : {min_fora}")
    print(f"Time out spec      : {tempo_fora_espec}")

    confirmar = input("\nConfirm insertion? (y/n): ")
    if confirmar is None or confirmar.strip().lower() != "y":
        print("\nInsertion canceled.")
        conn.close()
        return

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
            min_fora, tempo_fora_espec,
        ),
    )
    conn.commit()
    conn.close()
    print("\nNew deviation inserted successfully.")


# =========================
# ML scripts (from ml/*.py)
# =========================

def prepare_ml_data(db_path=None):
    ensure_core_dependencies()
    df = load_data(db_path)
    X, y = build_features(df)
    X, y, removed_rows = filter_training_rows(X, y)
    if removed_rows > 0:
        print(f"Ignoring {removed_rows} rows with densidade = 0.")

    X_num = X[NUM_COLS]
    num_imputer = SimpleImputer(strategy="median")
    X_final = pd.DataFrame(
        num_imputer.fit_transform(X_num),
        columns=NUM_COLS,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("Data prepared for ML (without reference)")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)


def train_baseline_model(db_path=None):
    ensure_core_dependencies()
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.base import clone

    df = load_data(db_path)
    X, y = build_features(df)
    X, y, removed_rows = filter_training_rows(X, y)
    if removed_rows > 0:
        print(f"Ignoring {removed_rows} rows with densidade = 0 for training/evaluation.")

    if len(X) < 2:
        print("Not enough rows after filtering densidade = 0.")
        return

    X_num = X[NUM_COLS]
    imputer = SimpleImputer(strategy="median")
    X_final = pd.DataFrame(
        imputer.fit_transform(X_num),
        columns=NUM_COLS,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    y_std = y_test.std()
    rmse_rel = rmse / y_std if y_std != 0 else float("nan")

    print("Explainable model (without reference)")
    print(f"R2   = {r2:.3f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"RMSE/STD(y) = {rmse_rel:.3f}")

    def evaluate_model(name, estimator, X_tr, y_tr, X_te, y_te, y_std_value):
        estimator.fit(X_tr, y_tr)
        preds = estimator.predict(X_te)
        r2_v = r2_score(y_te, preds)
        rmse_v = np.sqrt(mean_squared_error(y_te, preds))
        rmse_rel_v = rmse_v / y_std_value if y_std_value != 0 else float("nan")
        return (name, r2_v, rmse_v, rmse_rel_v)

    models = [
        ("DummyMean", DummyRegressor(strategy="mean")),
        ("Linear", LinearRegression()),
        ("Ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("Interactions_Ridge", make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
            StandardScaler(),
            Ridge(alpha=1.0),
        )),
        ("Poly2_Ridge", make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0),
        )),
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

    print("\nModel comparison (same train/test split)")
    split_metrics = []
    for name, est in models:
        m = evaluate_model(name, est, X_train, y_train, X_test, y_test, y_std)
        print(f"{m[0]:<16} R2={m[1]:.3f}  RMSE={m[2]:.4f}  RMSE/STD={m[3]:.3f}")
        split_metrics.append({
            "name": m[0],
            "r2": m[1],
            "rmse": m[2],
            "rmse_rel": m[3],
        })

    # Cross-validation (more stable estimate)
    n_samples = len(X_final)
    if n_samples < 2:
        print("\nNot enough samples for cross-validation.")
        return

    n_splits = 5 if n_samples >= 5 else 2
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {"r2": "r2", "mse": "neg_mean_squared_error"}
    y_std_all = y.std()
    cv_metrics = []

    print(f"\nCross-validation (KFold, n_splits={n_splits})")
    for name, est in models:
        cv_results = cross_validate(est, X_final, y, cv=cv, scoring=scoring)
        r2_mean = cv_results["test_r2"].mean()
        r2_std = cv_results["test_r2"].std()
        rmse_vals = np.sqrt(-cv_results["test_mse"])
        rmse_mean = rmse_vals.mean()
        rmse_std = rmse_vals.std()
        rmse_rel_mean = rmse_mean / y_std_all if y_std_all != 0 else float("nan")
        print(
            f"{name:<16} R2={r2_mean:.3f}±{r2_std:.3f}  "
            f"RMSE={rmse_mean:.4f}±{rmse_std:.4f}  "
            f"RMSE/STD={rmse_rel_mean:.3f}"
        )
        cv_metrics.append({
            "name": name,
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "rmse_rel_mean": rmse_rel_mean,
        })

    # Top interaction coefficients (Ridge with interactions)
    print("\nTop interaction coefficients (Interactions_Ridge)")
    interactions_model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
        StandardScaler(),
        Ridge(alpha=1.0),
    )
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

    # Scatter + correlation per feature
    plt = import_optional("matplotlib.pyplot", "matplotlib", check_import="matplotlib")
    df_corr = X_final.copy()
    df_corr[TARGET_COL] = y.values
    corr_series = df_corr.corr(numeric_only=True)[TARGET_COL].drop(TARGET_COL)
    corr_series = corr_series.sort_values(ascending=False)

    print("\nFeature correlations with target (Pearson)")
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

    for j in range(len(NUM_COLS), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    scatter_path = os.path.join(OUTPUTS_DIR, "feature_scatter.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"\nScatter plot saved to {scatter_path}")

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
            print(f"Invalid ML_SAL_MODEL='{env_choice}'. Using {default_name}.")
            return default_name

        print("\nChoose a model to save")
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

        print(f"Invalid choice '{raw}'. Using {default_name}.")
        return default_name

    model_dict = {name: est for name, est in models if name != "DummyMean"}
    model_names = list(model_dict.keys())
    split_best = min(
        [m for m in split_metrics if m["name"] in model_dict],
        key=lambda m: m["rmse"],
    )
    cv_best = min(
        [m for m in cv_metrics if m["name"] in model_dict],
        key=lambda m: m["rmse_mean"],
    )

    recommended_model = cv_best["name"] if cv_best is not None else split_best["name"]
    print("\nMost precise model")
    print(
        f"Holdout best: {split_best['name']} "
        f"(RMSE={split_best['rmse']:.4f}, R2={split_best['r2']:.3f})"
    )
    if cv_best is not None:
        print(
            f"CV best     : {cv_best['name']} "
            f"(RMSE={cv_best['rmse_mean']:.4f}, R2={cv_best['r2_mean']:.3f})"
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
    print(f"\nSaved model '{selected_name}' to {MODEL_PATH}")


# =========================
# Analysis scripts (from analysis/*.py)
# =========================

def global_explanation(db_path=None, outputs_dir=None):
    ensure_core_dependencies()
    db_path = _resolve_path(db_path, DEFAULT_DB_PATH)
    outputs_dir = _resolve_path(outputs_dir, OUTPUTS_DIR)

    shap = import_optional("shap", "shap")
    plt = import_optional("matplotlib.pyplot", "matplotlib", check_import="matplotlib")

    df = load_data(db_path)
    X, y = build_features(df)
    model, imputer = get_model(X, y)
    X_num = transform_features(imputer, X)

    sample = X_num.sample(
        n=min(300, len(X_num)),
        random_state=42,
    )

    background = X_num.sample(
        n=min(200, len(X_num)),
        random_state=42,
    )
    shap_values, shap_mode = compute_shap_values(
        shap,
        model,
        background,
        sample,
    )
    if shap_mode != "explainer":
        print("Using SHAP compatibility mode (KernelExplainer).")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)

    plt.tight_layout()
    os.makedirs(outputs_dir, exist_ok=True)
    shap_global_path = os.path.join(outputs_dir, "shap_global.png")
    plt.savefig(shap_global_path, dpi=150)
    plt.close()

    print(f"Global SHAP (without reference) saved to {shap_global_path}")


def local_explanation(db_path=None, outputs_dir=None):
    ensure_core_dependencies()
    db_path = _resolve_path(db_path, DEFAULT_DB_PATH)
    outputs_dir = _resolve_path(outputs_dir, OUTPUTS_DIR)

    shap = import_optional("shap", "shap")
    plt = import_optional("matplotlib.pyplot", "matplotlib", check_import="matplotlib")

    df = load_data(db_path)
    print(f"\nTotal available cases: {len(df)}")

    lote = input("\nEnter the BATCH to analyze: ").strip()
    df_lote = df[df["lote"] == lote].reset_index(drop=True)

    if df_lote.empty:
        print(f"\nNo records found for batch '{lote}'")
        print("Analysis finished.")
        return

    if len(df_lote) == 1:
        linha = df_lote.iloc[0]
    else:
        print(f"\nFound {len(df_lote)} records for batch {lote}:\n")
        for i, r in df_lote.iterrows():
            target_label = "salt %" if TARGET_COL == "pct_sal" else "salt % diff"
            print(
                f"[{i}] Date: {r['data']} | "
                f"Vat: {r['cuba']} | "
                f"{target_label}: {r[TARGET_COL]:.3f}"
            )

        while True:
            escolha = input("\nChoose the record number: ").strip()
            if escolha.isdigit() and int(escolha) in range(len(df_lote)):
                linha = df_lote.iloc[int(escolha)]
                break
            print("Invalid choice.")

    target_value = linha[TARGET_COL]
    dif_es = linha["dif_es"]
    dif_hfd = linha["dif_hfd"]

    lote = linha["lote"]
    cuba = linha["cuba"]
    data = linha["data"]

    X, y = build_features(df)
    model, imputer = get_model(X, y)
    X_num = transform_features(imputer, X)

    row_idx = df.index[
        (df["lote"] == linha["lote"]) &
        (df["data"] == linha["data"]) &
        (df["cuba"] == linha["cuba"])
    ][0]

    background = X_num.sample(
        n=min(200, len(X_num)),
        random_state=42,
    )
    row_data = X_num.iloc[[row_idx]]
    shap_values, shap_mode = compute_shap_values(
        shap,
        model,
        background,
        row_data,
        nsamples=300,
    )
    if shap_mode != "explainer":
        print("Using SHAP compatibility mode (KernelExplainer).")

    vals = shap_values[0]
    abs_vals = np.abs(vals)
    total = abs_vals.sum()
    percent = abs_vals / total * 100

    impact_df = pd.DataFrame({
        "variavel": X_num.columns,
        "impacto": vals,
        "percent": percent,
        "valor_medido": X_num.iloc[row_idx].values,
    })

    impact_df = impact_df[impact_df["percent"] >= 1]
    impact_df = impact_df.sort_values("percent", ascending=False)

    dominante = impact_df.iloc[0]
    var_dom = dominante["variavel"]

    aplicar_validacao = TARGET_COL == DEVIATION_COL and var_dom in ["dif_es", "dif_hfd"]
    inconclusivo = False
    motivos = []

    if aplicar_validacao:
        if var_dom == "dif_es":
            if (target_value > 0 and dif_es > 0) or (target_value < 0 and dif_es < 0):
                inconclusivo = True
                motivos.append(
                    "ES is the dominant variable, but the deviation sign "
                    "contradicts the expected inverse relationship."
                )

        if var_dom == "dif_hfd":
            if (target_value > 0 and dif_hfd < 0) or (target_value < 0 and dif_hfd > 0):
                inconclusivo = True
                motivos.append(
                    "HFD is the dominant variable, but the deviation sign "
                    "contradicts the expected direct relationship."
                )

    if inconclusivo:
        print("\nINCONCLUSIVE ANALYSIS")
        print("=" * 75)
        print(f"Batch: {lote} | Vat: {cuba} | Date: {data}")
        if TARGET_COL == "pct_sal":
            print(f"Observed salt %: {target_value:.3f}\n")
        else:
            print(f"Observed deviation (salt % diff): {target_value:.3f}\n")

        for m in motivos:
            print(f"- {m}")

        print("\nConclusion:")
        print("The dominant variable violates physical process assumptions.")
        print("This case should not be interpreted via ML.")
        print("\nNo chart was generated.")
        print("Analysis finished.")
        return

    impact_df_plot = impact_df.copy()
    if linha.get("ph_entrada") == 0:
        impact_df_plot = impact_df_plot[impact_df_plot["variavel"] != "ph_entrada"].copy()
        if not impact_df_plot.empty:
            total_plot = impact_df_plot["impacto"].abs().sum()
            impact_df_plot["percent"] = impact_df_plot["impacto"].abs() / total_plot * 100

    impact_df_plot["impacto_abs"] = impact_df_plot["impacto"].abs()
    impact_df_plot = impact_df_plot.sort_values("impacto_abs")

    impact_df_plot["label_y"] = impact_df_plot.apply(
        lambda r: f"{r['variavel']}\nvalue = {r['valor_medido']:.3g}",
        axis=1,
    )

    plt.figure(figsize=(10, 6))
    colors = impact_df_plot["impacto"].apply(
        lambda x: "#d62728" if x > 0 else "#2ca02c"
    )

    bars = plt.barh(
        impact_df_plot["label_y"],
        impact_df_plot["impacto"],
        color=colors,
    )

    plt.axvline(0, color="black", linewidth=0.8)

    for bar, impacto, perc in zip(
        bars,
        impact_df_plot["impacto"],
        impact_df_plot["percent"],
    ):
        y = bar.get_y() + bar.get_height() / 2
        label = f"{impacto:+.3f} ({perc:.1f}%)"

        if perc >= 15:
            plt.text(
                impacto * 0.95,
                y,
                label,
                va="center",
                ha="right" if impacto > 0 else "left",
                color="white",
                fontsize=9,
            )
        else:
            offset = 0.002
            x_text = -offset if impacto > 0 else offset
            plt.text(
                x_text,
                y,
                label,
                va="center",
                ha="right" if impacto > 0 else "left",
                fontsize=9,
                color="black",
            )

    label_target = "predicted salt %" if TARGET_COL == "pct_sal" else "predicted deviation"
    plt.xlabel(f"Local impact on {label_target}")
    title_label = "salt %" if TARGET_COL == "pct_sal" else "salt % diff"
    plt.title(
        f"Local analysis of salt\n"
        f"Batch: {lote} | Vat: {cuba} | {title_label}: {target_value:.3f}",
        fontsize=11,
    )

    plt.tight_layout()
    os.makedirs(outputs_dir, exist_ok=True)
    shap_local_path = os.path.join(outputs_dir, "shap_local.png")
    plt.savefig(shap_local_path, dpi=150)
    plt.close()

    print("\nRESULT INTERPRETATION")
    print("=" * 75)
    print("The analysis was considered COHERENT.")
    print("The dominant variable respects the")
    print("physical process assumptions, allowing interpretation.")
    print(f"\nFile generated: {shap_local_path}")


def what_if_simulation(db_path=None):
    ensure_core_dependencies()
    db_path = _resolve_path(db_path, DEFAULT_DB_PATH)
    df = load_data(db_path)

    print(f"\nTotal available cases: {len(df)}")
    row_id = int(input("Choose the row_id for the base case: "))

    if row_id < 0 or row_id >= len(df):
        raise ValueError("Invalid row_id")

    desvio_real = df.iloc[row_id][TARGET_COL]
    X, y = build_features(df)

    model, imputer = get_model(X, y)
    X_num = transform_features(imputer, X)

    x_base = X_num.iloc[row_id].copy()
    desvio_base = model.predict(pd.DataFrame([x_base]))[0]

    print("\nBASE CASE")
    if TARGET_COL == "pct_sal":
        print(f"REAL salt %        : {desvio_real:.3f}")
        print(f"PREDICTED salt %   : {desvio_base:.3f}")
    else:
        print(f"REAL deviation     : {desvio_real:.3f}")
        print(f"PREDICTED deviation: {desvio_base:.3f}")
    err_abs = abs(desvio_real - desvio_base)
    err_pct = err_abs / max(abs(desvio_real), 0.01) * 100
    print(f"Unexplained % (obs vs baseline): {err_pct:.1f}% (eps=0.01)")

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
                print("Invalid variable")
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

            print("\nRESULT")
            for v, (a, n) in alteracoes.items():
                print(f"{v}: {a} -> {n}")
        if TARGET_COL == "pct_sal":
            print(f"Predicted salt %: {desvio_sim:.3f}")
        else:
            print(f"Predicted deviation: {desvio_sim:.3f}")
            print(f"Impact vs base: {impacto:+.3f}")

        if input("\nNew simulation? (y/n): ").lower() != "y":
            break


# =========================
# Streamlit app (from app.py)
# =========================

def streamlit_app():
    ensure_core_dependencies()
    st = import_optional("streamlit", "streamlit")
    shap = import_optional("shap", "shap")
    plt = import_optional("matplotlib.pyplot", "matplotlib", check_import="matplotlib")

    st.set_page_config(page_title="ML-Sal", layout="wide")
    ERROR_EPS = 0.01

    def check_inconclusive(impact_df, dif_sal, dif_es, dif_hfd):
        if impact_df.empty:
            return False, []

        dominante = impact_df.iloc[0]
        var_dom = dominante["variavel"]
        motivos = []

        if var_dom == "dif_es":
            if (dif_sal > 0 and dif_es > 0) or (dif_sal < 0 and dif_es < 0):
                motivos.append(
                    "ES is the dominant variable, but the deviation sign "
                    "contradicts the expected inverse relationship."
                )

        if var_dom == "dif_hfd":
            if (dif_sal > 0 and dif_hfd < 0) or (dif_sal < 0 and dif_hfd > 0):
                motivos.append(
                    "HFD is the dominant variable, but the deviation sign "
                    "contradicts the expected direct relationship."
                )

        return len(motivos) > 0, motivos

    def build_impact_df(model, imputer, row_values, background):
        X_row = pd.DataFrame([row_values], columns=NUM_COLS)
        X_row_num = transform_features(imputer, X_row)

        shap_values, shap_mode = compute_shap_values(
            shap,
            model,
            background,
            X_row_num,
            nsamples=300,
        )
        if shap_mode != "explainer":
            st.info("Using SHAP compatibility mode (KernelExplainer).")

        vals = shap_values[0]
        abs_vals = np.abs(vals)
        total = abs_vals.sum() if abs_vals.sum() != 0 else 1.0
        percent = abs_vals / total * 100

        impact_df = pd.DataFrame({
            "variavel": X_row_num.columns,
            "impacto": vals,
            "percent": percent,
            "valor_medido": X_row_num.iloc[0].values,
        })

        impact_df = impact_df[impact_df["percent"] >= 1]
        impact_df = impact_df.sort_values("percent", ascending=False)
        return impact_df, X_row_num

    def plot_impact(impact_df, ph_entrada_value):
        if impact_df.empty:
            st.warning("No variables above 1% impact.")
            return

        impact_df_plot = impact_df.copy()
        if ph_entrada_value == 0:
            impact_df_plot = impact_df_plot[
                impact_df_plot["variavel"] != "ph_entrada"
            ].copy()
            if not impact_df_plot.empty:
                total_plot = impact_df_plot["impacto"].abs().sum()
                impact_df_plot["percent"] = (
                    impact_df_plot["impacto"].abs() / total_plot * 100
                )

        if impact_df_plot.empty:
            st.warning("All variables removed after filtering.")
            return

        impact_df_plot["impacto_abs"] = impact_df_plot["impacto"].abs()
        impact_df_plot = impact_df_plot.sort_values("impacto_abs")

        impact_df_plot["label_y"] = impact_df_plot.apply(
            lambda r: f"{r['variavel']}\nvalue = {r['valor_medido']:.3g}",
            axis=1,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = impact_df_plot["impacto"].apply(
            lambda x: "#d62728" if x > 0 else "#2ca02c"
        )

        bars = ax.barh(
            impact_df_plot["label_y"],
            impact_df_plot["impacto"],
            color=colors,
        )

        ax.axvline(0, color="black", linewidth=0.8)

        for bar, impacto, perc in zip(
            bars,
            impact_df_plot["impacto"],
            impact_df_plot["percent"],
        ):
            y = bar.get_y() + bar.get_height() / 2
            label = f"{impacto:+.3f} ({perc:.1f}%)"

            if perc >= 15:
                ax.text(
                    impacto * 0.95,
                    y,
                    label,
                    va="center",
                    ha="right" if impacto > 0 else "left",
                    color="white",
                    fontsize=9,
                )
            else:
                offset = 0.002
                x_text = -offset if impacto > 0 else offset
                ax.text(
                    x_text,
                    y,
                    label,
                    va="center",
                    ha="right" if impacto > 0 else "left",
                    fontsize=9,
                    color="black",
                )

        target_label = "predicted salt %" if TARGET_COL == "pct_sal" else "predicted deviation"
        ax.set_xlabel(f"Local impact on {target_label}")
        st.pyplot(fig)

    def validate_float(raw, label, min_v=None, max_v=None):
        if raw is None or raw.strip() == "":
            return None, f"{label}: value cannot be empty."

        value_str = raw.strip().replace(",", ".")
        try:
            value = float(value_str)
        except Exception:
            return None, f"{label}: invalid value."

        if min_v is not None and value < min_v:
            return None, f"{label}: must be >= {min_v}."
        if max_v is not None and value > max_v:
            return None, f"{label}: must be <= {max_v}."
        return value, None

    def validate_int(raw, label, min_v=None, max_v=None):
        if raw is None or raw.strip() == "":
            return None, f"{label}: value cannot be empty."

        try:
            value = int(raw.strip())
        except Exception:
            return None, f"{label}: invalid integer."

        if min_v is not None and value < min_v:
            return None, f"{label}: must be >= {min_v}."
        if max_v is not None and value > max_v:
            return None, f"{label}: must be <= {max_v}."
        return value, None

    @st.cache_data
    def get_data():
        return load_data()

    @st.cache_resource
    def get_model_cached(df):
        X, y = build_features(df)
        model, imputer = get_model(X, y)
        X_num = transform_features(imputer, X)
        background = X_num.sample(n=min(200, len(X_num)), random_state=42)
        return model, imputer, background

    st.title("ML-Sal - Local Explanation")

    mode = st.radio(
        "Choose a mode",
        ("Use existing record", "Manual input (no DB write)", "What-if simulation"),
    )

    df = get_data()
    if df.empty:
        st.error("No data available in the database.")
        st.stop()

    model, imputer, background = get_model_cached(df)

    if mode == "Use existing record":
        st.subheader("Select a record")

        lotes = sorted(df["lote"].dropna().unique())
        if not lotes:
            st.error("No batches available.")
            st.stop()

        lote = st.selectbox("Batch", lotes)
        df_lote = df[df["lote"] == lote].copy().reset_index()

        st.dataframe(
            df_lote[["index", "data", "cuba", TARGET_COL]],
            use_container_width=True,
        )

        def label_row(r):
            target_label = "salt %" if TARGET_COL == "pct_sal" else "diff"
            return (
                f"{r['index']} | {r['data']} | Vat {r['cuba']} | "
                f"{target_label} {r[TARGET_COL]:.3f}"
            )

        options = df_lote.apply(label_row, axis=1).tolist()
        selected = st.selectbox("Record", options)
        row_idx = int(selected.split("|")[0].strip())

        row = df.loc[row_idx]
        row_values = row[NUM_COLS].values
        impact_df, _ = build_impact_df(model, imputer, row_values, background)

        target_value = row[TARGET_COL]
        dif_es = row["dif_es"]
        dif_hfd = row["dif_hfd"]

        st.markdown("---")
        st.write(
            f"**Batch:** {row['lote']} | **Vat:** {row['cuba']} | **Date:** {row['data']}"
        )
        target_label = "salt %" if TARGET_COL == "pct_sal" else "deviation"
        st.write(f"**Observed {target_label}:** {target_value:.3f}")

        if TARGET_COL == DEVIATION_COL:
            inconclusivo, motivos = check_inconclusive(impact_df, target_value, dif_es, dif_hfd)
        else:
            inconclusivo, motivos = False, []
        if inconclusivo:
            st.error("INCONCLUSIVE ANALYSIS")
            for m in motivos:
                st.write(f"- {m}")
        else:
            st.success("COHERENT ANALYSIS")
            plot_impact(impact_df, row["ph_entrada"])

    elif mode == "Manual input (no DB write)":
        st.subheader("Manual input (not saved)")

        references = sorted(df["referencia"].dropna().unique())
        with st.form("manual_form"):
            st.markdown("**General data**")
            col1, col2, col3 = st.columns(3)
            with col1:
                date = st.date_input("Date")
            with col2:
                if references:
                    reference = st.selectbox("Reference", references)
                else:
                    reference = st.text_input("Reference")
            with col3:
                cuba_raw = st.text_input("Vat (integer)")

            lote = st.text_input("Batch")

            st.markdown("**Main variables**")
            col1, col2 = st.columns(2)
            with col1:
                pct_sal_raw = st.text_input("Measured salt % (e.g.: 1.60)")
            with col2:
                if TARGET_COL == DEVIATION_COL:
                    dif_pct_sal_raw = st.text_input("Salt % difference (TARGET) (e.g.: -0.040)")

            st.markdown("**Process variables**")
            col1, col2 = st.columns(2)
            with col1:
                dif_es_raw = st.text_input("Dif_ES (e.g.: 0.90)")
                ph_entrada = st.selectbox(
                    "Input pH",
                    [0, 1],
                    format_func=lambda x: "OK" if x == 0 else "NOK",
                )
                densidade_raw = st.text_input("Density (e.g.: 18.8)")
            with col2:
                dif_gs_raw = st.text_input("Dif_GS (e.g.: 0.20)")
                min_fora_raw = st.text_input("Minutes out of spec (e.g.: 25)")

            submitted = st.form_submit_button("Run analysis")

        if submitted:
            errors = []
            if lote.strip() == "":
                errors.append("Batch cannot be empty.")

            if reference is None or str(reference).strip() == "":
                errors.append("Reference cannot be empty.")

            cuba, err = validate_int(cuba_raw, "Vat", min_v=1)
            if err:
                errors.append(err)

            pct_sal, err = validate_float(pct_sal_raw, "Measured salt %", min_v=0)
            if err:
                errors.append(err)

            dif_pct_sal = None
            if TARGET_COL == DEVIATION_COL:
                dif_pct_sal, err = validate_float(dif_pct_sal_raw, "Salt % difference")
                if err:
                    errors.append(err)

            dif_es, err = validate_float(dif_es_raw, "Dif_ES")
            if err:
                errors.append(err)

            dif_gs, err = validate_float(dif_gs_raw, "Dif_GS")
            if err:
                errors.append(err)

            densidade, err = validate_float(densidade_raw, "Density", min_v=0)
            if err:
                errors.append(err)
            elif densidade <= 0:
                errors.append("Density must be > 0 (0 is excluded from model training).")

            min_fora, err = validate_float(min_fora_raw, "Minutes out of spec", min_v=0)
            if err:
                errors.append(err)

            if errors:
                for e in errors:
                    st.error(e)
            else:
                row_values = [
                    dif_es,
                    dif_gs,
                    ph_entrada,
                    densidade,
                    min_fora,
                ]

                impact_df, X_row_num = build_impact_df(model, imputer, row_values, background)
                pred = model.predict(X_row_num)[0]

                st.markdown("---")
                st.write(
                    f"**Batch:** {lote} | **Vat:** {cuba} | **Date:** {date}"
                )
                target_label = "salt %" if TARGET_COL == "pct_sal" else "deviation"
                target_value = dif_pct_sal if TARGET_COL == DEVIATION_COL else pct_sal
                st.write(f"**Observed {target_label}:** {target_value:.3f}")
                st.write(f"**Predicted {target_label}:** {pred:.3f}")

                if TARGET_COL == DEVIATION_COL:
                    dif_hfd = 0.0
                    inconclusivo, motivos = check_inconclusive(
                        impact_df, dif_pct_sal, dif_es, dif_hfd
                    )
                else:
                    inconclusivo, motivos = False, []

                if inconclusivo:
                    st.error("INCONCLUSIVE ANALYSIS")
                    for m in motivos:
                        st.write(f"- {m}")
                else:
                    st.success("COHERENT ANALYSIS")
                    plot_impact(impact_df, ph_entrada)

    else:
        st.subheader("What-if simulation")

        lotes = sorted(df["lote"].dropna().unique())
        if not lotes:
            st.error("No batches available.")
            st.stop()

        lote = st.selectbox("Batch", lotes, key="what_if_batch")
        df_lote = df[df["lote"] == lote].copy().reset_index()

        st.dataframe(
            df_lote[["index", "data", "cuba", TARGET_COL]],
            use_container_width=True,
        )

        def label_row_wi(r):
            target_label = "salt %" if TARGET_COL == "pct_sal" else "diff"
            return (
                f"{r['index']} | {r['data']} | Vat {r['cuba']} | "
                f"{target_label} {r[TARGET_COL]:.3f}"
            )

        options = df_lote.apply(label_row_wi, axis=1).tolist()
        selected = st.selectbox("Base record", options, key="what_if_record")
        row_idx = int(selected.split("|")[0].strip())

        row = df.loc[row_idx]
        raw_values = row[NUM_COLS].values
        base_X = transform_features(imputer, pd.DataFrame([raw_values], columns=NUM_COLS))
        base_values = base_X.iloc[0]
        base_values_list = base_values[NUM_COLS].values.astype(float)
        base_pred = model.predict(base_X)[0]
        obs = float(row[TARGET_COL])
        err_abs = abs(obs - base_pred)
        err_pct = err_abs / max(abs(obs), ERROR_EPS) * 100

        st.markdown("---")
        st.write(
            f"**Batch:** {row['lote']} | **Vat:** {row['cuba']} | **Date:** {row['data']}"
        )
        target_label = "salt %" if TARGET_COL == "pct_sal" else "deviation"
        st.write(f"**Observed {target_label}:** {obs:.3f}")
        st.write(f"**Baseline predicted {target_label}:** {base_pred:.3f}")
        st.write(f"**Unexplained % (obs vs baseline):** {err_pct:.1f}%")
        st.caption("Relative error uses max(|observed|, 0.01) to avoid near-zero blow-up.")

        with st.form("what_if_form"):
            st.markdown("**Adjust variables**")
            col1, col2 = st.columns(2)

            with col1:
                dif_es = st.number_input("Dif_ES", value=float(base_values["dif_es"]))
                ph_entrada = st.selectbox(
                    "Input pH",
                    [0, 1],
                    index=int(round(base_values["ph_entrada"])),
                    format_func=lambda x: "OK" if x == 0 else "NOK",
                )
                dens_min = min(0.0, float(base_values["densidade"]))
                densidade = st.number_input(
                    "Density",
                    min_value=dens_min,
                    value=float(base_values["densidade"]),
                )

            with col2:
                dif_gs = st.number_input("Dif_GS", value=float(base_values["dif_gs"]))
                min_fora_min = min(0.0, float(base_values["min_fora"]))
                min_fora = st.number_input(
                    "Minutes out of spec",
                    min_value=min_fora_min,
                    value=float(base_values["min_fora"]),
                )

            submitted = st.form_submit_button("Run what-if")

        if submitted:
            sim_values = [
                dif_es,
                dif_gs,
                ph_entrada,
                densidade,
                min_fora,
            ]

            sim_X = transform_features(imputer, pd.DataFrame([sim_values], columns=NUM_COLS))
            sim_pred = model.predict(sim_X)[0]
            impacto = sim_pred - base_pred

            st.markdown("---")
            st.write(f"**Simulated predicted {target_label}:** {sim_pred:.3f}")
            st.write(f"**Impact vs baseline:** {impacto:+.3f}")

            changes = []
            for name, base_val, new_val in zip(NUM_COLS, base_values_list, sim_values):
                if abs(float(new_val) - float(base_val)) > 1e-9:
                    changes.append({
                        "variable": name,
                        "base": float(base_val),
                        "new": float(new_val),
                        "delta": float(new_val) - float(base_val),
                    })

            if changes:
                st.markdown("**Changed variables**")
                st.dataframe(pd.DataFrame(changes), use_container_width=True)
            else:
                st.info("No changes from base values.")


# =========================
# CLI menu (from menu.py)
# =========================

def run_streamlit():
    try:
        import_optional("streamlit", "streamlit")
    except Exception as exc:
        print(str(exc))
        return

    script_path = os.path.abspath(__file__)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", script_path, "--", "--streamlit"],
        check=False,
    )


MENU = {
    "1": ("Import history", import_history),
    "2": ("Verify database", verify_database),
    "3": ("Verify inserts", verify_inserts),
    "4": ("Manual insert", insert_deviation_manual),
    "5": ("Data quality report", lambda: data_quality_report(load_data())),
    "6": ("List variables (CSV)", list_variables),
    "7": ("Pearson heatmap (CSV/DB)", pearson_heatmap_analysis),
    "8": ("Prepare ML data", prepare_ml_data),
    "9": ("Train baseline model", train_baseline_model),
    "10": ("Local explanation", local_explanation),
    "11": ("Global explanation", global_explanation),
    "12": ("What-if simulation", what_if_simulation),
    "13": ("Web interface (Streamlit)", run_streamlit),
    "14": ("Delete last record", delete_last_record),
    "0": ("Exit", None),
}

MENU_GROUPS = [
    ("Data Input & Validation", ["1", "2", "3", "4"]),
    ("Pre-Model Analysis", ["5", "6", "7", "8"]),
    ("Model Training", ["9"]),
    ("Final Analysis & Simulation", ["10", "11", "12", "13"]),
    ("Maintenance", ["14", "0"]),
]


def run_menu():
    while True:
        print("\nML-Sal Menu")
        print("=" * 40)
        for group_name, keys in MENU_GROUPS:
            print(f"\n{group_name}")
            for key in keys:
                label, _ = MENU[key]
                print(f"{key}. {label}")

        choice = input("\nChoose an option: ").strip()
        if choice not in MENU:
            print("Invalid option.")
            continue

        label, func = MENU[choice]
        if func is None:
            print("Bye.")
            return

        print(f"\nRunning: {label}")
        func()


def main():
    if "--streamlit" in sys.argv:
        streamlit_app()
        return
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        return
    try:
        ensure_core_dependencies()
    except ImportError as exc:
        print(str(exc))
        return
    run_menu()


if __name__ == "__main__":
    main()
