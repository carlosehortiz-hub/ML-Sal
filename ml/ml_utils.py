import os
import sqlite3
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

NUM_COLS = [
    "dif_es",
    "dif_hfd",
    "dif_gs",
    "ph_entrada",
    "ph_salga",
    "densidade",
    "temperatura",
    "min_fora",
    "tempo_fora_espec",
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

MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")


def load_data(db_path=None):
    if db_path is None:
        db_path = os.path.join(BASE_DIR, "ml_sal.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM desvios_sal", conn)
    conn.close()

    df = df.dropna(subset=["dif_pct_sal"]).reset_index(drop=True)
    return df


def build_features(df):
    y = df["dif_pct_sal"]
    X = df.drop(columns=DROP_COLS)
    return X, y


def fit_model(X, y, n_estimators=300, random_state=42):
    imputer = SimpleImputer(strategy="median")
    X_num = pd.DataFrame(
        imputer.fit_transform(X[NUM_COLS]),
        columns=NUM_COLS,
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_num, y)

    return model, imputer, X_num


def _load_model(path):
    if not os.path.exists(path):
        return None
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        return None
    if "model" not in payload or "imputer" not in payload:
        return None
    return payload


def _save_model(model, imputer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "num_cols": NUM_COLS,
        },
        path,
    )


def get_model(X, y, model_path=MODEL_PATH):
    retrain = os.environ.get("RETRAIN_MODEL") == "1"

    payload = None if retrain else _load_model(model_path)
    if payload is not None and payload.get("num_cols") == NUM_COLS:
        return payload["model"], payload["imputer"]

    model, imputer, _ = fit_model(X, y)
    _save_model(model, imputer, model_path)
    return model, imputer


def transform_features(imputer, X):
    return pd.DataFrame(
        imputer.transform(X[NUM_COLS]),
        columns=NUM_COLS,
    )


def data_quality_report(df):
    print("\n🔎 DATA QUALITY REPORT")
    print("=" * 70)

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("✅ No missing values detected.")
    else:
        print("❗ Missing values by column:")
        print(missing)

    print("\n⚙️  Rule checks")
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

        if min_v is not None:
            below = (series < min_v).sum()
        else:
            below = 0

        if max_v is not None:
            above = (series > max_v).sum()
        else:
            above = 0

        if below == 0 and above == 0:
            print(f"✅ {col}: within expected range")
        else:
            print(f"⚠️  {col}: {below} below, {above} above")

    print("\n📈 Outliers (IQR method)")
    numeric_cols = [c for c in NUM_COLS + ["dif_pct_sal"] if c in df.columns]
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
        print("✅ No IQR outliers detected.")
