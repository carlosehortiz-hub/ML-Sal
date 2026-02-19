import os
import re
import sqlite3

import pandas as pd

DB_PATH = "ml_sal.db"
DEFAULT_CSV = "novas_variaveis.csv"
UNIQUE_KEY_COLS = ["lote", "referencia", "cuba"]
UNIQUE_INDEX_NAME = "ux_desvios_sal_lote_ref_cuba"


def normalize_db_column_name(name):
    cleaned = str(name).strip().lower()
    cleaned = cleaned.replace("%", "pct")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "col"
    if cleaned[0].isdigit():
        cleaned = f"c_{cleaned}"
    return cleaned


def normalize_columns_unique(df):
    rename_map = {}
    used = set()
    for col in df.columns:
        base = normalize_db_column_name(col)
        final = base
        idx = 2
        while final in used:
            final = f"{base}_{idx}"
            idx += 1
        rename_map[col] = final
        used.add(final)
    return df.rename(columns=rename_map)


def normalize_unique_key_columns(df):
    df = df.copy()
    for col in UNIQUE_KEY_COLS:
        if col not in df.columns:
            continue
        if col == "cuba":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return df


def infer_sql_type_from_series(series):
    if series.dropna().empty:
        return "REAL"

    if pd.api.types.is_numeric_dtype(series):
        non_na = series.dropna()
        if (non_na % 1 == 0).all():
            return "INTEGER"
        return "REAL"

    as_text = series.astype(str).str.strip()
    as_text = as_text.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    numeric = pd.to_numeric(
        as_text.str.replace(",", ".", regex=False),
        errors="coerce",
    )
    n_non_na = int(as_text.notna().sum())
    n_num = int(numeric.notna().sum())
    if n_non_na > 0 and n_num == n_non_na:
        if (numeric.dropna() % 1 == 0).all():
            return "INTEGER"
        return "REAL"
    return "TEXT"


def prepare_series_for_sql(series, sql_type):
    if sql_type in {"INTEGER", "REAL"}:
        prepared = pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if sql_type == "INTEGER":
            return prepared.astype("Int64")
        return prepared.astype(float)

    prepared = series.astype(str).str.strip()
    prepared = prepared.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return prepared


def has_db_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def values_differ(a, b):
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True

    try:
        if float(a) == float(b):
            return False
    except Exception:
        pass
    return str(a) != str(b)


def main():
    csv_path = input(f"CSV file with extra variables [Enter={DEFAULT_CSV}]: ").strip()
    csv_path = csv_path or DEFAULT_CSV

    env_overwrite = os.environ.get("ML_SAL_IMPORT_OVERWRITE", "").strip().lower()
    if env_overwrite in {"1", "true", "yes", "y", "sim", "s"}:
        overwrite_existing = True
    elif env_overwrite in {"0", "false", "no", "n", "nao", "não"}:
        overwrite_existing = False
    else:
        overwrite_raw = input(
            "Overwrite existing values already present in DB? (y/N): "
        ).strip().lower()
        overwrite_existing = overwrite_raw in {"y", "yes", "1", "sim", "s"}

    mode_label = "overwrite enabled" if overwrite_existing else "fill only empty fields"
    print(f"Import mode: {mode_label}")

    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, sep=None, engine="python")
    if df.empty:
        print("❌ CSV is empty.")
        return

    df = normalize_columns_unique(df)
    missing = [c for c in UNIQUE_KEY_COLS if c not in df.columns]
    if missing:
        print(
            "❌ CSV must include key columns: lote, referencia, cuba "
            f"(missing: {missing})"
        )
        return

    df = normalize_unique_key_columns(df)
    invalid_key_rows = int(df[UNIQUE_KEY_COLS].isna().any(axis=1).sum())
    if invalid_key_rows > 0:
        df = df[~df[UNIQUE_KEY_COLS].isna().any(axis=1)].copy()

    total_rows = len(df)
    before_internal = len(df)
    df = df.drop_duplicates(subset=UNIQUE_KEY_COLS, keep="last")
    skipped_internal = before_internal - len(df)

    value_cols = [c for c in df.columns if c not in UNIQUE_KEY_COLS]
    if not value_cols:
        print("❌ No extra variables found in CSV.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    table_exists = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='desvios_sal'"
    ).fetchone()
    if table_exists is None:
        conn.close()
        print("❌ Table 'desvios_sal' not found. Import history first.")
        return

    try:
        cursor.execute(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS {UNIQUE_INDEX_NAME}
            ON desvios_sal (lote, referencia, cuba)
            """
        )
    except sqlite3.IntegrityError:
        pass

    table_cols = {
        row[1] for row in cursor.execute("PRAGMA table_info(desvios_sal)").fetchall()
    }
    added_cols = []
    sql_types = {}
    for col in value_cols:
        sql_type = infer_sql_type_from_series(df[col])
        sql_types[col] = sql_type
        if col not in table_cols:
            cursor.execute(f'ALTER TABLE desvios_sal ADD COLUMN "{col}" {sql_type}')
            added_cols.append(f"{col} ({sql_type})")

    for col in value_cols:
        df[col] = prepare_series_for_sql(df[col], sql_types[col])

    updated_keys = 0
    missing_keys = 0
    unchanged_keys = 0
    skipped_prefilled_cells = 0
    overwritten_cells = 0
    select_cols = ", ".join([f'"{c}"' for c in value_cols])
    select_sql = (
        f"SELECT id, {select_cols} FROM desvios_sal "
        "WHERE lote = ? AND referencia = ? AND cuba = ? "
        "LIMIT 1"
    )
    for _, row in df.iterrows():
        key_values = [row["lote"], row["referencia"], int(row["cuba"])]
        cursor.execute(select_sql, key_values)
        existing_row = cursor.fetchone()
        if existing_row is None:
            missing_keys += 1
            continue

        row_id = existing_row[0]
        existing_by_col = {
            col: existing_row[idx + 1]
            for idx, col in enumerate(value_cols)
        }

        cols_to_update = []
        update_values = []
        for col in value_cols:
            incoming = row[col]
            if pd.isna(incoming):
                continue
            existing_value = existing_by_col[col]
            existing_has_value = has_db_value(existing_value)

            if existing_has_value and not overwrite_existing:
                skipped_prefilled_cells += 1
                continue

            if hasattr(incoming, "item"):
                incoming = incoming.item()

            if existing_has_value and overwrite_existing:
                if values_differ(existing_value, incoming):
                    overwritten_cells += 1
                else:
                    continue

            cols_to_update.append(col)
            update_values.append(incoming)

        if not cols_to_update:
            unchanged_keys += 1
            continue

        set_clause = ", ".join([f'"{c}" = ?' for c in cols_to_update])
        cursor.execute(
            f"UPDATE desvios_sal SET {set_clause} WHERE id = ?",
            update_values + [row_id],
        )
        updated_keys += 1

    conn.commit()
    conn.close()

    print("✅ Extra variables import completed.")
    print(f"📊 Rows read from CSV: {total_rows}")
    print(f"⚠️ Rows skipped (invalid key): {invalid_key_rows}")
    print(f"⚠️ Rows skipped (duplicate key in CSV): {skipped_internal}")
    print(f"✅ Rows updated in DB: {updated_keys}")
    print(f"⚠️ Rows with key not found in DB: {missing_keys}")
    print(f"ℹ️ Rows not changed (already filled / no new values): {unchanged_keys}")
    print(f"ℹ️ Cells skipped (already had value): {skipped_prefilled_cells}")
    print(f"ℹ️ Cells overwritten (previous value replaced): {overwritten_cells}")
    if added_cols:
        print("🧱 New columns created:")
        for item in added_cols:
            print(f" - {item}")
    else:
        print("ℹ️ No new columns were created (all already existed).")


if __name__ == "__main__":
    main()
