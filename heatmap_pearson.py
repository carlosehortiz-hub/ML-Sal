"""
Generate a Pearson correlation heatmap from matriz_cloretos.csv (or any CSV),
and provide data-driven suggestions for excluding variables.

Usage:
  python heatmap_pearson.py
  python heatmap_pearson.py --csv matriz_cloretos.csv --out outputs/pearson_heatmap.png
  python heatmap_pearson.py --min-target 0.05 --max-pair 0.90 --write-filtered outputs/matriz_filtrada.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def clean_csv(csv_path):
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

    return df


def compute_correlations(df):
    df_num = df.select_dtypes(include="number")
    corr = df_num.corr(method="pearson")
    return df_num, corr


def find_high_pairs(corr, threshold):
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = []
    for row in upper.index:
        for col in upper.columns:
            val = upper.loc[row, col]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append((row, col, float(val)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def suggest_exclusions(corr, target_col, min_target, max_pair):
    low_target = []
    target_corr = None
    if target_col in corr.columns:
        target_corr = corr[target_col].drop(labels=[target_col]).abs().sort_values()
        low_target = target_corr[target_corr < min_target].index.tolist()

    pairs = find_high_pairs(corr, max_pair)
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
    return target_corr, pairs, low_target, collinear_drop, suggested


def plot_heatmap(corr, out_path):
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except Exception:
        print("Missing seaborn/matplotlib. Install with: pip install seaborn matplotlib")
        return False

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)

    plt.show()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Pearson heatmap and exclusion helper.")
    parser.add_argument("--csv", default="matriz_cloretos.csv", help="Input CSV path.")
    parser.add_argument("--out", default="outputs/pearson_heatmap.png", help="Output image path.")
    parser.add_argument("--target", default="dif_pct_sal", help="Target column name.")
    parser.add_argument("--min-target", type=float, default=0.05,
                        help="Minimum abs correlation with target to keep.")
    parser.add_argument("--max-pair", type=float, default=0.90,
                        help="High collinearity threshold for pairwise correlations.")
    parser.add_argument("--top-pairs", type=int, default=20,
                        help="Show top N correlated pairs.")
    parser.add_argument("--exclude", default="",
                        help="Comma-separated columns to exclude manually.")
    parser.add_argument("--write-filtered", default="",
                        help="Write filtered CSV to this path.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        sys.exit(1)

    df = clean_csv(args.csv)
    df_num, corr = compute_correlations(df)

    print("\nNumeric columns used in correlation:")
    print(", ".join(df_num.columns))

    target_corr, pairs, low_target, collinear_drop, suggested = suggest_exclusions(
        corr,
        args.target,
        args.min_target,
        args.max_pair,
    )

    if target_corr is not None:
        print(f"\nCorrelation with target '{args.target}' (abs sorted):")
        print(target_corr.sort_values(ascending=False))
    else:
        print(f"\nTarget '{args.target}' not found in numeric columns.")

    print(f"\nTop correlated pairs (abs >= {args.max_pair}):")
    if pairs:
        for a, b, v in pairs[:args.top_pairs]:
            print(f"{a} <-> {b}: {v:+.3f}")
    else:
        print("No pairs above threshold.")

    print(f"\nLow correlation with target (abs < {args.min_target}):")
    print(low_target if low_target else "None")

    print(f"\nSuggested drops due to collinearity (>= {args.max_pair}):")
    print(sorted(set(collinear_drop)) if collinear_drop else "None")

    manual_exclude = [c.strip() for c in args.exclude.split(",") if c.strip()]
    final_exclude = sorted(set(suggested + manual_exclude))

    print("\nSuggested exclusions (combined):")
    print(final_exclude if final_exclude else "None")

    if args.write_filtered:
        keep_cols = [c for c in df.columns if c not in final_exclude]
        df_filtered = df[keep_cols].copy()
        os.makedirs(os.path.dirname(args.write_filtered), exist_ok=True)
        df_filtered.to_csv(args.write_filtered, index=False)
        print(f"\nFiltered CSV written to: {args.write_filtered}")

    plot_heatmap(corr, args.out)


if __name__ == "__main__":
    main()
