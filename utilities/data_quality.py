import sys
from pathlib import Path

# Ensure project root is on sys.path when running from subdirectories.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.ml_utils import load_data, data_quality_report


def main():
    df = load_data()
    print(f"\n📊 Total available cases: {len(df)}")
    data_quality_report(df)


if __name__ == "__main__":
    main()
