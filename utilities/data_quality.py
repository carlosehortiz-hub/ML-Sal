from ml.ml_utils import load_data, data_quality_report


def main():
    df = load_data()
    print(f"\n📊 Total available cases: {len(df)}")
    data_quality_report(df)


if __name__ == "__main__":
    main()
