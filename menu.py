import subprocess
import sys

MENU = {
    "1": ("Import history", "database/import_history.py"),
    "2": ("Import extra variables from CSV", "database/import_variables_from_csv.py"),
    "3": ("Verify database", "database/verify_database.py"),
    "4": ("Data quality report", "utilities/data_quality.py"),
    "5": ("Local explanation", "analysis/local_explanation.py"),
    "6": ("Global explanation", "analysis/global_explanation.py"),
    "7": ("What-if simulation", "analysis/what_if_simulation.py"),
    "8": ("Manual insert", "utilities/insert_deviation_manual.py"),
    "9": ("Verify inserts", "database/verify_inserts.py"),
    "10": ("Delete last record", "database/delete_last_record.py"),
    "11": ("Web interface (Streamlit)", "__STREAMLIT__"),
    "0": ("Exit", None),
}


def run_script(script_name):
    subprocess.run([sys.executable, script_name], check=False)


def run_streamlit():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=False)


def main():
    while True:
        print("\n🧭 ML-Sal Menu")
        print("=" * 40)
        for key, (label, _) in MENU.items():
            print(f"{key}. {label}")

        choice = input("\nChoose an option: ").strip()
        if choice not in MENU:
            print("❌ Invalid option.")
            continue

        label, script = MENU[choice]
        if script is None:
            print("✅ Bye!")
            return

        print(f"\n▶️ Running: {label}")
        if script == "__STREAMLIT__":
            run_streamlit()
        else:
            run_script(script)


if __name__ == "__main__":
    main()
