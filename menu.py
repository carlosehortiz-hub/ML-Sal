import subprocess
import sys

MENU = {
    "1": ("Import history", "database/import_history.py"),
    "2": ("Verify database", "database/verify_database.py"),
    "3": ("Data quality report", "utilities/data_quality.py"),
    "4": ("Local explanation", "analysis/local_explanation.py"),
    "5": ("Global explanation", "analysis/global_explanation.py"),
    "6": ("What-if simulation", "analysis/what_if_simulation.py"),
    "7": ("Manual insert", "utilities/insert_deviation_manual.py"),
    "8": ("Verify inserts", "database/verify_inserts.py"),
    "9": ("Delete last record", "database/delete_last_record.py"),
    "0": ("Exit", None),
}


def run_script(script_name):
    subprocess.run([sys.executable, script_name], check=False)


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
        run_script(script)


if __name__ == "__main__":
    main()
