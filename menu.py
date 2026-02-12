import subprocess
import sys

MENU = {
    "1": ("Import history", "base_dados/importar_historico.py"),
    "2": ("Verify database", "base_dados/verificar_base.py"),
    "3": ("Data quality report", "utilitarios/qualidade_dados.py"),
    "4": ("Local explanation", "analise/explicacao_local.py"),
    "5": ("Global explanation", "analise/explicacao_global.py"),
    "6": ("What-if simulation", "analise/simulacao_what_if.py"),
    "7": ("Manual insert", "utilitarios/inserir_desvio_manual.py"),
    "8": ("Verify inserts", "base_dados/verificar_insercao.py"),
    "9": ("Delete last record", "base_dados/apagar_ultimo_registo.py"),
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
