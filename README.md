ML-Sal

Explainable analysis of salt deviations in industrial processes

📌 Project objective

This project aims to analyze and explain salt deviations (salt %) in an industrial process using explainable Machine Learning, without changing the database and while respecting physical process assumptions.

The system was designed to answer the question:

“Given an observed salt deviation, which measured variables contributed most to that deviation in this specific case?”

⸻

🧠 Core principles

The project is based on four key principles:

1️⃣ Process before model

Before any ML analysis, the system validates whether the sign of the observed deviation (dif_pct_sal) is physically coherent with known process assumptions (e.g., ES and HFD).

Incoherent cases are automatically classified as INCONCLUSIVE.

⸻

2️⃣ The model does not change data
	•	The database is never modified during analyses
	•	The user only consults and simulates
	•	The system is safe to share

⸻

3️⃣ Local explanation, not global

The focus is always:
	•	a batch
	•	a specific record
	•	a local explanation

No global inferences or automatic generalizations are made.

⸻

4️⃣ Human explainability

Results are presented in a way that is:
	•	visual
	•	directional (pushes vs compensates)
	•	with real process values
	•	understandable for people without an ML background

⸻

🗂️ Project structure
ML-Sal/
│
├── .gitignore
├── README.md
│
├── dados/
│   ├── matriz_cloretos.csv        # Original historical source (optional)
│   └── ml_sal.db                  # SQLite database (consolidated history)
│
├── base_dados/
│   ├── criar_tabela.py            # Create SQLite table
│   ├── importar_historico.py      # Initial history import
│   ├── verificar_base.py          # Database integrity check
│   ├── verificar_insercao.py      # Insert confirmation
│   └── apagar_ultimo_registo.py   # Controlled removal of the last record
│
├── ml/
│   ├── preparar_dados_ml.py       # Prepare data for ML
│   └── treinar_modelo_baseline.py # Train baseline model
│
├── analise/
│   ├── explicacao_local.py        # ⭐ Explainable local analysis (main)
│   ├── explicacao_global.py       # Global analysis (exploratory)
│   └── simulacao_what_if.py       # Counterfactual simulations (what-if)
│
├── utilitarios/
│   ├── inserir_desvio_manual.py   # Robust manual insertion (validated inputs)
│   └── listar_variaveis.py        # List available variables
│
├── outputs/
│   ├── shap_local.png             # Local explanation chart
│   └── shap_global.png            # Global explanation chart
│
├── docs/
│   └── ordem_execucao.txt         # Recommended execution order
│
└── subir_github.sh                # Script for commit & push to GitHub


⸻

⚙️ Technologies used
	•	Python 3
	•	SQLite
	•	pandas
	•	scikit-learn
	•	SHAP
	•	matplotlib

⸻

🔍 Recommended usage flow

1️⃣ Prepare the database (once)

python criar_tabela.py
python importar_historico.py
python verificar_base.py
⸻

2️⃣ Analyze a specific deviation (normal use)
python explicacao_local.py
The user:
	•	enters the batch
	•	chooses the record (if needed)
	•	receives:
	•	physical validation
	•	explainable chart (shap_local.png)
	•	text interpretation

⸻

3️⃣ “What-if” simulations (optional)
python simulacao_what_if.py
Allows testing hypothetical scenarios without saving data.

⸻

📊 Local explanation chart structure

The generated chart (shap_local.png) follows these rules:
	•	🔴 Red → variable that pushes the deviation
	•	🟢 Green → variable that compensates the deviation
	•	Bar length → magnitude of local impact
	•	Percentage → relative weight in the analyzed case
	•	Measured value → shown on the Y axis
	•	Labels:
	•	≥ 15% → inside the bar
	•	< 15% → near the central axis (0)

The chart does not represent absolute causality, only local model sensitivity.

⸻

🚫 Inconclusive cases

The analysis is automatically classified as INCONCLUSIVE when:
	•	the sign of the observed deviation (dif_pct_sal)
	•	contradicts physical process assumptions
	•	and the variable involved is dominant in the case

In these cases:
	•	❌ no chart is generated
	•	✅ a clear textual explanation is provided

⸻

🎓 Academic context

This project was developed with a focus on:
	•	explainability
	•	conceptual robustness
	•	integration between process knowledge and ML
	•	transparency in decision-making

It is suitable for:
	•	academic work
	•	engineering projects
	•	demonstrations of explainable ML applied to industry

⸻

📌 Known limitations
	•	The model is local, not causal
	•	Unmeasured variables may explain part of the deviation
	•	Results should be interpreted with process knowledge

⸻

📄 License

Project for educational and academic purposes.
No identification of specific industrial context.

⸻
