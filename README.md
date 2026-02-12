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

🗂️ Project structure (current flat layout)

ML-Sal/
├── .gitignore
├── README.md
├── apagar_ultimo_registo.py
├── criar_tabela.py
├── explicacao_global.py
├── explicacao_local.py
├── importar_historico.py
├── inserir_desvio_manual.py
├── listar_variaveis.py
├── matriz_cloretos.csv
├── menu.py
├── ml_sal.db
├── ml_utils.py
├── ordem_execuao.txt
├── preparar_dados_ml.py
├── qualidade_dados.py
├── shap_global.png
├── shap_local.png
├── simulacao_what_if.py
├── subir_github.sh
├── subir_github_branch.sh
├── treinar_modelo_baseline.py
├── verificar_base.py
└── verificar_insercao.py

models/
└── rf_model.joblib


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

Option A — Use the menu

python menu.py

Option B — Run steps manually

1️⃣ Prepare the database (once)

python criar_tabela.py
python importar_historico.py
python verificar_base.py

⸻

2️⃣ Run a data quality report

python qualidade_dados.py

⸻

3️⃣ Analyze a specific deviation (normal use)

python explicacao_local.py

The user:
	•	enters the batch
	•	chooses the record (if needed)
	•	receives:
	•	physical validation
	•	explainable chart (shap_local.png)
	•	text interpretation

⸻

4️⃣ “What-if” simulations (optional)

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

🧠 Model cache

Local/global explanations and what-if simulations reuse a cached model to avoid retraining on every run. If you want to force a retrain, run with:

RETRAIN_MODEL=1 python explicacao_local.py

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
