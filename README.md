ML-Sal

Análise explicável de desvios de sal em processos industriais

📌 Objetivo do projeto

Este projeto tem como objetivo analisar e explicar desvios de sal (% sal) num processo industrial, utilizando Machine Learning explicável, sem alterar a base de dados e respeitando pressupostos físicos do processo.

O sistema foi desenhado para responder à pergunta:

“Dado um desvio de sal observado, quais as variáveis medidas que mais contribuíram para esse desvio neste caso específico?”

⸻

🧠 Princípios fundamentais

O projeto assenta em quatro princípios-chave:

1️⃣ O processo vem antes do modelo

Antes de qualquer análise de ML, o sistema valida se o sinal do desvio observado (dif_pct_sal) é fisicamente coerente com pressupostos conhecidos do processo (ex.: ES e HFD).

Casos incoerentes são automaticamente classificados como INCONCLUSIVOS.

⸻

2️⃣ O modelo não altera dados
	•	A base de dados nunca é modificada durante análises
	•	O utilizador apenas consulta e simula
	•	O sistema é seguro para partilha

⸻

3️⃣ Explicação local, não global

O foco é sempre:
	•	um lote
	•	um registo específico
	•	uma explicação local

Não são feitas inferências globais ou generalizações automáticas.

⸻

4️⃣ Explicabilidade humana

Os resultados são apresentados de forma:
	•	visual
	•	direcional (empurra vs compensa)
	•	com valores reais do processo
	•	compreensíveis para pessoas sem background em ML

⸻

🗂️ Estrutura do projeto
ML-Sal/
│
├── .gitignore
├── README.md
│
├── dados/
│   ├── matriz_cloretos.csv        # Fonte histórica original (opcional)
│   └── ml_sal.db                  # Base de dados SQLite (histórico consolidado)
│
├── base_dados/
│   ├── criar_tabela.py            # Criação da tabela SQLite
│   ├── importar_historico.py      # Importação inicial do histórico
│   ├── verificar_base.py          # Verificação da integridade da base
│   ├── verificar_insercao.py      # Confirmação de inserções
│   └── apagar_ultimo_registo.py   # Remoção controlada do último registo
│
├── ml/
│   ├── preparar_dados_ml.py       # Preparação dos dados para ML
│   └── treinar_modelo_baseline.py # Treino do modelo base
│
├── analise/
│   ├── explicacao_local.py        # ⭐ Análise local explicável (principal)
│   ├── explicacao_global.py       # Análise global (exploratória)
│   └── simulacao_what_if.py       # Simulações contrafactuais (what-if)
│
├── utilitarios/
│   ├── inserir_desvio_manual.py   # Inserção manual robusta (inputs validados)
│   └── listar_variaveis.py        # Listagem das variáveis disponíveis
│
├── outputs/
│   ├── shap_local.png             # Gráfico de explicação local
│   └── shap_global.png            # Gráfico de explicação global
│
├── docs/
│   └── ordem_execucao.txt         # Ordem recomendada de execução
│
└── subir_github.sh                # Script para commit & push para GitHub


⸻

⚙️ Tecnologias utilizadas
	•	Python 3
	•	SQLite
	•	pandas
	•	scikit-learn
	•	SHAP
	•	matplotlib

⸻

🔍 Fluxo de utilização recomendado

1️⃣ Preparar a base de dados (uma vez)

python criar_tabela.py
python importar_historico.py
python verificar_base.py
⸻

2️⃣ Analisar um desvio específico (uso normal)
python explicacao_local.py
O utilizador:
	•	introduz o lote
	•	escolhe o registo (se necessário)
	•	recebe:
	•	validação física
	•	gráfico explicável (shap_local.png)
	•	interpretação textual

⸻

3️⃣ Simulações “what-if” (opcional)
python simulacao_what_if.py
Permite testar cenários hipotéticos sem guardar dados.

⸻

📊 Estrutura do gráfico de explicação local

O gráfico gerado (shap_local.png) segue estas regras:
	•	🔴 Vermelho → variável que empurra o desvio
	•	🟢 Verde → variável que compensa o desvio
	•	Comprimento da barra → magnitude do impacto local
	•	Percentagem → peso relativo no caso analisado
	•	Valor medido → apresentado no eixo Y
	•	Legendas:
	•	≥ 15% → dentro da barra
	•	< 15% → junto ao eixo central (0)

O gráfico não representa causalidade absoluta, apenas sensibilidade local do modelo.

⸻

🚫 Casos inconclusivos

A análise é automaticamente classificada como INCONCLUSIVA quando:
	•	O sinal do desvio observado (dif_pct_sal)
	•	contradiz pressupostos físicos do processo
	•	e a variável em causa é dominante no caso

Nestes casos:
	•	❌ nenhum gráfico é gerado
	•	✅ é apresentada uma explicação textual clara

⸻

🎓 Contexto académico

Este projeto foi desenvolvido com enfoque em:
	•	explicabilidade
	•	robustez conceptual
	•	integração entre conhecimento de processo e ML
	•	transparência na tomada de decisão

É adequado para:
	•	trabalhos académicos
	•	projetos de engenharia
	•	demonstrações de ML explicável aplicado à indústria

⸻

📌 Limitações conhecidas
	•	O modelo é local, não causal
	•	Variáveis não medidas podem explicar parte do desvio
	•	Resultados devem ser interpretados com conhecimento de processo

⸻

📄 Licença

Projeto para fins educativos e académicos.
Sem identificação de contexto industrial específico.

⸻
