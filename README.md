# ML-Sal  
Análise do desvio de sal em processos industriais através de Machine Learning explicável

## 1. Enquadramento
O controlo do teor de sal em processos de salga é um fator crítico para a conformidade do produto e para a estabilidade do processo industrial. No entanto, o desvio do teor de sal relativamente às especificações pode resultar da interação de múltiplas variáveis de processo, nem sempre facilmente identificáveis por métodos tradicionais.

Neste contexto, o presente projeto tem como objetivo aplicar técnicas de **Machine Learning explicável** para analisar o **desvio percentual de sal**, explorando padrões históricos do processo e apoiando a compreensão das variáveis mais relevantes associadas a esses desvios.

---

## 2. Objetivos
Os principais objetivos do projeto são:
- modelar a relação entre condições operacionais e o desvio percentual de sal;
- identificar, de forma global e local, as variáveis mais influentes no desvio observado;
- permitir a análise de casos específicos (lotes);
- realizar simulações do tipo *what-if* para avaliar cenários alternativos;
- apoiar a tomada de decisão em contexto de melhoria contínua.

O foco do trabalho é **explicativo e analítico**, e não a previsão em tempo real.

---

## 3. Formulação do problema
O problema foi formulado como um **problema de regressão supervisionada**, onde:
- a variável dependente é o desvio percentual de sal (`dif_pct_sal`);
- as variáveis independentes incluem parâmetros de processo (pH, temperatura, densidade, tempos fora de especificação), diferenças operacionais e a referência do produto.

Os dados utilizados correspondem a registos históricos do processo.

---

## 4. Modelo de Machine Learning
Foi utilizado um **Random Forest Regressor**, um modelo baseado em *ensemble learning* que combina múltiplas árvores de decisão independentes.

Este modelo foi escolhido por apresentar:
- capacidade de modelar relações não lineares;
- robustez a ruído e a dados incompletos;
- boa performance em conjuntos de dados industriais;
- compatibilidade com métodos de interpretação local e global.

A previsão final resulta da média das previsões das árvores individuais, reduzindo o risco de sobreajuste.

---

## 5. Interpretação do modelo
Para garantir transparência e interpretabilidade, foram utilizadas técnicas de **SHAP (SHapley Additive exPlanations)**, permitindo:
- análise global da importância das variáveis ao longo do conjunto de dados;
- análise local para explicar desvios em casos específicos.

Estas explicações permitem compreender quais variáveis contribuem positiva ou negativamente para o desvio previsto, em cada contexto.

---

## 6. Simulações “what-if”
O modelo é utilizado para realizar simulações contrafactuais, mantendo todas as variáveis constantes e alterando apenas uma ou mais condições de processo.

Estas simulações permitem:
- avaliar a sensibilidade do desvio a alterações específicas;
- comparar cenários alternativos;
- distinguir variáveis explicativas de variáveis que constituem potenciais alavancas de atuação.

Os resultados devem ser interpretados como **evidência estatística baseada em dados históricos**, e não como causalidade física direta.

---

## 7. Estrutura do projeto
ML-Sal/
│
├── importar_historico.py        # Importação inicial de dados históricos
├── criar_tabela.py              # Criação da estrutura da base de dados
├── inserir_desvio_manual.py     # Inserção manual de novos desvios
├── verificar_base.py            # Verificação da base de dados
│
├── preparar_dados_ml.py         # Preparação e limpeza dos dados
├── treinar_modelo_baseline.py   # Treino do modelo de Random Forest
│
├── explicacao_global.py         # Análise explicativa global (SHAP)
├── explicacao_local.py          # Análise explicativa local
│
├── simulacao_what_if.py         # Simulações interativas de cenários
├── listar_variaveis.py          # Listagem das variáveis disponíveis
│
├── ordem_execuao.txt            # Ordem de execução dos scripts
├── .gitignore                   # Exclusão de dados e outputs
└── README.md                    # Documentação do projeto

> Nota: Os dados reais não são versionados no repositório por motivos de confidencialidade.

---

## 8. Fluxo de utilização
O fluxo típico de utilização do projeto é:
1. inserção de novos desvios na base de dados;
2. preparação dos dados;
3. treino do modelo com o histórico atualizado;
4. análise explicativa global ou local, conforme necessário;
5. realização de simulações *what-if* para apoio à decisão.

A sequência detalhada encontra-se descrita no ficheiro `ordem_execuao.txt`.

---

## 9. Limitações
- O modelo não estabelece relações causais físicas;
- os resultados são válidos apenas dentro do domínio dos dados históricos;
- o modelo não substitui conhecimento de processo nem ferramentas clássicas de controlo estatístico.

---

## 10. Contexto académico
Este projeto foi desenvolvido no âmbito de um **mestrado**, integrando conceitos de ciência de dados, machine learning explicável e melhoria contínua de processos industriais.

---

## Autor
Carlos Ortiz