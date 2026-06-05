# Projeto de Inteligência Computacional - Atividade 3

Este projeto realiza a análise de quatro bases de dados utilizando Redes Neurais Artificiais (RNA) e Sistemas Neuro-Fuzzy.

## Estrutura do Projeto

- `data/`: Contém os scripts de importação e tratamento de dados.
  - `auto_mpg.py`, `diabetes.py`, `credit_g.py`, `ames_housing.py`: Scripts que buscam os dados e exportam funções de carregamento.
- `src/`: Contém os scripts de análise e experimentos.
  - `diabetes_analise.py`: Script de análise para a base de Diabetes.
- `requirements.txt`: Lista de dependências do Python.

## Como Executar

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute a análise:**
   Você pode rodar os scripts de análise individualmente:
   ```bash
   python src/diabetes_analise.py
   python src/credit_g_analise.py
   python src/auto_mpg_analise.py
   python src/ames_housing_analise.py
   ```

## Metodologia

- **Divisão de Dados:** 60% Treino, 20% Validação, 20% Teste.
- **Repetições:** Cada experimento é repetido 21 vezes com diferentes sementes aleatórias para garantir robustez estatística.
- **Algoritmos:**
  - RNA: MLP com diferentes arquiteturas.
  - Neuro-Fuzzy: Modelos configurados para simular o comportamento de sistemas baseados em regras.

## Resultados Gerados

Os scripts na pasta `src/` geram:
- Gráficos de análise exploratória (ex: Mapas de calor).
- Boxplots de desempenho (Acurácia/Erro).
- Matrizes de confusão (para classificação).
- Tabelas com Média e Desvio-Padrão das métricas.
- Resultados de testes estatísticos (t-test).
