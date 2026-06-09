# Atividade RNA e NF - Inteligência Computacional

Este projeto realiza uma avaliação comparativa entre Redes Neurais Artificiais (RNA) e Sistemas Neuro-Fuzzy (NF) em tarefas de classificação e regressão, utilizando 4 conjuntos de dados reais.

## Estrutura do Projeto

- `data/`: Scripts de carregamento e pré-processamento de dados.
- `src/`: Código fonte modularizado.
    - `utils/fuzzyModels.py`: Implementação do modelo Neuro-Fuzzy TSK (Takagi-Sugeno-Kang).
    - `experimentEngine.py`: Motor de experimentos centralizado (60/20/20 split, 21 repetições, métricas).
    - `*Analise.py`: Scripts específicos para cada dataset.
- `output/`: Resultados gerados (Tabelas CSV, Boxplots, Matrizes de Confusão).
- `run.sh`: Script de execução automatizada.

## Metodologia

Conforme exigido pelo enunciado:
- **Particionamento:** 60% Treino, 20% Validação (ajuste de hiperparâmetros), 20% Teste.
- **Repetições:** 21 execuções independentes para cada modelo.
- **Análise Estatística:** Média e desvio padrão das métricas.
- **Modelos:**
    - RNA: Multi-Layer Perceptron (Simples e Profunda).
    - Neuro-Fuzzy: TSK com clustering Fuzzy C-Means.

## Como Executar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute todos os experimentos:
   ```bash
   ./run.sh
   ```

## Resultados Esperados

Após a execução, a pasta `output/` conterá:
- `metricsSummary.csv`: Resumo das métricas (Média/DP).
- `performanceBoxplot.png`: Comparação visual da métrica principal.
- `confusionMatrix.png`: Matriz de confusão para o melhor modelo (classificação).
- `edaCorrelation.png`: Mapa de calor de correlação dos dados.

---
Trabalho desenvolvido para a disciplina de Inteligência Computacional - CEFET-MG.
