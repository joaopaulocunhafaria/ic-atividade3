# Inteligência Computacional - Atividade 3: Redes Neurais e Neuro-Fuzzy

Este projeto realiza uma avaliação comparativa sistemática entre modelos de Redes Neurais Artificiais (MLP e RBF) e Sistemas Neuro-Fuzzy (TSK) em tarefas de regressão e classificação.

## 📋 Objetivo
Avaliar o desempenho, estabilidade e interpretabilidade de diferentes arquiteturas de Inteligência Computacional utilizando 4 conjuntos de dados reais, seguindo uma metodologia experimental rigorosa (21 execuções independentes, busca em grade de hiperparâmetros e testes estatísticos).

## 📊 Conjuntos de Dados
O projeto utiliza os seguintes datasets:
1.  **Ames Housing (Regressão)**: Previsão do preço de venda de casas.
2.  **Auto MPG (Regressão)**: Previsão do consumo de combustível.
3.  **Credit-G (Classificação)**: Classificação de risco de crédito (bom/mau pagador).
4.  **Diabetes (Classificação)**: Diagnóstico de diabetes.

## 🤖 Algoritmos Avaliados
Foram implementados e testados:
-   **MLP (Multi-Layer Perceptron)**: Redes neurais tradicionais com múltiplas camadas ocultas.
-   **RBF (Radial Basis Function)**: Redes com camada oculta baseada em núcleos radiais (K-Means + Ridge).
-   **TSK (Takagi-Sugeno-Kang)**: Sistema fuzzy neuro-adaptativo utilizando Fuzzy C-Means para os antecedentes e modelos lineares para os consequentes.
    -   *Variação 1*: Focado em poucos clusters (regras).
    -   *Variação 2*: Focado em maior granularidade de clusters.

## ⚙️ Metodologia Experimental
-   **Divisão de Dados**: 60% Treino, 20% Validação (para hiperparâmetros), 20% Teste.
-   **Repetições**: 21 execuções independentes com diferentes sementes aleatórias para garantir validade estatística.
-   **Busca de Parâmetros**: Grid Search sistemático utilizando o conjunto de validação.
-   **Análise Estatística**: 
    -   *Wilcoxon Signed-Rank Test*: Comparação pareada dentro de cada dataset.
    -   *Friedman Test*: Comparação global dos algoritmos através de todos os datasets.

## 🚀 Como Executar
Certifique-se de ter as dependências instaladas:
```bash
pip install -r requirements.txt
```

Para rodar todos os experimentos e gerar os resultados:
```bash
chmod +x run.sh
./run.sh
```

## 📁 Estrutura do Projeto
-   `src/`: Código fonte modularizado.
    -   `experimentEngine.py`: Motor principal de execução dos experimentos.
    -   `utils/`: Implementações dos modelos RBF, TSK e geradores de metadados.
-   `output/`: Gráficos, tabelas de métricas e análises estatísticas geradas automaticamente.
-   `data/`: Scripts de carregamento dos conjuntos de dados.

## 📈 Resultados Esperados
Após a execução, a pasta `output/` conterá:
-   `performanceBoxplot.png`: Estabilidade das métricas (Accuracy/R2) ao longo das 21 execuções.
-   `metricsSummary.csv`: Média e desvio padrão de todas as métricas (MSE, RMSE, MAE, R2 ou Accuracy, Precision, Recall, F1).
-   `statisticalTests.csv`: Resultados do teste de Wilcoxon.
-   `finalComparison/`: Matriz global e teste de Friedman.

---
**Professor:** Alisson Marques da Silva  
**Disciplina:** Inteligência Computacional (CEFET-MG)
