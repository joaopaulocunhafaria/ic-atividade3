# Inteligência Computacional - Atividade 3: RNA e Neuro-Fuzzy

Este projeto tem como objetivo avaliar e comparar o desempenho de algoritmos de Redes Neurais Artificiais (RNA) e Sistemas Neuro-Fuzzy em diferentes conjuntos de dados reais, seguindo uma metodologia experimental rigorosa.

## 🚀 Como Executar

Para rodar todos os experimentos e gerar os resultados (gráficos e métricas), utilize o script de execução:

```bash
chmod +x run.sh
./run.sh
```

### Pré-requisitos
Certifique-se de ter as dependências instaladas:
```bash
pip install -r requirements.txt
```

## 📂 Estrutura do Projeto

- `src/`: Código fonte modularizado.
    - `experimentEngine.py`: Motor principal que gerencia o ciclo experimental (split 60/20/20, busca de parâmetros e repetições).
    - `utils/`: Implementações de modelos customizados (RBF e TSK).
    - `*Analise.py`: Scripts específicos para cada base de dados.
- `data/`: Scripts de carregamento e pré-processamento de dados.
- `output/`: Resultados gerados (CSV de métricas, Boxplots de performance e Matrizes de Confusão).
- `run.sh`: Script de automação.

## 🧪 Metodologia Experimental

Seguindo as diretrizes do Professor Alisson Marques da Silva (CEFET-MG):
- **Divisão de Dados:** 60% para Treino, 20% para Validação (ajuste de hiperparâmetros) e 20% para Teste final.
- **Repetições:** Cada experimento é executado 21 vezes com inicializações aleatórias para garantir validade estatística.
- **Algoritmos Avaliados:**
    1. **MLP (Multi-Layer Perceptron):** RNA com retropropagação.
    2. **RBF (Radial Basis Function):** RNA baseada em funções de base radial e agrupamento.
    3. **TSK (Takagi-Sugeno-Kang):** Sistema Neuro-Fuzzy utilizando clustering para definição de antecedentes.
    4. **TSK Variation:** Variação sistemática do número de clusters e expoente de fuzziness.

## 📊 Resultados e Análise

Os resultados são salvos em `output/<dataset_name>/metricsSummary.csv`, contendo média e desvio padrão para:
- **Classificação:** Acurácia, Precisão, Recall e F1-Score.
- **Regressão:** MSE, RMSE, MAE e R².

Os gráficos de Boxplot gerados permitem visualizar a estabilidade de cada modelo ao longo das 21 execuções, facilitando a análise crítica para o relatório final no padrão IEEE.

---
*Desenvolvido para a disciplina de Inteligência Computacional.*
