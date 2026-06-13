import os
class DatasetInfoGenerator:

    def generateDatasetInfo(dataFrame, taskType, outputDir, description="", targetColumn=None):
        """
        Gera um arquivo de texto descritivo com a análise exploratória estruturada do dataset.
        """
        infoPath = os.path.join(outputDir, "dataset_info.txt")
        numSamples = dataFrame.shape[0]
        
        # If targetColumn is not provided, assume the last one
        if targetColumn is None:
            targetColumn = dataFrame.columns[-1]
            
        numAttributes = dataFrame.shape[1] - 1
        
        with open(infoPath, "w", encoding="utf-8") as file:
            file.write("==================================================\n")
            file.write(f"ANÁLISE EXPLORATÓRIA BÁSICA: {os.path.basename(outputDir).upper()}\n")
            file.write("==================================================\n\n")
            file.write(f"Descrição do Problema: {description}\n\n")
            file.write(f"Tipo de Tarefa: {taskType.capitalize()}\n")
            file.write(f"Número de Amostras: {numSamples}\n")
            file.write(f"Número de Atributos: {numAttributes}\n")
            file.write(f"Variável de Saída (Target): {targetColumn}\n\n")
            
            file.write("Distribuição / Resumo Estatístico da Saída:\n")
            if taskType == "classification":
                classDistribution = dataFrame[targetColumn].value_counts()
                for className, count in classDistribution.items():
                    percentage = (count / numSamples) * 100
                    file.write(f"  Classe '{className}': {count} amostras ({percentage:.2f}%)\n")
            else:
                targetStats = dataFrame[targetColumn].describe()
                file.write(f"  Mínimo: {targetStats['min']:.4f}\n")
                file.write(f"  Máximo: {targetStats['max']:.4f}\n")
                file.write(f"  Média: {targetStats['mean']:.4f}\n")
                file.write(f"  Desvio Padrão: {targetStats['std']:.4f}\n")
                
            file.write("\nIntegridade dos Dados:\n")
            missingValues = dataFrame.isnull().sum().sum()
            file.write(f"  Total de valores ausentes/nulos: {missingValues}\n")