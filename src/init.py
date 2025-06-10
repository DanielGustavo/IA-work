from MLP import MLP
import numpy as np

trainingData = [
    ([0.0], [0, 0, 0]),
    ([0.14], [0, 0, 1]),
    ([0.28], [0, 1, 0]),
    ([0.42], [0, 1, 1]),
    ([0.57], [1, 0, 0]),
    ([0.71], [1, 0, 1]),
    ([0.85], [1, 1, 0]),
    ([1.0], [1, 1, 1])
]

inputX = np.array([item[0] for item in trainingData])
expectedY = np.array([item[1] for item in trainingData])

# Configuração da Arquitetura da Rede
inputNeurons = inputX.shape[1]
hiddenNeurons = 5
outputNeurons = expectedY.shape[1]
learningRate = 0.2
epochs = 20000

# Treinamento da Rede
neuralNet = MLP(inputNeurons, hiddenNeurons, outputNeurons, learningRate)
         
print("Iniciando o treinamento da MLP com Backpropagation...")
print(f"Camada de Entrada: {inputNeurons} neurônio(s)")
print(f"Camada Oculta: {hiddenNeurons} neurônio(s)")
print(f"Camada de Saída: {outputNeurons} neurônio(s)")
print(f"Taxa de Aprendizado: {learningRate}")
print(f"Número de Épocas: {epochs}")
print("-" * 50)
 
neuralNet.train(inputX, expectedY, epochs)
         
print("-" * 50)
print("Treinamento concluído!")

# Teste da Rede Treinada
print("\n--- Testando a Rede Treinada ---")

previsions = neuralNet.predict(inputX)

print("\nEntrada (Decimal) | Saída Desejada | Classificação Final | Status")
print("-" * 70)
 
for i in range(inputX.shape[0]):
    decimalInput = i
    expectedYString = ''.join(map(str, expectedY[i]))
    previsionString = ''.join(map(str, previsions[i].astype(int)))
     
    status = ""
    if np.array_equal(previsions[i], expectedY[i]):
        status = "CORRETO"
    else:
        status = "ERRADO"

    print(f"{decimalInput:<17} | {expectedYString:<14} | {previsionString:<19} | {status}")

print("-" * 70)
