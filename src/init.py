from MLP import MLP
import numpy as np

training_data = [
    ([0.0], [0, 0, 0]),
    ([0.14], [0, 0, 1]),
    ([0.28], [0, 1, 0]),
    ([0.42], [0, 1, 1]),
    ([0.57], [1, 0, 0]),
    ([0.71], [1, 0, 1]),
    ([0.85], [1, 1, 0]),
    ([1.0], [1, 1, 1])
]

X = np.array([item[0] for item in training_data])
y_desejada = np.array([item[1] for item in training_data])

# Configuração da Arquitetura da Rede
input_neurons = X.shape[1]
hidden_neurons = 5
output_neurons = y_desejada.shape[1]
learning_rate = 0.1
epochs = 50000

# Treinamento da Rede
# Chamamos o método train da instância da rede, passando os dados de treinamento e o número de épocas.
rede_neural = MLP(input_neurons, hidden_neurons, output_neurons, learning_rate)
 
# Etapa 2
rede_neural.train(X, y_desejada, epochs)

# Teste da Rede Treinada
print("\n--- Testando a Rede Treinada ---")

previsoes_arredondadas = rede_neural.predict(X)

print("\nEntrada (Decimal) | Saída Desejada | Classificação Final | Status")
print("-" * 70)
for i in range(X.shape[0]):
    entrada_decimal = int(X[i, 0] * 7)
    saida_desejada_str = ''.join(map(str, y_desejada[i]))
    classificacao_final_str = ''.join(map(str, previsoes_arredondadas[i].astype(int)))
    status = "CORRETO" if np.array_equal(previsoes_arredondadas[i], y_desejada[i]) else "ERRADO"

    print(f"{entrada_decimal:<17} | {saida_desejada_str:<14} | {classificacao_final_str:<19} | {status}")

print("-" * 70)
