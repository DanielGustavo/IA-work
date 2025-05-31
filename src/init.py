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

# Extrai as entradas (X) e saídas desejadas (y_desejada) da lista training_data
# Usamos list comprehensions para separar as entradas e saídas
# e np.array para convertê-las em arrays NumPy.
X = np.array([item[0] for item in training_data])
y_desejada = np.array([item[1] for item in training_data])

# 2. Configuração da Arquitetura da Rede
input_neurons = X.shape[1]
hidden_neurons = 5         # Experimente diferentes valores (2, 3, 4, 5, 6)
output_neurons = y_desejada.shape[1]
learning_rate = 0.1        # Experimente diferentes valores (0.01, 0.05, 0.5)
epochs = 20000             # Experimente diferentes valores (5000, 10000, 20000, 50000)

# 3. Criação de uma instância da MLP
# Passamos os parâmetros da arquitetura para o construtor da classe.
rede_neural = MLP(input_neurons, hidden_neurons, output_neurons, learning_rate)

# 4. Treinamento da Rede
# Chamamos o método train da instância da rede, passando os dados de treinamento e o número de épocas.
rede_neural.train(X, y_desejada, epochs)

# 5. Teste da Rede Treinada
print("\n--- Testando a Rede Treinada ---")

# Fazemos as previsões usando o método predict da instância da rede.
previsoes_arredondadas = rede_neural.predict(X)

print("\nEntrada (Decimal) | Saída Desejada | Classificação Final | Status")
print("-" * 70)
for i in range(X.shape[0]):
    # Para exibir a entrada decimal original, multiplicamos o valor normalizado por 7.
    # Isso assume que 1.0 corresponde a 7, 0.0 a 0, e os outros são proporcionais.
    entrada_decimal = int(X[i, 0] * 7)
    saida_desejada_str = ''.join(map(str, y_desejada[i]))
    classificacao_final_str = ''.join(map(str, previsoes_arredondadas[i].astype(int)))
    status = "CORRETO" if np.array_equal(previsoes_arredondadas[i], y_desejada[i]) else "ERRADO"

    print(f"{entrada_decimal:<17} | {saida_desejada_str:<14} | {classificacao_final_str:<19} | {status}")

print("-" * 70)
