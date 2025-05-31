from MLP import MLP
import numpy as np

# --- Configuração e Execução do Problema de Classificação de Dígitos ---

# Dados de treinamento conforme especificado no "TrabalhoRNA.pdf"
# "Entrada (Normalizada)", "Saída Desejada" (binária de 3 bits)
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

# Parâmetros da rede neural
input_neurons = 1  # Uma entrada normalizada
hidden_neurons = 5 # Número de neurônios na camada oculta (pode ser ajustado para testes)
output_neurons = 3 # Três saídas binárias
learning_rate = 0.1 # Taxa de aprendizado
epochs = 20000     # Número de épocas de treinamento

# Cria e treina a MLP
mlp = MLP(input_neurons, hidden_neurons, output_neurons, learning_rate)
mlp.train(training_data, epochs)

print("\nTestando a rede treinada:")
# Testa a rede com os dados de treinamento para verificar o desempenho
for inputs, desired_output in training_data:
    inputs_array = np.array(inputs).reshape(1, -1)
    predicted_raw = mlp.forward_pass(inputs_array)
    # Converte a saída raw para binário (arredondando para o inteiro mais próximo)
    predicted_binary = np.round(predicted_raw).astype(int)

    print(f"Entrada: {inputs[0]:.2f}, Desejado: {desired_output}, "
          f"Previsto (raw): [{predicted_raw[0,0]:.2f}, {predicted_raw[0,1]:.2f}, {predicted_raw[0,2]:.2f}], "
          f"Previsto (binário): {predicted_binary[0]}")


