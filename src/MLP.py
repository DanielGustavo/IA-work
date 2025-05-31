import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate

        # Inicializa os pesos sinápticos e biases com valores aleatórios entre -1 e 1.
        # Pesos da camada de entrada para a camada oculta (W_ji^o nos slides)
        self.weights_ih = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias_h = np.random.uniform(-1, 1, (1, hidden_size))

        # Pesos da camada oculta para a camada de saída (W_kj^s nos slides)
        self.weights_ho = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_o = np.random.uniform(-1, 1, (1, output_size))

    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return 1

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        # f'(x) = 1 - (f(x))^2
        return 1 - (np.tanh(x))**2

    def forward_pass(self, inputs):
        # Etapa 3: Calcula os nets dos neurônios da camada oculta
        # net_j^o = sum(x_i * w_j,i^o) + b_j^o
        self.hidden_net = np.dot(inputs, self.weights_ih) + self.bias_h
        # Etapa 4: Aplica a função de transferência para obter as saídas da camada oculta
        self.hidden_output = self.tanh(self.hidden_net) # Usando tanh como função não linear

        # Etapa 5: Calcula os nets dos neurônios da camada de saída
        # net_k^s = sum(i_j * w_k,j^s) + b_k^s
        self.output_net = np.dot(self.hidden_output, self.weights_ho) + self.bias_o
        # Etapa 6: Calcula as saídas Ok dos neurônios da camada de saída
        self.predicted_output = self.linear(self.output_net) # Usando linear para a saída

        return self.predicted_output

    def backward_pass(self, inputs, desired_output):
        # Etapa 7: Calcula os erros para os neurônios da camada de saída
        # delta_k^s = (d_k - o_k) * f_k^s'(net_k^s)
        output_error = desired_output - self.predicted_output
        # Para a função linear, a derivada é 1, então (d_k - o_k) * 1
        output_delta = output_error * self.linear_derivative(self.output_net)

        # Etapa 8: Calcula os erros nos neurônios da camada oculta
        # delta_j^o = f_j^o'(net_j^o) * sum(delta_k^s * w_k,j)
        hidden_error = np.dot(output_delta, self.weights_ho.T)
        hidden_delta = hidden_error * self.tanh_derivative(self.hidden_net)

        # Etapa 9: Atualiza os pesos da camada de saída
        # w_kj^s = w_kj^s + (n) * (delta_k^s) * (X_j)
        self.weights_ho += self.learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.bias_o += self.learning_rate * output_delta # Atualiza o bias da saída

        # Etapa 10: Atualiza os pesos da camada oculta
        # w_ji^o = w_ji^o + (n) * (delta_j^h) * (x_i)
        self.weights_ih += self.learning_rate * np.dot(inputs.T, hidden_delta)
        self.bias_h += self.learning_rate * hidden_delta # Atualiza o bias da camada oculta

        # Etapa 11: Calcula o erro da rede (para monitoramento)
        E_r = 0.5 * np.sum(output_error**2)
        return E_r

    def train(self, training_data, epochs, error_threshold=0.001):
        print("Iniciando o treinamento da rede...")
        for epoch in range(epochs):
            total_epoch_error = 0
            for inputs, desired_output in training_data:
                # Garante que as entradas e saídas desejadas tenham a forma correta
                inputs = np.array(inputs).reshape(1, -1)
                desired_output = np.array(desired_output).reshape(1, -1)

                self.forward_pass(inputs)
                total_epoch_error += self.backward_pass(inputs, desired_output)

            # Verifica se o erro total da época está abaixo do limiar
            if total_epoch_error < error_threshold:
                print(f"Treinamento parado na época {epoch+1} devido ao limiar de erro ({error_threshold:.4f}) atingido.")
                break
            if (epoch + 1) % 1000 == 0: # Imprime o erro a cada 1000 épocas
                print(f"Época {epoch+1}, Erro Total: {total_epoch_error:.6f}")
        print("Treinamento concluído.")

