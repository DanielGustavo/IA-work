import numpy as np


class MLP:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, learning_rate):
        # Etapa 1
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate

        self.pesos_entrada_oculta = np.random.uniform(low=-0.1, high=0.1, size=(self.input_neurons, self.hidden_neurons))
        self.vieses_oculta = np.random.uniform(low=-0.1, high=0.1, size=(1, self.hidden_neurons))

        self.pesos_oculta_saida = np.random.uniform(low=-0.1, high=0.1, size=(self.hidden_neurons, self.output_neurons))
        self.vieses_saida = np.random.uniform(low=-0.1, high=0.1, size=(1, self.output_neurons))

    def sigmoide(x):
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoide(x):
        s = MLP.sigmoide(x)
        return s * (1 - s)

    def feedforward(self, X_input):
        # Calculando a entrada ponderada para a camada oculta (Z_oculta)
        # Etapa 3
        self.hiddenLayer1 = np.dot(X_input, self.pesos_entrada_oculta) + self.vieses_oculta

        # Aplicando a função de ativação na camada oculta (A_oculta)
        # Etapa 4
        self.outputHiddenLayer1 = MLP.sigmoide(self.hiddenLayer1)

        # Calculando a entrada ponderada para a camada de saída (Z_saida)
        # Etapa 5
        self.hiddenLayer2 = np.dot(self.outputHiddenLayer1, self.pesos_oculta_saida) + self.vieses_saida

        # Aplicando a função de ativação na camada de saída (A_saida)
        # Etapa 6
        self.outputHiddenLayer2 = MLP.sigmoide(self.hiddenLayer2)

        return self.outputHiddenLayer1, self.outputHiddenLayer2

    def backpropagation(self, X_input, y_desejada, outputHiddenLayer1, outputHiddenLayer2):
        # Etapa 7
        erro_saida = y_desejada - outputHiddenLayer2 # Calculando o erro na camada de saída
        delta_saida = erro_saida * MLP.derivada_sigmoide(self.hiddenLayer2) # Calculando o delta (gradiente local) da camada de saída

        # Etapa 8
        delta_oculta = MLP.derivada_sigmoide(self.hiddenLayer1) * np.dot(delta_saida, self.pesos_oculta_saida.T)

        # Calculando os gradientes para atualização (Etapa 9)
        d_pesos_oculta_saida = np.dot(outputHiddenLayer1.T, delta_saida)
        d_vieses_saida = np.sum(delta_saida, axis=0, keepdims=True)

        # Calculando os gradientes para atualização (Etapa 10)
        d_pesos_entrada_oculta = np.dot(X_input.T, delta_oculta)
        d_vieses_oculta = np.sum(delta_oculta, axis=0, keepdims=True)

        return d_pesos_oculta_saida, d_vieses_saida, d_pesos_entrada_oculta, d_vieses_oculta

    def update_weights(self, d_pesos_oculta_saida, d_vieses_saida, d_pesos_entrada_oculta, d_vieses_oculta):
        # Etapa 9
        self.pesos_oculta_saida += d_pesos_oculta_saida * self.learning_rate
        self.vieses_saida += d_vieses_saida * self.learning_rate

        # Etapa 10
        self.pesos_entrada_oculta += d_pesos_entrada_oculta * self.learning_rate
        self.vieses_oculta += d_vieses_oculta * self.learning_rate

    def train(self, X_train, y_train, epochs):
        print("Iniciando o treinamento da MLP com Backpropagation...")
        print(f"Camada de Entrada: {self.input_neurons} neurônio(s)")
        print(f"Camada Oculta: {self.hidden_neurons} neurônio(s)")
        print(f"Camada de Saída: {self.output_neurons} neurônio(s)")
        print(f"Taxa de Aprendizado: {self.learning_rate}")
        print(f"Número de Épocas: {epochs}")
        print("-" * 50)

        for epoch in range(epochs):
            outputHiddenLayer1, outputHiddenLayer2 = self.feedforward(X_train)

            d_pesos_oculta_saida, d_vieses_saida, d_pesos_entrada_oculta, d_vieses_oculta = \
                self.backpropagation(X_train, y_train, outputHiddenLayer1, outputHiddenLayer2)

            self.update_weights(d_pesos_oculta_saida, d_vieses_saida, d_pesos_entrada_oculta, d_vieses_oculta)

            # Monitoramento do Erro
            if epoch % 1000 == 0:
                loss = 0.5 * np.sum(np.square((y_train - outputHiddenLayer2)))
                print(f"Época {epoch}, Erro: {loss}")

        print("-" * 50)
        print("Treinamento concluído!")

    def predict(self, X_test):
        # Realiza o feedforward para as entradas de teste
        _, previsoes_finais = self.feedforward(X_test)
        # Arredonda as saídas para 0 ou 1 para a classificação binária
        return np.round(previsoes_finais)
