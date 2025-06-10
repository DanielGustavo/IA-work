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

    def backpropagation(self, X_input, y_desejada, a_oculta, a_saida):
        # Calculando o erro na camada de saída
        erro_saida = y_desejada - a_saida

        # Calculando o delta (gradiente local) da camada de saída
        delta_saida = erro_saida * MLP.derivada_sigmoide(self.hiddenLayer2)

        # Calculando o erro na camada oculta (retropropagando o erro da saída)
        erro_oculta = np.dot(delta_saida, self.pesos_oculta_saida.T)

        # Calculando o delta (gradiente local) da camada oculta
        delta_oculta = erro_oculta * MLP.derivada_sigmoide(self.hiddenLayer1)

        # Calculando os gradientes para atualização
        d_pesos_oculta_saida = np.dot(a_oculta.T, delta_saida)
        d_vieses_saida = np.sum(delta_saida, axis=0, keepdims=True)
        d_pesos_entrada_oculta = np.dot(X_input.T, delta_oculta)
        d_vieses_oculta = np.sum(delta_oculta, axis=0, keepdims=True)

        return d_pesos_oculta_saida, d_vieses_saida, d_pesos_entrada_oculta, d_vieses_oculta

    def update_weights(self, d_pesos_oculta_saida, d_vieses_saida, d_pesos_entrada_oculta, d_vieses_oculta):
        """
        Atualiza os pesos e vieses da rede usando os gradientes calculados.

        Args:
            d_pesos_oculta_saida (np.array): Gradiente dos pesos da camada oculta para a saída.
            d_vieses_saida (np.array): Gradiente dos vieses da camada de saída.
            d_pesos_entrada_oculta (np.array): Gradiente dos pesos da entrada para a camada oculta.
            d_vieses_oculta (np.array): Gradiente dos vieses da camada oculta.
        """
        self.pesos_oculta_saida += d_pesos_oculta_saida * self.learning_rate
        self.vieses_saida += d_vieses_saida * self.learning_rate
        self.pesos_entrada_oculta += d_pesos_entrada_oculta * self.learning_rate
        self.vieses_oculta += d_vieses_oculta * self.learning_rate

    def train(self, X_train, y_train, epochs):
        """
        Treina a rede neural usando o algoritmo Backpropagation.

        Args:
            X_train (np.array): Dados de entrada para treinamento.
            y_train (np.array): Saídas desejadas para treinamento.
            epochs (int): Número de épocas para o treinamento.
        """
        print("Iniciando o treinamento da MLP com Backpropagation...")
        print(f"Camada de Entrada: {self.input_neurons} neurônio(s)")
        print(f"Camada Oculta: {self.hidden_neurons} neurônio(s)")
        print(f"Camada de Saída: {self.output_neurons} neurônio(s)")
        print(f"Taxa de Aprendizado: {self.learning_rate}")
        print(f"Número de Épocas: {epochs}")
        print("-" * 50)

        for epoch in range(epochs):
            # Realiza o feedforward
            a_oculta, a_saida = self.feedforward(X_train)

            # Realiza o backpropagation e calcula os gradientes
            d_pesos_oculta_saida, d_vieses_saida, d_pesos_entrada_oculta, d_vieses_oculta = \
                self.backpropagation(X_train, y_train, a_oculta, a_saida)

            # Atualiza os pesos e vieses
            self.update_weights(d_pesos_oculta_saida, d_vieses_saida, d_pesos_entrada_oculta, d_vieses_oculta)

            # Monitoramento do Erro
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y_train - a_saida)) # Erro Quadrático Médio (MSE)
                print(f"Época {epoch}, Erro: {loss:.6f}")

        print("-" * 50)
        print("Treinamento concluído!")

    def predict(self, X_test):
        """
        Faz previsões com a rede neural treinada.

        Args:
            X_test (np.array): Dados de entrada para fazer previsões.

        Returns:
            np.array: As saídas arredondadas da rede (classificação final).
        """
        # Realiza o feedforward para as entradas de teste
        _, previsoes_finais = self.feedforward(X_test)
        # Arredonda as saídas para 0 ou 1 para a classificação binária
        return np.round(previsoes_finais)
