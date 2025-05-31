import numpy as np # Importa a biblioteca NumPy para operações numéricas eficientes.

class MLP:
    """
    Classe que implementa uma Rede Neural Perceptron de Múltiplas Camadas (MLP)
    com o algoritmo Backpropagation para treinamento.
    """

    def __init__(self, input_neurons, hidden_neurons, output_neurons, learning_rate):
        """
        Construtor da classe MLP.
        Inicializa a arquitetura da rede e os pesos/vieses.

        Args:
            input_neurons (int): Número de neurônios na camada de entrada.
            hidden_neurons (int): Número de neurônios na camada oculta.
            output_neurons (int): Número de neurônios na camada de saída.
            learning_rate (float): Taxa de aprendizado para o ajuste dos pesos.
        """
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate

        # --- Inicialização de Pesos e Vieses (Bias) ---
        # Pesos e vieses são inicializados aleatoriamente com valores pequenos
        # para evitar saturação inicial das funções de ativação.

        # Pesos da camada de entrada para a camada oculta (input_neurons x hidden_neurons)
        self.pesos_entrada_oculta = np.random.uniform(low=-0.1, high=0.1, size=(self.input_neurons, self.hidden_neurons))
        # Vieses da camada oculta (1 x hidden_neurons)
        self.vieses_oculta = np.random.uniform(low=-0.1, high=0.1, size=(1, self.hidden_neurons))

        # Pesos da camada oculta para a camada de saída (hidden_neurons x output_neurons)
        self.pesos_oculta_saida = np.random.uniform(low=-0.1, high=0.1, size=(self.hidden_neurons, self.output_neurons))
        # Vieses da camada de saída (1 x output_neurons)
        self.vieses_saida = np.random.uniform(low=-0.1, high=0.1, size=(1, self.output_neurons))

    @staticmethod
    def sigmoide(x):
        """
        Função de ativação Sigmoide.
        É um método estático porque não depende de nenhuma instância da classe.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivada_sigmoide(x):
        """
        Derivada da função Sigmoide.
        Também é um método estático.
        """
        s = MLP.sigmoide(x) # Chama a função sigmoide através do nome da classe
        return s * (1 - s)

    def feedforward(self, X_input):
        """
        Realiza a propagação para frente (feedforward) através da rede.

        Args:
            X_input (np.array): As entradas para a rede.

        Returns:
            tuple: Uma tupla contendo as saídas da camada oculta (a_oculta)
                   e as saídas da camada de saída (a_saida).
        """
        # 1. Calculando a entrada ponderada para a camada oculta (Z_oculta)
        self.z_oculta = np.dot(X_input, self.pesos_entrada_oculta) + self.vieses_oculta

        # 2. Aplicando a função de ativação na camada oculta (A_oculta)
        self.a_oculta = MLP.sigmoide(self.z_oculta)

        # 3. Calculando a entrada ponderada para a camada de saída (Z_saida)
        self.z_saida = np.dot(self.a_oculta, self.pesos_oculta_saida) + self.vieses_saida

        # 4. Aplicando a função de ativação na camada de saída (A_saida)
        self.a_saida = MLP.sigmoide(self.z_saida)

        return self.a_oculta, self.a_saida

    def backpropagation(self, X_input, y_desejada, a_oculta, a_saida):
        """
        Realiza a propagação para trás (backpropagation) e calcula os gradientes.

        Args:
            X_input (np.array): As entradas originais.
            y_desejada (np.array): As saídas desejadas para as entradas.
            a_oculta (np.array): As saídas da camada oculta (do feedforward).
            a_saida (np.array): As saídas da camada de saída (do feedforward).

        Returns:
            tuple: Uma tupla contendo os gradientes para os pesos e vieses
                   (d_pesos_oculta_saida, d_vieses_saida, d_pesos_entrada_oculta, d_vieses_oculta).
        """
        # 1. Calculando o erro na camada de saída
        erro_saida = y_desejada - a_saida

        # 2. Calculando o delta (gradiente local) da camada de saída
        delta_saida = erro_saida * MLP.derivada_sigmoide(self.z_saida)

        # 3. Calculando o erro na camada oculta (retropropagando o erro da saída)
        erro_oculta = np.dot(delta_saida, self.pesos_oculta_saida.T)

        # 4. Calculando o delta (gradiente local) da camada oculta
        delta_oculta = erro_oculta * MLP.derivada_sigmoide(self.z_oculta)

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