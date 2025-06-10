import numpy as np

class MLP:
    def __init__(self, inputNeurons, hiddenNeurons, outputNeurons, learningRate):
        # Etapa 1
        self.inputNeurons = inputNeurons
        self.hiddenNeurons = hiddenNeurons
        self.outputNeurons = outputNeurons
        self.learningRate = learningRate

        self.hiddenInputWeights = np.random.uniform(low=-0.1, high=0.1, size=(self.inputNeurons, self.hiddenNeurons))
        self.hiddenBias = np.random.uniform(low=-0.1, high=0.1, size=(1, self.hiddenNeurons))

        self.hiddenOutputWeights = np.random.uniform(low=-0.1, high=0.1, size=(self.hiddenNeurons, self.outputNeurons))
        self.outputBias = np.random.uniform(low=-0.1, high=0.1, size=(1, self.outputNeurons))

    def sigmoide(x):
        return 1 / (1 + np.exp(-x))

    def sigmoideDerivate(x):
        s = MLP.sigmoide(x)
        return s * (1 - s)

    def feedforward(self, inputX):
        # Etapa 3
        self.hiddenLayer1 = np.dot(inputX, self.hiddenInputWeights) + self.hiddenBias

        # Etapa 4
        self.outputHiddenLayer1 = MLP.sigmoide(self.hiddenLayer1)

        # Etapa 5
        self.hiddenLayer2 = np.dot(self.outputHiddenLayer1, self.hiddenOutputWeights) + self.outputBias

        # Etapa 6
        self.outputHiddenLayer2 = MLP.sigmoide(self.hiddenLayer2)

        return self.outputHiddenLayer1, self.outputHiddenLayer2

    def backPropagation(self, inputX, expectedY, outputHiddenLayer1, outputHiddenLayer2):
        # Etapa 7
        outputDelta = (expectedY - outputHiddenLayer2) * MLP.sigmoideDerivate(self.hiddenLayer2)

        # Etapa 8
        hiddenDelta = MLP.sigmoideDerivate(self.hiddenLayer1) * np.dot(outputDelta, self.hiddenOutputWeights.T)

        # Etapa 9
        d_pesos_oculta_saida = np.dot(outputHiddenLayer1.T, outputDelta)
        d_vieses_saida = np.sum(outputDelta, axis=0, keepdims=True)

        # Etapa 10
        d_pesos_entrada_oculta = np.dot(inputX.T, hiddenDelta)
        d_vieses_oculta = np.sum(hiddenDelta, axis=0, keepdims=True)

        return d_pesos_oculta_saida, d_vieses_saida, d_pesos_entrada_oculta, d_vieses_oculta

    def update_weights(self, d_pesos_oculta_saida, d_vieses_saida, d_pesos_entrada_oculta, d_vieses_oculta):
        # Etapa 9
        self.hiddenOutputWeights += d_pesos_oculta_saida * self.learningRate
        self.outputBias += d_vieses_saida * self.learningRate

        # Etapa 10
        self.hiddenInputWeights += d_pesos_entrada_oculta * self.learningRate
        self.hiddenBias += d_vieses_oculta * self.learningRate

    def train(self, X_train, y_train, epochs):
        print("Iniciando o treinamento da MLP com Backpropagation...")
        print(f"Camada de Entrada: {self.inputNeurons} neurônio(s)")
        print(f"Camada Oculta: {self.hiddenNeurons} neurônio(s)")
        print(f"Camada de Saída: {self.outputNeurons} neurônio(s)")
        print(f"Taxa de Aprendizado: {self.learningRate}")
        print(f"Número de Épocas: {epochs}")
        print("-" * 50)

        for epoch in range(epochs):
            outputHiddenLayer1, outputHiddenLayer2 = self.feedforward(X_train)

            d_pesos_oculta_saida, d_vieses_saida, d_pesos_entrada_oculta, d_vieses_oculta = \
                self.backPropagation(X_train, y_train, outputHiddenLayer1, outputHiddenLayer2)

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
