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

        return outputDelta, hiddenDelta

    def updateWeights(self, outputDelta, hiddenDelta, inputX):
        # Etapa 9
        self.hiddenOutputWeights += np.dot(self.outputHiddenLayer1.T, outputDelta) * self.learningRate
        self.outputBias += np.sum(outputDelta, axis=0, keepdims=True) * self.learningRate

        # Etapa 10
        self.hiddenInputWeights += np.dot(inputX.T, hiddenDelta) * self.learningRate
        self.hiddenBias += np.sum(hiddenDelta, axis=0, keepdims=True) * self.learningRate

    def train(self, inputX, expectedY, epochs):
        for epoch in range(epochs):
            outputHiddenLayer1, outputHiddenLayer2 = self.feedforward(inputX)

            outputDelta, hiddenDelta = self.backPropagation(inputX, expectedY, outputHiddenLayer1, outputHiddenLayer2)

            self.updateWeights(outputDelta, hiddenDelta, inputX)

            # Monitoramento do Erro
            if epoch % 1000 == 0:
                loss = 0.5 * np.sum(np.square((expectedY - outputHiddenLayer2)))
                print(f"Época {epoch}, Erro: {loss}")

    def predict(self, inputX):
        _, predictionResult = self.feedforward(inputX)
        return np.round(predictionResult)
