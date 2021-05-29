# SABSI
# lato 2021 eit mgr
# OK KL MR

# main.py
# tu całe radio działa

"""
Główne założenia:
1. Uczenie sieci neuronowej (NN) żeby rozpoznawała gatunek(genre) piosenki na podstawie jej elementów charakterystycznych
   używany do nauki dataset to np https://github.com/mdeff/fma "fma_small.zip" czyli 8k utworów po 30s w 8 gatunkach
   Dataset może być wgrywany i analizowany jaka sample o różnych częstotliwościach,  może spektrogram
   biblioteka - librosa https://librosa.org/doc/latest/index.html

2. Inteligentne radio ma wczytać piosenkę której ścieżkę w systemie plików podamy.
   Piosenka będzie sprawdzana przez NN i klasyfikowana jako jeden z N gatunków
   Jeśli gatunek będzie taki sam jak wybrany przez nas forbiddenGenre to nie gramy jej, wychodzi error

3. NN trza będzie napisać od zera, tzn obiekty pojedynczych neuronów będą tworzyły warstwy (layers).
   Między warstwami będą wykonywane funkcje:
   feedForward - przesyłanie danych wejściowych dalej przez wszystkie neurony aż do wyjścia i uzyskania jakiegoś wyniku
   backPropagate - najważniejsza w procesie uczenia sieci, liczy błąd klasyfikacji sieci na jej wyjściu i informuje o nim
                   poprzednie warstwy z neuronami, aby poprawiły swoje parametry wag i biasy



Podział zadań:
KL 1. Napisanie sieci neuronowej zaczynając od pojedynczych neuronów i przez warstwy, razem z funkcjami feedforward i backpropagate.
   Wynikiem powinno być API dla następnych etapów do trenowania i uruchamiania NN w radiu.
MR 2. Opracowanie sieci neuronowej jaka będzie używana w radiu(wejścia, wyjścia i layery) i trenowanie NN tak aby miała jak największą skuteczność.
   Trza będzie ogarnąć dane do trenowania (dataset), ale też sposób w jaki będą one w ogóle analizowane.
OK 3. Przygotowanie frontendu i klasy radia czyli umożliwienie włączania muzyki użytkownikowi.
   Dodatkowo przygotowanie danych(muzyki) innych niż te którymi trenowano aby zweryfikować NN.

"""

import numpy as np  # for better representation of data and to simplify mathematical operations that happen between neurons


class IntelligentRadio:
    POP = 0
    DISCOPOLO = 1
    HIPHOP = 2
    # itp itd.

    def __init__(self, neuralNetwork, forbiddenGenre):
        self.intelligence = neuralNetwork
        # music player object
        self.forbiddenGenre = forbiddenGenre

    def classifyAndPlay(self, mp3_path):
        # song = mp3open(mp3_path)
        # genre = checkGenre(song)
        # if genre == self.forbiddenGenre:
        #   dont play this song!
        # playmp3(song)
        pass

    def checkGenre(self, song):
        # classifiedGenre = self.intelligence.classify(song)
        pass


class NeuralNetwork:
    def __init__(self, noLayers=4, alpha=0.1):
        np.random.seed(1)
        # inputLayer = NeuronsLayer()
        # hiddenLayer1 = NeuronsLayer()
        # inputLayer2 = NeuronsLayer()
        # outputLayer = NeuronsLayer()
        self.weights = 2 * np.random.random((3, 1)) - 1  # 3x1 matrix
        self.biases = np.zeros((1, 3))  # 3 biases, 1 for each neuron
        self.alpha = alpha  # learning rate of neural network

    def train(self, X, y, iterations=100000):
        print("Starting NN training")
        y = y.astype(float)
        for iteration in range(iterations):
            # Pass training set through network
            output = self._feedForward(X)

            # Calculate the error of each neuron
            error = y - output

            # Multiply error ny input and gradient of activation function
            # Because of the gradient, more significant weights are adjusted more
            adjustments = np.dot(X.T, error * self._SigmoidDerivative(output))

            # Adjust the weights
            self.weights = self.weights + adjustments

            if iteration % 10000 == 0:
                print(f"On iteration number {iteration}, error = {error}")


    # To niepotrzebne w sumie, chyba że trenowanie będzie trwało 50 godzin to może lepiej wtedy zaimplementować
    def saveNetworkParameters(self, fileName='nn_parameters'):
        print(f'Saving neural network parameters to file {fileName}')

    # To niepotrzebne w sumie, chyba że trenowanie będzie trwało 50 godzin to może lepiej wtedy zaimplementować
    def loadNetworkParameters(self, fileName='nn_parameters'):
        print(f'Loading neural network parameters from file {fileName}')

    def classify(self, inputData):
        activations = self._feedForward(inputData)
        return np.sum(activations, axis=1, keepdims=True)

    def _feedForward(self, inputData):
        inputData = inputData.astype(float)
        activation = self._SigmoidActivation(np.dot(inputData, self.weights))
        #activation = self._SoftmaxActivation(activation)
        return activation

    def _backPropagate(self):
        pass

    def _ReLuActivation(self, x):
        return np.maximum(0, x)

    def _SigmoidActivation(self, x):
        output = 1 / (1 + np.exp(-x))
        return output

    def _SigmoidDerivative(self, x):
        output = x * (1 - x)
        return output

    def _SoftmaxActivation(self, x):
        expValues = np.exp(x - np.max(x, axis=1, keepdims=True))
        output = expValues / np.sum(expValues, axis=1, keepdims=True)
        return output

#
# class NeuronsLayer:
#     def __init__(self, noNeurons):
#         pass
#
#
# # możliwe że nie będzie potrzebne, albo że za bardzo wszystko utrudni
# class Neuron:
#     def __init__(self):
#         self.weight = 0.5 # random
#         self.bias = 0.1


RELEASE_VERSION = 0
KL_TEST_VERSION = 1

if __name__ == '__main__':
    if RELEASE_VERSION:
        print('Starting intelligent radio')
        radioIntelligence = NeuralNetwork()
        radioIntelligence.train()  # albo musicNN.loadNetworkParameters()
        # musicNN.saveNetworkParameters()

        radyjko = IntelligentRadio(radioIntelligence, IntelligentRadio.DISCOPOLO, )
        wantedSongName = 'moje_ulubione_discopolo.mp3'
        radyjko.classifyAndPlay(wantedSongName)
        print('Turning off the radio')

    elif KL_TEST_VERSION:
        print("Starting KL part test")

        def test_1_layer_nn():
            print("Testing NN with 3 inputs and 1 output - for example XOR can be the function for NN to learn")
            trainingX = np.array([ [0, 0, 0],
                                   [0, 0, 1],
                                   [0, 1, 0],
                                   [0, 1, 1],
                                   [1, 0, 0],
                                   [1, 0, 1],
                                   [1, 1, 0],
                                   [1, 1, 1] ])
            trainingy = np.array([[0, 1, 1, 0, 1, 0, 0, 1]]).T
            testNn = NeuralNetwork()
            print(f"Weights at the beginning: {testNn.weights}")
            output = testNn.classify(trainingX)
            # print(f"Before training got output:\n{output}")
            # print(trainingy == output)

            testNn.train(trainingX, trainingy, iterations=3)
            print(f"Weights after training: {testNn.weights}")
            output = testNn.classify(trainingX)
            # print(f"After training got output:\n{output}")
            # print(trainingy == output)

        test_1_layer_nn()



