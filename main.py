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
    DEBUG = True
    def __init__(self, noLayers=4, alpha=0.1):
        np.random.seed(1)
        # inputLayer = NeuronsLayer()
        # hiddenLayer1 = NeuronsLayer()
        # inputLayer2 = NeuronsLayer()
        # outputLayer = NeuronsLayer()
        self.weights = 2 * np.random.random((4, 1)) - 1  # 3x1 matrix
        self.biases = np.zeros((1, 1))  # all biases, 1 for each output neuron - here only 1
        self.alpha = alpha  # learning rate of neural network

    def train(self, X, y, iterations=100000):
        print("Starting NN training")
        for iteration in range(iterations):
            # Pass training set through network
            output = self.classify(X)

            # Calculate the error rate
            error = y - output
            if NeuralNetwork.DEBUG:
                if iteration % 10000 == 0:
                    print(f"On iteration number {iteration}, error = {error}")

            # Multiply error ny input and gradient of activation function
            # Because of the gradient, more significant weights are adjusted more
            adjustments = self.alpha * np.dot(X.T, np.multiply(error, self._SigmoidDerivative(output)))

            # Adjust the weights
            self.weights += adjustments


    # To niepotrzebne w sumie, chyba że trenowanie będzie trwało 50 godzin to może lepiej wtedy zaimplementować
    def saveNetworkParameters(self, fileName='nn_parameters'):
        print(f'Saving neural network parame ers to file {fileName}')

    # To niepotrzebne w sumie, chyba że trenowanie będzie trwało 50 godzin to może lepiej wtedy zaimplementować
    def loadNetworkParameters(self, fileName='nn_parameters'):
        print(f'Loading neural network parameters from file {fileName}')

    def classify(self, inputData):
        activations = self._feedForward(inputData)
        return activations

    def _feedForward(self, inputData):
        output = self._SigmoidActivation(np.dot(inputData, self.weights))
        #output = self._ReLuActivation(np.dot(inputData, self.weights)) #  works far worse in 1 layer case, will probably be superior when working with more
        return output

    def _backPropagate(self):
        pass

    def _ReLuActivation(self, x):
        return np.maximum(0, x)

    def _SigmoidActivation(self, x):
        return 1 / (1 + np.exp(-x))

    def _SigmoidDerivative(self, x):
        return x * (1 - x)

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


def prettyPrintResults(expectedOutput, NNOutput):
    for output_sample in range(min(len(expectedOutput), len(NNOutput))):
        print(f"Test sample {output_sample}:\nExpected output: {expectedOutput[output_sample]} NN output: {NNOutput[output_sample]}")
    print() # newline

RELEASE_VERSION = False
KL_TEST_VERSION = True

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
            print("Testing NN with 4 inputs and 1 output. We want NN to output 1 whenever the rightmost of 4 input bits is 1")
            # For some reason there is problem when trying to make NN learn XOR:
            # http://home.agh.edu.pl/~vlsi/AI/xor_t/en/main.html

            trainingX = np.array([ [0, 0, 0, 0],
                                   [0, 0, 1, 1],  # this outputs 1
                                   [0, 1, 0, 1],  # and this
                                   [0, 1, 1, 0],
                                   [1, 0, 0, 1],  # and this
                                   [1, 0, 1, 0],
                                   [1, 1, 0, 0],
                                   [1, 1, 1, 0] ])
            trainingy = np.array([[0, 1, 1, 0, 1, 0, 0, 0]]).T

            testNn = NeuralNetwork(alpha=0.5)
            print(f"Weights at the beginning: {testNn.weights}")
            output = testNn.classify(trainingX)
            prettyPrintResults(trainingy, output)

            testNn.train(trainingX, trainingy, iterations=100000)
            print(f"Weights after training: {testNn.weights}")
            output = testNn.classify(trainingX)
            prettyPrintResults(trainingy, output)

            testX = np.array([[0, 0, 0, 1],
                              [1, 0, 1, 0],
                              [0, 0, 0, 1],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
            testy = np.array([[1, 0, 1, 1, 0, 1]]).T
            output = testNn.classify(testX)
            prettyPrintResults(testy, output)
        test_1_layer_nn()



