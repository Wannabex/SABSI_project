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
        np.random.seed(0)
        # inputLayer = NeuronsLayer()
        # hiddenLayer1 = NeuronsLayer()
        # inputLayer2 = NeuronsLayer()
        # outputLayer = NeuronsLayer()
        self.weights = 2 * np.random.random((3, 1)) - 1  # 3x1 matrix
        self.biases = np.zeros((1, 3))  # 3 biasses, 1 for each neuron
        self.alpha = alpha  # learning rate of neural network

    def train(self, X, y, iterations=1000):
        print(f'Starting training with {iterations} iterations on inputs:\n {X},\n that result in outputs:\n {y}')
        for iteration in range(iterations):
            pass


    # To niepotrzebne w sumie, chyba że trenowanie będzie trwało 50 godzin to może lepiej wtedy zaimplementować
    def saveNetworkParameters(self, fileName='nn_parameters'):
        print(f'Saving neural network parameters to file {fileName}')

    # To niepotrzebne w sumie, chyba że trenowanie będzie trwało 50 godzin to może lepiej wtedy zaimplementować
    def loadNetworkParameters(self, fileName='nn_parameters'):
        print(f'Loading neural network parameters from file {fileName}')

    def classify(self, inputData):
        output = self._ReLuActivation(np.dot(inputData, self.weights) + self.biases)
        output = self._SoftmaxActivation(output)
        return np.sum(output, axis=1, keepdims=True).astype(int)

    def feedForward(self):
        pass

    def backPropagate(self):
        pass


    def _ReLuActivation(self, inputs):
        return np.maximum(0, inputs)

    def _SoftmaxActivation(self, inputs):
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return expValues / np.sum(expValues, axis=1, keepdims=True)

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
            testX = trainingX[2:6]
            testy = trainingy[2:6]
            testNn = NeuralNetwork()
            print(testNn.weights)
            testNn.train(trainingX, trainingy)

            print(f"Sending inputs:\n {testX}\n and expecting outputs:\n {testy}")
            testOutput = testNn.classify(testX)
            print(f"Got output:\n{testOutput}")
            print(testy == testOutput)

        test_1_layer_nn()



