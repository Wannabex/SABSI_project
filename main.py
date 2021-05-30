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
    # implementation based on https://www.pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/

    DEBUG = True

    def __init__(self, layers):
        np.random.seed(1)

        self.weights = []
        self.layers = layers  # List of ints representing network's architecture



        for i in range(len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1)  # MxN matrix that connects every node in current layer to every node in next layer
            self.weights.append(w / np.sqrt(layers[i]))  # normalization of variance of each neuron's output

        # the last two layers are a special case where the input
        # connections need a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.weights.append(w / np.sqrt(layers[-2]))

    def train(self, X, y, iterations=100000, alpha=0.2):
        print("Starting NN training")
        X = np.c_[X, np.ones((X.shape[0]))]  # c_ joins slice objects to concatenation along the second axis
        #  we do that concatenation because we want to insert column of 1's as biases
        for iteration in range(iterations):

            # Pass training set through network
            output = self.classify(X, addBias=False)

            for (x, target) in zip(X, y):  # zip joins X and y into tuple
                self._trainPartial(x, target, alpha)  # train partial works on every individual sample in X, y separately

            # Calculate the error rate
            error = y - output
            if NeuralNetwork.DEBUG:
                if iteration % 10000 == 0:
                    print(f"On iteration number {iteration}, loss = {self._calculateLoss(X, y)}")

    def _trainPartial(self, x, y, alpha):
        activations = [np.atleast_2d(x)]
        # feed forward here
        for layer in range(len(self.weights)):
            net = activations[layer].dot(self.weights[layer])
            activations.append(self._SigmoidActivation(net))

        # back propagate here
        error = activations[-1] - y  # activations[-1] is the last layer - output
        deltas = [error * self._SigmoidDerivative(activations[-1])]
        for layer in range(len(activations) - 2, 0, -1):
            delta = deltas[-1].dot(self.weights[layer].T)
            delta = delta * self._SigmoidDerivative(activations[layer])
            deltas.append(delta)
        deltas = deltas[::-1]  # reverse this list

        # weights update
        for layer in range(len(self.weights)):
            self.weights[layer] += -alpha * activations[layer].T.dot(deltas[layer])


    def _calculateLoss(self, X, y):
        y = np.atleast_2d(y)
        classifications = self.classify(X, addBias=False)
        loss = 0.5 * np.sum((classifications - y) ** 2)
        
        return loss

    # To niepotrzebne w sumie, chyba że trenowanie będzie trwało 50 godzin to może lepiej wtedy zaimplementować
    def saveNetworkParameters(self, fileName='nn_parameters'):
        print(f'Saving neural network parame ers to file {fileName}')

    # To niepotrzebne w sumie, chyba że trenowanie będzie trwało 50 godzin to może lepiej wtedy zaimplementować
    def loadNetworkParameters(self, fileName='nn_parameters'):
        print(f'Loading neural network parameters from file {fileName}')

    def classify(self, inputData, addBias=True):
        classification = np.atleast_2d(inputData)
        if addBias:
            classification = np.c_[classification, np.ones((classification.shape[0]))]

        for layer in range(len(self.weights)):
            classification = self._SigmoidActivation(np.dot(classification, self.weights[layer]))
        return classification

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
        print(f"Test sample {output_sample+1}:\nExpected output: {expectedOutput[output_sample]} NN output: {NNOutput[output_sample]}")
    print()  # newline

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

        def test1_simple_one_layer_nn():
            print("\nTesting NN with 4 inputs and 1 output. We want NN to output 1 whenever the rightmost of 4 input bits is 1.\n")
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

            testNN = NeuralNetwork([4, 1])
            print(f"Weights at the beginning: {testNN.weights}")
            output = testNN.classify(trainingX)
            prettyPrintResults(trainingy, output)

            testNN.train(trainingX, trainingy, iterations=100000, alpha=0.5)
            print(f"Weights after training: {testNN.weights}")
            output = testNN.classify(trainingX)
            prettyPrintResults(trainingy, output)

            testX = np.array([[0, 0, 0, 1],
                              [1, 0, 1, 0],
                              [0, 0, 0, 1],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
            testy = np.array([[1, 0, 1, 1, 0, 1]]).T
            output = testNN.classify(testX)
            prettyPrintResults(testy, output)

        def test2_parametrized_w_hidden_layer_nn():
            print("\nTesting neural network number 1 - infamous 3 input XOR which so horribly failed to be trained to 1-layer NN in test 1.\n")
            trainingX = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0],
                                  [0, 1, 1],
                                  [1, 0, 0],
                                  [1, 0, 1],
                                  [1, 1, 0],
                                  [1, 1, 1]])
            trainingy = np.array([[0, 1, 1, 0, 1, 0, 0, 1]]).T

            testNN = NeuralNetwork([3, 4, 1])
            testNN.train(trainingX, trainingy, iterations=30001)

            output = testNN.classify(trainingX)
            prettyPrintResults(trainingy, output)
            del testNN
            print("\nTesting neural network number 2 - the same function which worked in test 1.\n")
            trainingX = np.array([[0, 0, 0, 0],
                                  [0, 0, 1, 1],  # this outputs 1
                                  [0, 1, 0, 1],  # and this
                                  [0, 1, 1, 0],
                                  [1, 0, 0, 1],  # and this
                                  [1, 0, 1, 0],
                                  [1, 1, 0, 0],
                                  [1, 1, 1, 0]])
            trainingy = np.array([[0, 1, 1, 0, 1, 0, 0, 0]]).T

            testNN = NeuralNetwork([4, 3, 4, 1])
            testNN.train(trainingX, trainingy, iterations=30001)

            testX = np.array([[0, 0, 0, 1],
                              [1, 0, 1, 0],
                              [0, 0, 0, 1],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
            testy = np.array([[1, 0, 1, 1, 0, 1]]).T
            output = testNN.classify(testX)
            prettyPrintResults(testy, output)

        test1_simple_one_layer_nn()
        #test2_parametrized_w_hidden_layer_nn()



