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
import json


JSON_PATH = "./data_short.json"

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
    def __init__(self, layers):
        np.random.seed(1)
        self.weights = []
        self.layers = layers  # List of ints representing network's architecture (number of nodes on every layer)

        for i in range(len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1)  # MxN matrix that connects every node in current layer to every node in next layer; +1 are for biases
            self.weights.append(w / np.sqrt(layers[i]))  # normalization of variance of each neuron's output

        # the last two layers are a special case where the input
        # connections need a bias term but the output (the last layer, of index [-1]) does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.weights.append(w / np.sqrt(layers[-2]))

    def train(self, X, y, learningRate=0.1, iterations=100000, displayUpdate=1000, normalizeX = False):
        print("Starting NN training")
        if normalizeX:
            X += np.abs(np.min(X))
            X /= np.max(np.abs(X), axis=0)
        X = np.c_[X, np.ones((X.shape[0]))]  # c_ joins slice objects to concatenation along the second axis
        # here we use it insert column of 1s - which is our bias. This approach will make it trainable parameter
        #  we do that concatenation because we want to insert column of 1's as biases

        for iteration in range(iterations):
            for (X_sample, target) in zip(X, y):  # zip joins X and y into tuples
                self._trainPartial(X_sample, target, learningRate)  # train partial works on every individual sample in X, y. One by one these are passed as (X_sample, target) tuples.

            if iteration % displayUpdate == 0:
                print(f"On iteration number {iteration}, loss = {self._calculateLoss(X, y)}")
                self.getAccuracy(X, y)

    def _trainPartial(self, x, y, learningRate):
        x = np.array(x, ndmin=2)  # transform x into matrix
        activations = self._feedForward(x)
        deltas = self._backPropagate(activations, y)
        # weights update
        for layer in range(len(self.weights)):
            self.weights[layer] += -learningRate * activations[layer].T.dot(deltas[layer])

    def getAccuracy(self, X,y):
        #X=np.c_[X, np.ones((X.shape[0]))]
        totalCorrect = 0
        total = 0 
        for(x, target) in zip(X,y):
            x = np.array(x, ndmin=2)
            logits = self._feedForward(x)[-1]
            pred_y = np.array(logits).argmax()
            target = np.array(target).argmax()
            total += 1
            totalCorrect += pred_y == target
        print("Acc is ", totalCorrect/total)

    def _feedForward(self, input_layer):
        layers_activations = [input_layer]
        for layer in range(len(self.weights)):
            current_layer_output = np.dot(layers_activations[layer], self.weights[layer])
            layers_activations.append(self._SigmoidActivation(current_layer_output))
            #layers_activations.append(self._ReLuActivation(current_layer_output))
        #last_layer_output = np.dot(layers_activations[-1], self.weights[-1])
        #layers_activations.append(self._SoftmaxActivation(last_layer_output))
        return layers_activations



    def _backPropagate(self, layers_activations, target_output):
        error = layers_activations[-1] - target_output  # activations[-1] is the last layer - output
        deltas = [error * self._SigmoidDerivative(layers_activations[-1])]
        #deltas = [error * self._ReLuDerivative(layers_activations[-1])]
        for layer in range(len(layers_activations) - 2, 0, -1):  # moving back in the loop, hence the name
            delta = np.dot(deltas[-1], self.weights[layer].T)
            delta = delta * self._SigmoidDerivative(layers_activations[layer])
            #delta = delta * self._ReLuDerivative(layers_activations[layer])
            deltas.append(delta)
        return deltas[::-1]  # reverse this list before returning

    def _calculateLoss(self, X, y):
        y = np.array(y, ndmin=2)
        classifications = self.classify(X, addBias=False)
        loss = 0.5 * np.sum((classifications - y) ** 2)
        return loss

    def classify(self, inputData, addBias=True, normalizeX = False):
        if normalizeX:
            inputData += np.abs(np.min(inputData))
            inputData /= np.max(np.abs(inputData), axis=0)
        classification = np.array(inputData, ndmin=2)
        if addBias:
            classification = np.c_[classification, np.ones((classification.shape[0]))]

        for layer in range(len(self.weights)):
            classification = self._SigmoidActivation(np.dot(classification, self.weights[layer]))
            #classification = self._ReLuActivation(np.dot(classification, self.weights[layer]))
        #classification = self._SoftmaxActivation(np.dot(classification, self.weights[-1]))
        return classification

    def _SigmoidActivation(self, x):
        return 1 / (1 + np.exp(-x))

    def _SigmoidDerivative(self, x):
        return x * (1 - x)

    def _ReLuActivation(self, x):
        return np.maximum(0, x)

    def _ReLuDerivative(self, x):
        return 1.0 * (x > 0)

    def _SoftmaxActivation(self, x):
        expValues = np.exp(x - np.max(x))
        output = expValues / np.sum(expValues)
        return output

    # To niepotrzebne w sumie, chyba że trenowanie będzie trwało 50 godzin to może lepiej wtedy zaimplementować
    def saveNetworkParameters(self, fileName='nn_parameters.npy'):
        print(f'Saving neural network parameters to file {fileName}')
        from numpy import save
        save(fileName, self.weights)


    # To niepotrzebne w sumie, chyba że trenowanie będzie trwało 50 godzin to może lepiej wtedy zaimplementować
    def loadNetworkParameters(self, fileName='nn_parameters.npy'):
        print(f'Loading neural network parameters from file {fileName}')
        from numpy import load
        self.weights = load(fileName)


def prettyPrintResults(expectedOutput, NNOutput):
    for output_sample in range(min(len(expectedOutput), len(NNOutput))):
        print(f"Test sample {output_sample+1}:\nExpected output: {expectedOutput[output_sample]} NN output: {NNOutput[output_sample]}")
    print()  # newline


RELEASE_VERSION = False
RUN_TESTS = True

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

    if RUN_TESTS:
        print("Starting tests")

        def test1_simple_one_layer_nn():
            print(f"\nStarting test1 - {test1_simple_one_layer_nn.__name__}")
            print("Testing NN with 4 inputs and 1 output. We want NN to output 1 whenever the rightmost of 4 input bits is 1.\n")
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

            testNN = NeuralNetwork([len(trainingX[0]), 1])
            print(f"Weights at the beginning: {testNN.weights}")
            output = testNN.classify(trainingX)
            prettyPrintResults(trainingy, output)

            testNN.train(trainingX, trainingy, iterations=100000, displayUpdate=10000)
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
            print(f"\nStarting test2 - {test2_parametrized_w_hidden_layer_nn.__name__}")
            print("Testing neural network number 1 - infamous 3 input XOR which so horribly failed to be trained to 1-layer NN in test 1.\n")
            trainingX = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0],
                                  [0, 1, 1],
                                  [1, 0, 0],
                                  [1, 0, 1],
                                  [1, 1, 0],
                                  [1, 1, 1]])
            trainingy = np.array([[0, 1, 1, 0, 1, 0, 0, 1]]).T

            testNN = NeuralNetwork([len(trainingX[0]), 4, 1])
            testNN.train(trainingX, trainingy, iterations=30001, displayUpdate=10000)

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

            testNN = NeuralNetwork([len(trainingX[0]), 3, 4, 1])
            testNN.train(trainingX, trainingy, iterations=30001, displayUpdate=10000)

            testX = np.array([[0, 0, 0, 1],
                              [1, 0, 1, 0],
                              [0, 0, 0, 1],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
            testy = np.array([[1, 0, 1, 1, 0, 1]]).T
            print (testX)
            output = testNN.classify(testX)
            prettyPrintResults(testy, output)

        def test3_MNIST_classification_nn():
            print(f"\nStarting test3 - {test3_MNIST_classification_nn.__name__}")
            print("\nTesting neural network number 3 - MNIST dataset with much larger NN than all the previous times.\n")
            from sklearn import datasets
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelBinarizer
            from sklearn.metrics import classification_report

            # load the MNIST dataset and apply min/max scaling to scale the
            # pixel intensity values to the range [0, 1] (each image is
            # represented by an 8 x 8 = 64-dim feature vector)
            digits = datasets.load_digits()  # load MNIST dataset
            data = digits.data.astype("float")
            data = (data - data.min()) / (data.max() - data.min())
            print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

            # convert the labels from integers to vectors
            (trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

            # convert the labels from integers to vectors
            trainY = LabelBinarizer().fit_transform(trainY)
            testY = LabelBinarizer().fit_transform(testY)

            testNN = NeuralNetwork([trainX.shape[1], 32, 16, 10])  # 10 outputs is necessary for every digit
            print("[INFO] training network...")
            testNN.train(trainX, trainY, learningRate=0.1, iterations=1000, displayUpdate=100)

            print("[INFO] evaluating network...")
            classifications = testNN.classify(testX)
            prettyPrintResults(testY, classifications)
            #classifications = classifications.argmax(axis=1)  # argmax finds class classification with the greatest node output value
            #print(classification_report(testY.argmax(axis=1), classifications))

        def test4_music_classification():
            print(f"\nStarting test4 - {test4_music_classification.__name__}")
            print("Training NN to recognize genre, then verifying with one song.\n")

            with open(JSON_PATH, "r") as fp:
              data = json.load(fp)

            # convert lists to numpy arrays
            trainingMfcc = np.array(data["mfcc"])
            trainingMfccFlat = trainingMfcc.reshape((trainingMfcc.shape[0], trainingMfcc.shape[1]*trainingMfcc.shape[2]))
            trainingLabels = np.array(data["labels"])
            print(trainingMfcc.shape)
            print(trainingMfccFlat.shape)
            print(trainingLabels.shape)

            testNN = NeuralNetwork([len(trainingMfccFlat[0]), 512, 265, 64, 10])

            if JSON_PATH == "./data_original.json":
                testNN.train(trainingMfccFlat, trainingLabels, learningRate=0.001, iterations=10, displayUpdate=1, normalizeX=True)
                testMfcc = [trainingMfccFlat[0], trainingMfccFlat[2100], trainingMfccFlat[3700], trainingMfccFlat[5500]]
                testMfcc = np.array(testMfcc)
                testLabels = [trainingLabels[0], trainingLabels[2100], trainingLabels[3700], trainingLabels[5500]]
                testLabels = np.array(testLabels)
            elif JSON_PATH == "./data_short.json":
                testNN.train(trainingMfccFlat, trainingLabels, learningRate=0.1, iterations=500, displayUpdate=100, normalizeX=True)
                testMfcc = [trainingMfccFlat[0], trainingMfccFlat[30], trainingMfccFlat[55], trainingMfccFlat[89]]
                testMfcc = np.array(testMfcc)
                testLabels = [trainingLabels[0], trainingLabels[30], trainingLabels[55], trainingLabels[89]]
                testLabels = np.array(testLabels)
            elif JSON_PATH == "./data_shorter.json":
                testNN.train(trainingMfccFlat, trainingLabels, learningRate=0.001, iterations=10000, displayUpdate=1000, normalizeX=True)
                testMfcc = [trainingMfccFlat[0], trainingMfccFlat[3], trainingMfccFlat[5], trainingMfccFlat[8]]
                testMfcc = np.array(testMfcc)
                testLabels = [trainingLabels[0], trainingLabels[3], trainingLabels[5], trainingLabels[8]]
                testLabels = np.array(testLabels)

            output = testNN.classify(testMfcc, normalizeX=True)
            prettyPrintResults(testLabels, output)

            # output = testNN.classify(trainingMfccFlat)
            # prettyPrintResults(trainingLabels, output)


        def test5_music_classification_1output():
            print(f"\nStarting test5 - {test5_music_classification_1output.__name__}")
            print("Training NN with 1 output to recognize specified genre from index 0, that is Blues.\n")

            with open(JSON_PATH, "r") as fp:
              data = json.load(fp)

            # convert lists to numpy arrays
            trainingMfcc = np.array(data["mfcc"])
            trainingMfccFlat = trainingMfcc.reshape((trainingMfcc.shape[0], trainingMfcc.shape[1]*trainingMfcc.shape[2]))
            trainingLabels = np.array(data["labels"])
            print(trainingMfcc.shape)
            print(trainingMfccFlat.shape)
            print(trainingLabels.shape)

            testNN = NeuralNetwork([len(trainingMfccFlat[0]), 512, 264, 64, 1])

            if JSON_PATH == "./data_original.json":
                testNN.train(trainingMfccFlat, trainingLabels, learningRate=0.001, iterations=10, displayUpdate=1, normalizeX=True)
                testMfcc = [trainingMfccFlat[0], trainingMfccFlat[2100], trainingMfccFlat[3700], trainingMfccFlat[5500]]
                testMfcc = np.array(testMfcc)
                testLabels = [trainingLabels[0], trainingLabels[2100], trainingLabels[3700], trainingLabels[5500]]
                testLabels = np.array(testLabels)
            elif JSON_PATH == "./data_short.json":
                testNN.train(trainingMfccFlat, trainingLabels, learningRate=0.001, iterations=100, displayUpdate=10, normalizeX=True)
                testMfcc = [trainingMfccFlat[0], trainingMfccFlat[30], trainingMfccFlat[55], trainingMfccFlat[89]]
                testMfcc = np.array(testMfcc)
                testLabels = [trainingLabels[0], trainingLabels[30], trainingLabels[55], trainingLabels[89]]
                testLabels = np.array(testLabels)
            elif JSON_PATH == "./data_shorter.json":
                testNN.train(trainingMfccFlat, trainingLabels, learningRate=0.001, iterations=10000, displayUpdate=1000, normalizeX=True)
                testMfcc = [trainingMfccFlat[0], trainingMfccFlat[3], trainingMfccFlat[5], trainingMfccFlat[8]]
                testMfcc = np.array(testMfcc)
                testLabels = [trainingLabels[0], trainingLabels[3], trainingLabels[5], trainingLabels[8]]
                testLabels = np.array(testLabels)

            output = testNN.classify(testMfcc)
            prettyPrintResults(testLabels, output)

            # output = testNN.classify(trainingMfccFlat)
            # prettyPrintResults(trainingLabels, output)

        def test6_save_nn_to_file():
            print(f"\nStarting test6 - {test6_save_nn_to_file.__name__}")
            print("Testing NN with 4 inputs and 1 output. We want NN to output 1 whenever the rightmost of 4 input bits is 1. NN is saved to file in the end\n")
            trainingX = np.array([[0, 0, 0, 0],
                                  [0, 0, 1, 1],  # this outputs 1
                                  [0, 1, 0, 1],  # and this
                                  [0, 1, 1, 0],
                                  [1, 0, 0, 1],  # and this
                                  [1, 0, 1, 0],
                                  [1, 1, 0, 0],
                                  [1, 1, 1, 0]])
            trainingy = np.array([[0, 1, 1, 0, 1, 0, 0, 0]]).T

            testNN = NeuralNetwork([len(trainingX[0]), 1])

            testNN.train(trainingX, trainingy, iterations=100000, displayUpdate=10000)
            testX = np.array([[0, 0, 0, 1],
                              [1, 0, 1, 0],
                              [0, 0, 0, 1],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
            testy = np.array([[1, 0, 1, 1, 0, 1]]).T
            output = testNN.classify(testX)
            prettyPrintResults(testy, output)
            testNN.saveNetworkParameters("4in1outNN.npy")

        def test7_load_nn_from_file():
            print(f"\nStarting test7 - {test7_load_nn_from_file.__name__}")
            print("Loading NN from file. Testing NN with 4 inputs and 1 output. We want NN to output 1 whenever the rightmost of 4 input bits is 1.\n")
            testNN = NeuralNetwork([4, 1])
            testNN.loadNetworkParameters("4in1outNN.npy")
            testX = np.array([[0, 0, 0, 1],
                              [1, 0, 1, 0],
                              [0, 0, 0, 1],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
            testy = np.array([[1, 0, 1, 1, 0, 1]]).T
            output = testNN.classify(testX)
            prettyPrintResults(testy, output)


        #test1_simple_one_layer_nn()
        #test2_parametrized_w_hidden_layer_nn()
        #test3_MNIST_classification_nn()
        #test4_music_classification()
        #test5_music_classification_1output()
        test6_save_nn_to_file()
        test7_load_nn_from_file()



