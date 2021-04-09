# SABSI
# lato 2021 eit mgr
# OK KL MR

# main.py
# tu całe radio działa

"""
Plan działania:

1. Uczenie sieci neuronowej (NN) żeby rozpoznawała gatunek(genre) piosenki na podstawie jej elementów charakterystycznych
   używany do nauki dataset to np https://github.com/mdeff/fma "fma_small.zip" czyli 8k utworów po 30s w 8 gatunkach
   Dataset może być wgrywany i analizowany jaka sample o różnych częstotliwościach,  może spektrogram
   biblioteka - librosa https://librosa.org/doc/latest/index.html

2. Inteligentne radio ma wczytać piosenkę której ścieżkę w systemie plików podamy.
   Piosenka będzie sprawdzana przez NN i klasyfikowana jako jeden z 8 gatunków
   Jeśli gatunek będzie taki sam jak wybrany przez nas forbiddenGenre to nie gramy jej, wychodzi error

3. NN trza będzie napisać od zera, tzn obiekty pojedynczych neuronów będą tworzyły warstwy (layers).
   Między warstwami będą wykonywane funkcje:
   feedForward - przesyłanie danych wejściowych dalej przez wszystkie neurony aż do wyjścia i uzyskania jakiegoś wyniku
   backPropagate - najważniejsza w procesie uczenia sieci, liczy błąd klasyfikacji sieci na jej wyjściu i informuje o nim
                   poprzednie warstwy z neuronami, aby poprawiły swoje parametry wag i biasy
"""


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
    def __init__(self, noLayers=4):
        # inputLayer = NeuronsLayer()
        # hiddenLayer1 = NeuronsLayer()
        # inputLayer2 = NeuronsLayer()
        # outputLayer = NeuronsLayer()
        pass

    def startTraining(self, iterations=1000):
        print(f'Starting training with {iterations} iterations')

    # To niepotrzebne w sumie, chyba że trenowanie będzie trwało 50 godzin to może lepiej wtedy zaimplementować
    def saveNetworkParameters(self, fileName='nn_parameters'):
        print(f'Saving neural network parameters to file {fileName}')

    # To niepotrzebne w sumie, chyba że trenowanie będzie trwało 50 godzin to może lepiej wtedy zaimplementować
    def loadNetworkParameters(self, fileName='nn_parameters'):
        print(f'Loading neural network parameters from file {fileName}')

    def classify(self, inputData):
        pass


class NeuronsLayer:
    def __init__(self, noNeurons):
        pass


# możliwe że nie będzie potrzebne, albo że za bardzo wszystko utrudni
class Neuron:
    def __init__(self):
        self.weight = 0.5 # random
        self.bias = 0.1


if __name__ == '__main__':
    print('Starting intelligent radio')
    radioIntelligence = NeuralNetwork()
    radioIntelligence.startTraining()  # albo musicNN.loadNetworkParameters()
    # musicNN.saveNetworkParameters()

    radyjko = IntelligentRadio(radioIntelligence, IntelligentRadio.DISCOPOLO, )
    wantedSongName = 'moje_ulubione_discopolo.mp3'
    radyjko.classifyAndPlay(wantedSongName)
    print('Turning off the radio')
