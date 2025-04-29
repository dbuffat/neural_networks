from . import neurones as nr
from . import neurones_keras as nk

def main_neural_numpy():
    nrnp = nr.NeuralNetwork()
    nrnp.RunMain()
    
def main_neural_keras():
    nrnk = nk.NeuralNetworkKeras()
    nrnk.RunMain()
    
def main_train1():
    nrnk = nk.NeuralNetworkKeras()
    nrnk.Run1()

def main_train2():
    nrnk = nk.NeuralNetworkKeras()
    nrnk.Run2()
    
def main_train3():
    nrnk = nk.NeuralNetworkKeras()
    nrnk.Run3()
    
def main_train4():
    nrnk = nk.NeuralNetworkKeras()
    nrnk.Run4()
    
def main_train5():
    nrnk = nk.NeuralNetworkKeras()
    nrnk.Run5()
