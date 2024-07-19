import pandas as pd
import numpy as np
import random, math
import MNISTdata

def getData(filePath):
    data = pd.read_csv(filePath)
    data = np.array(data).T
    labels = data[0]

    return labels, np.delete(data, 0, axis=0) / 255

# TrainingY (60000x1) | TrainingX (784x60000)
# TestingY (10000x1) | TestingX (784x10000)
# trainingY, trainingX = getData(MNISTdata.trainingData)
# testingY, testingX = getData(MNISTdata.testingData)

def ReLU(array): return np.maximum(array, 0)
def dReLU(array): return (array > 0) * 1
def LeakyReLU(array, alpha): return np.maximum(array * alpha, array)
def dLeakyReLU(array, alpha): return np.where(array > 0, 1, alpha)

def SoftMax(array): return np.exp(array) / sum(np.exp(array))
def dSoftMax(array): return SoftMax(array) * (1 - SoftMax(array))
def StableSoftMax(array): return np.exp(array - max(array)) / sum(np.exp(array))
def dStableSoftMax(array): return StableSoftMax(array) * (1 - StableSoftMax(array))

def initialiseParameters():
    # Generate 16x784 Weight Matrix & 16x1 Bias Matrix
    W1 = np.random.uniform(-0.5, 0.5, (16,784))
    b1 = np.random.uniform(-0.5, 0.5, (16,1))

    # Generate 10x16 Weight Matrix & 10x1 Bias Matrix
    W2 = np.random.uniform(-0.5, 0.5, (10,16))
    b2 = np.random.uniform(-0.5, 0.5, (10,1))

    return W1, b1, W2, b2

W1, b1, W2, b2 = initialiseParameters()