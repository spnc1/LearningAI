import pandas as pd
import numpy as np
import random, math
import MNISTdata

def getData(filePath):
    data = pd.read_csv(filePath).to_numpy().T
    labels = data[0]
    return labels, np.delete(data, 0, axis=0) / 255

def oneHot(array, oneHotLength):
    oneHot = np.zeros((array.size, oneHotLength))
    oneHot[np.arange(array.size), array] = 1
    return oneHot.T

# TrainingY (60000x1) | TrainingX (784x60000)
# TestingY (10000x1) | TestingX (784x10000)
# trainingY, trainingX = getData(MNISTdata.trainingData)
# trainingY = oneHot(trainingY, 10)
testingY, testingX = getData(MNISTdata.testingData)
testingY = oneHot(testingY, 10)

# Activation Functions + Derivatives
def ReLU(array): return np.maximum(array, 0)
def dReLU(array): return (array > 0) * 1
def LeakyReLU(array, alpha): return np.maximum(array * alpha, array)
def dLeakyReLU(array, alpha): return np.where(array > 0, 1, alpha)

def SoftMax(array): return np.exp(array) / sum(np.exp(array))
def dSoftMax(array): return SoftMax(array) * (1 - SoftMax(array))
def StableSoftMax(array):
    exp = np.exp(array - np.max(array))
    return exp / np.sum(exp)
def dStableSoftMax(array): return StableSoftMax(array) * (1 - StableSoftMax(array))

def initialiseParameters():
    # Generate 16x784 Weight Matrix & 16x1 Bias Matrix
    W1 = np.random.rand(16, 784) - 0.5
    b1 = np.random.rand(16, 1) - 0.5

    # Generate 10x16 Weight Matrix & 10x1 Bias Matrix
    W2 = np.random.rand(10, 16) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def forwardPropagation(W1, b1, W2, b2, Z0):
    # Z0 currently a row vector not a column vector
    Z0 = Z0.reshape(784,1)

    Z1 = W1 @ Z0 + b1
    A1 = LeakyReLU(Z1, 0.1)
    Z2 = W2 @ Z1 + b2
    A2 = StableSoftMax(Z2)
    return Z0, Z1, A1, Z2, A2

def backPropagation(Z0, Z1, A1, A2, W1, W2, Y):
    # Y currently a row vector not a column vector
    Y = Y.reshape(10,1)

    dZ2 = A2 - Y
    dW2 = dZ2 @ A1.T
    dZ1 = W2.T @ dZ2 * dLeakyReLU(Z1, 0.1)
    dW1 = dZ1 @ Z0.T
    return dW1, dZ1, dW2, dZ2

W1, b1, W2, b2 = initialiseParameters()
Z0, Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, testingX[:, 0])
dW1, db1, dW2, db2 = backPropagation(Z0, Z1, A1, A2, W1, W2, testingY[:,0])