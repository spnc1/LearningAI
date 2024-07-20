import pandas as pd
import numpy as np
import MNISTdata

import matplotlib.pyplot as plt

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
def LeakyReLU(array, decay = 0.1): return np.maximum(array * decay, array)
def dLeakyReLU(array, decay = 0.1): return np.where(array > 0, 1, decay)

def SoftMax(array): return np.exp(array) / sum(np.exp(array))
def dSoftMax(array): return SoftMax(array) * (1 - SoftMax(array))
def StableSoftMax(array):
    exp = np.exp(array - np.max(array))
    return exp / np.sum(exp)
def dStableSoftMax(array): return StableSoftMax(array) * (1 - StableSoftMax(array))

def initialiseParameters():
    # Generate 16x784 Weight Matrix & 16x1 Bias Matrix
    W1 = np.random.uniform(-0.5, 0.5, (16, 784))
    b1 = np.random.uniform(-0.5, 0.5, (16, 1))

    # Generate 10x16 Weight Matrix & 10x1 Bias Matrix
    W2 = np.random.uniform(-0.5, 0.5, (10, 16))
    b2 = np.random.uniform(-0.5, 0.5, (10, 1))

    return W1, b1, W2, b2

# Propagation Functions
def forwardPropagation(W1, b1, W2, b2, Z0):
    Z1 = W1 @ Z0 + b1
    A1 = LeakyReLU(Z1, 0.1)
    Z2 = W2 @ Z1 + b2
    A2 = StableSoftMax(Z2)
    return Z0, Z1, A1, Z2, A2

def backPropagation(Z0, Z1, A1, A2, W2, Y):
    dZ2 = A2 - Y
    dW2 = dZ2 @ A1.T
    dZ1 = (W2.T @ dZ2) * dLeakyReLU(Z1, 0.1)
    dW1 = dZ1 @ Z0.T
    return dW1, dZ1, dW2, dZ2

def updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate):
    W1 = W1 - learningRate * dW1
    b1 = b1 - learningRate * db1
    W2 = W2 - learningRate * dW2
    b2 = b2 - learningRate * db2

    return W1, b1, W2, b2

def batchData(data, labels, batchSize, axis):
    if data.shape[axis] % batchSize != 0: raise ValueError(f'Data of shape {data.shape} is not divisible along axis {axis} by {batchSize}')
    splits = data.shape[axis] // batchSize
    splitData = np.array(np.split(data, splits, axis))
    splitLabels = np.array(np.split(labels, splits, axis))
    
    return splitData, splitLabels

# Testing Functions
def testModelLoss(data, labels, W1, b1, W2, b2):
    losses = []
    for i in range(data.shape[1]):
        testX = data[:, i].reshape(784, 1)
        testY = labels[:, i].reshape(10, 1)
        Z0, Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, testX)
        losses.append(CategoricalCrossEntropyLoss(A2, testY))
    return sum(losses) / data.shape[1]

def testModelAccuracy(data, labels, W1, b1, W2, b2):
    count = 0
    for i in range(data.shape[1]):
        testX = data[:, i].reshape(784, 1)
        testY = labels[:, i].reshape(10, 1)
        Z0, Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, testX)

        if np.argmax(testY, 0)[0] == np.argmax(A2, 0)[0]: count += 1
    
    return count / data.shape[1]

def CategoricalCrossEntropyLoss(Output, Y, epsilon = 1e-15):
    clippedOutput = np.clip(Output, epsilon, 1 - epsilon)
    return -np.sum(Y * np.log(clippedOutput))

# W1, b1, W2, b2 = initialiseParameters()
testingX, testingY = batchData(testingX, testingY, 10, 1)

print(testingY[0])