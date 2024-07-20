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
trainingY, trainingX = getData(MNISTdata.trainingData)
trainingY = oneHot(trainingY, 10)
testingY, testingX = getData(MNISTdata.testingData)
testingY = oneHot(testingY, 10)

# Activation Functions + Derivatives
def ReLU(array): return np.maximum(array, 0)
def dReLU(array): return (array > 0) * 1

def LeakyReLU(array, decay = 0.1): return np.maximum(array * decay, array)
def dLeakyReLU(array, decay = 0.1): return np.where(array > 0, 1, decay)

def SoftMax(array): return np.exp(array) / sum(np.exp(array))
def StableSoftMax(array):
    exp = np.exp(array - np.max(array, axis=0, keepdims=True))
    return exp / np.sum(exp)

# Propagation Functions
def initialiseParameters():
    # Generate 16x784 Weight Matrix & 16x1 Bias Matrix
    W1 = np.random.uniform(-0.5, 0.5, (16, 784))
    b1 = np.random.uniform(-0.5, 0.5, (16, 1))

    # Generate 10x16 Weight Matrix & 10x1 Bias Matrix
    W2 = np.random.uniform(-0.5, 0.5, (10, 16))
    b2 = np.random.uniform(-0.5, 0.5, (10, 1))

    return W1, b1, W2, b2

def forwardPropagation(W1, b1, W2, b2, Z0):
    Z1 = W1 @ Z0 + b1
    A1 = LeakyReLU(Z1, 0.1)
    Z2 = W2 @ Z1 + b2
    A2 = StableSoftMax(Z2)
    return Z0, Z1, A1, Z2, A2

def backPropagation(Z0, Z1, A1, A2, W2, Y):
    batchSize = A2.shape[1]

    dZ2 = (A2 - Y)
    dW2 = (dZ2 @ A1.T) / batchSize
    db2 = np.sum(dZ2, axis=1, keepdims=True) / batchSize

    dZ1 = ((W2.T @ dZ2) * dLeakyReLU(Z1, 0.1))
    dW1 = (dZ1 @ Z0.T) / batchSize
    db1 = np.sum(dZ1, axis=1, keepdims=True) / batchSize

    return dW1, db1, dW2, db2

def updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate):
    W1 = W1 - learningRate * dW1
    b1 = b1 - learningRate * db1
    W2 = W2 - learningRate * dW2
    b2 = b2 - learningRate * db2
    return W1, b1, W2, b2

def batchData(data, labels, batchSize, axis = 1):
    if data.shape[axis] % batchSize != 0: raise ValueError(f'Data of shape {data.shape} is not divisible along axis {axis} by {batchSize}')
    splits = data.shape[axis] // batchSize
    splitData = np.array(np.split(data, splits, axis))
    splitLabels = np.array(np.split(labels, splits, axis))
    
    return splitData, splitLabels

# Testing Functions
def testModelLoss(X, Y, W1, b1, W2, b2):
    losses = []
    for i in range(Y.shape[0]):
        lossX = X[i]
        lossY = Y[i]
        
        Z0, Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, lossX)
        losses.append(CategoricalCrossEntropyLoss(A2, lossY))
    return sum(losses) / Y.shape[0]

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
    return np.mean(-np.sum(Y * np.log(clippedOutput), axis=0))

def gradientDescent(W1, b1, W2, b2, X, Y, *, learningRate, epochs):
    for epoch in range(epochs):
        for i in range(Y.shape[0]):
            x = X[i]
            y = Y[i]

            Z0, Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, x)
            dW1, db1, dW2, db2 = backPropagation(Z0, Z1, A1, A2, W2, y)
            W1, b1, W2, b2 = updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate)

            # if i == 20000:
            #     print(dW1.shape)
            #     print(db1.shape)
            #     print(dW2.shape)
            #     print(db2.shape)
                # print(CategoricalCrossEntropyLoss(A2, y))
    
    return W1, b1, W2, b2

batchedX, batchedY = batchData(trainingX, trainingY, 2)

W1, b1, W2, b2 = initialiseParameters()
print(testModelAccuracy(trainingX, trainingY, W1, b1, W2, b2))

W1, b1, W2, b2 = gradientDescent(W1, b1, W2, b2, batchedX, batchedY, learningRate=0.001, epochs=1)
print(W1.shape, b1.shape)
print(W2.shape, b2.shape)
print(testModelAccuracy(trainingX, trainingY, W1, b1, W2, b2))