import pandas as pd
import numpy as np
import math, datetime, os
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

def SoftMax(array): return np.exp(array) / sum(np.exp(array), axis=0)
def StableSoftMax(array):
    exp = np.exp(array - np.max(array, axis=0))
    return exp / np.sum(exp, axis=0)

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

def gradientDescent(W1, b1, W2, b2, X, Y, *, learningRate, epochs, decay, patience):
    bestLoss = float('inf')
    for epoch in range(epochs):        
        for i in range(Y.shape[0]):
            x, y = X[i], Y[i]
            Z0, Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, x)
            dW1, db1, dW2, db2 = backPropagation(Z0, Z1, A1, A2, W2, y)
            W1, b1, W2, b2 = updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate)
        
        loss = testModelLoss(trainingX, trainingY, W1, b1, W2, b2)
        
        print(f'Epoch {epoch + 1}{" " * (3 - math.floor(math.log10(epoch + 1)))} | Loss: {loss}')

        if loss <= bestLoss: bestLoss, patienceCount = loss, 0
        else: patienceCount += 1
        if patienceCount >= patience:
            print(f'\nSTOPPING TRAINING AT EPOCH {epoch + 1}')
            break

        learningRate = round(learningRate * decay, 6)

    return W1, b1, W2, b2

# Testing Functions
def testModelLoss(X, Y, W1, b1, W2, b2):
    Z0, Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, X)
    return CategoricalCrossEntropyLoss(A2, Y)

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

# Save/Load Network Functions
def saveParameters(path, W1, b1, W2, b2):
    if not os.path.exists(path): print('Path not found, folder created'), os.mkdir(path)

    pd.DataFrame(W1).to_csv(path, index=False, sep=',')
    pd.DataFrame(b1).to_csv(path, index=False, sep=',')
    pd.DataFrame(W2).to_csv(path, index=False, sep=',')
    pd.DataFrame(b2).to_csv(path, index=False, sep=',')

def loadParameters(path):
    parameters = []
    for filename in os.listdir(path): parameters.append(pd.read_csv(os.path.join(path, filename)).to_numpy())
    return parameters[0], parameters[1], parameters[2], parameters[3]

# Model
batchedX, batchedY = batchData(trainingX, trainingY, 15)
W1, b1, W2, b2 = initialiseParameters()
W1, b1, W2, b2 = gradientDescent(W1, b1, W2, b2, batchedX, batchedY, learningRate=0.0015, epochs=200, decay=0.96, patience=5)

print(f'\nAccuracy on training set {round(testModelAccuracy(trainingX, trainingY, W1, b1, W2, b2) * 100, 2)}')
print(f'Accuracy on testing set {round(testModelAccuracy(testingX, testingY, W1, b1, W2, b2) * 100, 2)}')

saveModel = input('Would you like to save the model? ').lower()
if saveModel == 'y' or saveModel == 'yes': saveParameters(input('Specify file path: '), W1, b1, W2, b2)