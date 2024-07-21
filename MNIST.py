import matplotlib.pyplot as plt
import MNISTdata, math, os
import pandas as pd
import numpy as np

# Network Tools
def getData(filePath: str):
    data = pd.read_csv(filePath).to_numpy().T
    return data[0], np.delete(data, 0, axis=0) / 255

def batchData(data, labels, batchSize: int, axis: int = 1):
    """
    Split data into batches of size batchSize

    Parameters
    ----------
    data: Image data
    labels: One hot encoded labels
    batchSize: Size of each batch

    Outputs
    -------
    splitData: Batched image data
    splitLabels: Batched label data
    """
    if data.shape[axis] % batchSize != 0: raise ValueError(f'Data of shape {data.shape} is not divisible along axis {axis} by {batchSize}')
    splits = data.shape[axis] // batchSize
    splitData = np.array(np.split(data, splits, axis))
    splitLabels = np.array(np.split(labels, splits, axis))
    return splitData, splitLabels

def oneHot(array, oneHotLength: int):
    oneHot = np.zeros((array.size, oneHotLength))
    oneHot[np.arange(array.size), array] = 1
    return oneHot.T

def plotLoss(losses, epochs: list[int]):
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs'), plt.ylabel('Loss')
    plt.plot(epochs, losses)
    plt.show()

# Save/Load Network Functions
def saveParameters(filePath: str, W1, b1, W2, b2, learningRate: float, learningRateClipping: int, epochs: int, decay: float, patience: int):
    if not os.path.exists(filePath): os.mkdir(filePath)

    specs = open(f'{filePath}/specs.txt', 'w')
    specs.writelines([
        f'Learning Rate          : {learningRate}\n',
        f'Learning Rate Clipping : {learningRateClipping}\n',
        f'Epochs                 : {epochs}\n',
        f'Decay                  : {decay}\n',
        f'Patience               : {patience}'
        ])
    specs.close()
    
    pd.DataFrame(W1).to_csv(f'{filePath}/W1.csv', index=False, sep=',')
    pd.DataFrame(b1).to_csv(f'{filePath}/b1.csv', index=False, sep=',')
    pd.DataFrame(W2).to_csv(f'{filePath}/W2.csv', index=False, sep=',')
    pd.DataFrame(b2).to_csv(f'{filePath}/b2.csv', index=False, sep=',')

def loadParameters(path: str):
    "Loads parameters from folder created with saveParameters function"
    parameters = []
    for filename in os.listdir(path):
        if 'csv' in filename: parameters.append(pd.read_csv(os.path.join(path, filename)).to_numpy())
        else:
            with open(os.path.join(path, filename)) as specSheet: specs = [float(line.rstrip()[25:]) for line in specSheet]
    return parameters[0], parameters[1], parameters[2], parameters[3], specs

# TrainingY (60000x1) | TrainingX (784x60000)
# TestingY (10000x1) | TestingX (784x10000)
trainingY, trainingX = getData(MNISTdata.trainingData)
testingY, testingX = getData(MNISTdata.testingData)
oneHotTrainingY = oneHot(trainingY, 10)
oneHotTestingY = oneHot(testingY, 10)

# Activation Functions + Derivatives
def ReLU(array): return np.maximum(array, 0)
def dReLU(array): return (array > 0) * 1

def LeakyReLU(array, decay: float = 0.1): return np.maximum(array * decay, array)
def dLeakyReLU(array, decay: float = 0.1): return np.where(array > 0, 1, decay)

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

def updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate: float):
    W1 = W1 - learningRate * dW1
    b1 = b1 - learningRate * db1
    W2 = W2 - learningRate * dW2
    b2 = b2 - learningRate * db2
    return W1, b1, W2, b2

def gradientDescent(W1, b1, W2, b2, X, Y, *, learningRate: float, learningRateClipping: int, epochs: int, decay: float, patience: int):
    originalLearningRate = learningRate
    bestLoss = float('inf')
    losses = [testModelLoss(trainingX, oneHotTrainingY, W1, b1, W2, b2)]

    for epoch in range(epochs):
        # Gradient Descent   
        for i in range(Y.shape[0]):
            x, y = X[i], Y[i]
            Z0, Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, x)
            dW1, db1, dW2, db2 = backPropagation(Z0, Z1, A1, A2, W2, y)
            W1, b1, W2, b2 = updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate)
        
        # Loss
        loss = testModelLoss(trainingX, oneHotTrainingY, W1, b1, W2, b2)
        losses.append(loss)
        print(f'Epoch {epoch + 1}{" " * (3 - math.floor(math.log10(epoch + 1)))} | Loss: {loss}')

        # Patience
        if loss <= bestLoss: bestLoss, patienceCount = loss, 0
        else: patienceCount += 1
        if patienceCount >= patience:
            print(f'\nSTOPPING TRAINING AT EPOCH {epoch + 1}')
            break

        learningRate = round(learningRate * decay, learningRateClipping)

    plotLoss(losses, range(epoch + 2))

    return W1, b1, W2, b2, originalLearningRate, learningRateClipping, epoch + 1, decay, patience

# Testing Functions
def testModelLoss(X, Y, W1, b1, W2, b2):
    Z0, Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, X)
    return CategoricalCrossEntropyLoss(A2, Y)

def testModelAccuracy(data, labels, W1, b1, W2, b2):
    Z0, Z1, A1, Z2, A2 = forwardPropagation(W1, b1, W2, b2, data)
    return np.sum(np.argmax(A2, axis=0) == labels) / labels.shape[0]

def CategoricalCrossEntropyLoss(Output, Y, epsilon: float = 1e-15):
    clippedOutput = np.clip(Output, epsilon, 1 - epsilon)
    return np.mean(-np.sum(Y * np.log(clippedOutput), axis=0))

# Model
batchedX, batchedY = batchData(trainingX, oneHotTrainingY, 15)

W1, b1, W2, b2 = initialiseParameters()
W1, b1, W2, b2, learningRate, learningRateClipping, epochs, decay, patience = gradientDescent(W1, b1, W2, b2, batchedX, batchedY, learningRate=0.0015, learningRateClipping=7, epochs=3, decay=0.965, patience=3)

print(f'\nAccuracy on training set {round(testModelAccuracy(trainingX, trainingY, W1, b1, W2, b2) * 100, 2)}')
print(f'Accuracy on testing set {round(testModelAccuracy(testingX, testingY, W1, b1, W2, b2) * 100, 2)}')

saveModel = input('Would you like to save the model? ').lower()
if saveModel == 'y' or saveModel == 'yes': saveParameters(input('Specify file path: '), W1, b1, W2, b2, learningRate, learningRateClipping, epochs, decay, patience)