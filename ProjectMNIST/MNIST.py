import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, random


### Network Tools ###


def getData(filePath: str) -> tuple[np.ndarray]:
    """
    Retrieves MNIST data from a complete filepath

    Inputs
    ------
    filePath: Complete file path to MNIST data

    Outputs
    -------
    data: The X data (Images)
    labels: The Y data (Labels)
    """

    data = pd.read_csv(filePath).to_numpy().T
    labels = data[0]
    return np.delete(data, 0, axis=0) / 255, labels

def batchData(data: np.ndarray, labels: np.ndarray, batchSize: int) -> tuple[np.ndarray]:
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
    
    if data.shape[1] % batchSize != 0: raise ValueError(f'Data of shape {data.shape} is not divisible along axis {1} by {batchSize}')
    splits = data.shape[1] // batchSize
    splitData = np.array(np.split(data, splits, 1))
    splitLabels = np.array(np.split(labels, splits, 1))
    return splitData, splitLabels

def oneHot(array: np.ndarray, oneHotLength: int) -> np.ndarray:
    """
    One hot encoding tool

    Inputs
    ------
    array: NDArray to be one hot encoded
    oneHotLength: Rows required for one hot encoding (10 classifications, 10 oneHotLength)

    Outputs
    -------
    oneHot.T: One hot encoded NDArray
    """
    
    oneHot = np.zeros((array.size, oneHotLength))
    oneHot[np.arange(array.size), array] = 1
    return oneHot.T

def splitIntoImages(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray]:
    """
    Splits data set into individual images for displaying

    Inputs
    ------
    X: Image data
    Y: Labels

    Outputs
    -------
    splitX: Split image data by image
    splitY: Split labels by image
    """
    
    splits = X.shape[1]
    splitX = np.array(np.split(X, splits, 1))
    splitY = np.array(np.split(Y, splits, 1))
    return splitX, splitY

def displayImage(parameters: dict[str, np.ndarray], X: np.ndarray, Y: np.ndarray) -> None:
    """
    Finds a random sample from training set and shows network prediction and the image

    Inputs
    ------
    parameters: Weights and biases
    X: Batched images with batchSize 1 | Images passed through splitIntoImages()
    Y: Batched labels with batchSize 1 | Labels passed through splitIntoImages()
    """
    
    randomExample = random.randint(0, Y.shape[0])
    x, y = X[randomExample], Y[randomExample]
    layers = forwardPropagation(x, parameters)

    print(f"\nExample #{randomExample + 1}")
    print(f"Predicted {np.argmax(layers['A2'])}")
    print(f"Answer {np.argmax(y)}")

    plt.gray()
    plt.imshow(x.reshape((28, 28)) * 255, interpolation='nearest')
    plt.show()

def displayLossCurve(netStats: dict[str, float | int]) -> None:
    """
    Displays Categorical Cross-Entropy Loss on testing set against Epochs

    Inputs
    ------
    netStats: netStats output from Gradient Descent
    """
    
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs'), plt.ylabel('Loss')
    plt.plot(range(netStats['trainingEpochs'] + 1), netStats['losses'])
    plt.show()

def simpleGetData(filePath: str, batchSize: int) -> tuple[np.ndarray]:
    """
    Import all the data you need for training OR testing

    Inputs
    ------
    filePath: The full path to the MNIST data set
    batchSize: 0 for no batching | n for batches of size n
    IMPORTANT: Dataset MUST be evenly divisble by batches of size n

    Outputs
    -------
    data: The X data (images)
    labels: The Y data (classification)
    splitData: The X data split into batches of size batchSize
    labels: The Y data split into batches of size batchSize
    """
    
    data = pd.read_csv(filePath).to_numpy().T
    labels = oneHot(data[0], 10)
    data = np.delete(data, 0, axis=0) / 255
    
    if batchSize != 0:
        if data.shape[1] % batchSize != 0: raise ValueError(f'Data of shape {data.shape} is not divisible along axis {1} by {batchSize}')
        splits = data.shape[1] // batchSize
        splitData = np.array(np.split(data, splits, 1))
        splitLabels = np.array(np.split(labels, splits, 1))
        return data, labels, splitData, splitLabels
    
    return data, labels


### Save/Load Network Functions ###


def saveNetwork(filePath: str, parameters: dict[str, np.ndarray], hyperParameters: dict[str, float | int], netStats: dict[str, float | int]) -> None:
    """
    Saves a network to a specified folder and creates one in parent folder if with specified name if none found

    Inputs
    ------
    filePath: Complete file path to target folder
    parameters: Weights and biases
    hyperParameters: User adjusted parameters for the network
    netStats: Network statistics
    IMPORTANT: All inputs besides filePath are returned by gradientDescent()
    """

    if not os.path.exists(filePath): os.mkdir(filePath)

    file = open(f'{filePath}/NetworkInfo.txt', 'w')
    file.writelines([
            f"Testing Set Accuracy   : {round(netStats['testAccuracy'] * 100, 2)}\n",
            f"Initial Loss           : {netStats['losses'][0]}\n",
            f"Final Loss             : {netStats['losses'][-1]}\n",
            f"Learning Rate          : {hyperParameters['learningRate']}\n",
            f"Learning Rate Clipping : {hyperParameters['learningRateClipping']}\n",
            f"Set Epochs             : {hyperParameters['epochs']}\n",
            f"Completed Epochs       : {netStats['trainingEpochs']}\n",
            f"Learning Rate Decay    : {hyperParameters['decay']}\n",
            f"Network Patience       : {hyperParameters['patience']}\n",
            f"Batch Size             : {hyperParameters['batchSize']}"
        ])
    file.close()

    file = open(f'{filePath}/epochloss.txt', 'w')
    for loss in netStats['losses']: file.write(f'{loss}\n')
    file.close()
    os.mkdir(f'{filePath}/Model')
    pd.DataFrame(parameters['W1']).to_csv(f'{filePath}/Model/W1.csv', index=False, sep=',')
    pd.DataFrame(parameters['b1']).to_csv(f'{filePath}/Model/b1.csv', index=False, sep=',')
    pd.DataFrame(parameters['W2']).to_csv(f'{filePath}/Model/W2.csv', index=False, sep=',')
    pd.DataFrame(parameters['b2']).to_csv(f'{filePath}/Model/b2.csv', index=False, sep=',')

def loadNetwork(filePath: str) -> dict[str, np.ndarray]:
    """
    Takes a filepath to a model and imports the parameters
    
    Inputs
    ------
    filePath: Complete file path to model folder

    Outputs
    -------
    parameters: Weights and biases
    """
    parameters = {}
    for filename in os.listdir(filePath):
        if 'csv' in filename: parameters[filename.split('.')[0]] = pd.read_csv(os.path.join(filePath, filename)).to_numpy()
    return parameters


### Activation Functions + Derivatives ###


def ReLU(array: np.ndarray): return np.maximum(array, 0)
def dReLU(array: np.ndarray): return (array > 0) * 1

def LeakyReLU(array: np.ndarray, decay: float = 0.1): return np.maximum(array * decay, array)
def dLeakyReLU(array: np.ndarray, decay: float = 0.1): return np.where(array > 0, 1, decay)

def SoftMax(array: np.ndarray): return np.exp(array) / sum(np.exp(array), axis=0)
def StableSoftMax(array: np.ndarray):
    exp = np.exp(array - np.max(array, axis=0))
    return exp / np.sum(exp, axis=0)


### Testing Functions ###


def testModelLoss(parameters: dict[str, np.ndarray], X: np.ndarray, Y: np.ndarray) -> float:
    """
    Tests the model's loss over the entirety of the given data set using Categorical Cross-Entropy Loss function

    Inputs
    ------
    parameters: Weights and biases
    X: Image data
    Y: Labels

    Outputs
    -------
    Loss: The Categorical Cross-Entropy
    """
    layers = forwardPropagation(X, parameters)
    return CategoricalCrossEntropyLoss(layers['A2'], Y)

def testModelAccuracy(parameters: dict[str, np.ndarray], X: np.ndarray, Y: np.ndarray) -> float:
    """
    Tests the model's accuracy over the entirety of the given data set

    Inputs
    ------
    parameters: Weights and biases
    X: Image data
    Y: Labels

    Outputs
    -------
    Accuracy: Percentage correctly predicted
    """
    layers = forwardPropagation(X, parameters)
    return np.sum(np.argmax(layers['A2'], axis=0) == np.argmax(Y, axis=0)) / Y.shape[1]

def CategoricalCrossEntropyLoss(predictedY: np.ndarray, Y: np.ndarray, epsilon: float = 1e-15) -> float:
    """
    Calculates loss given predictions and labels

    Inputs
    ------
    predictedY: Output layer of model
    Y: Labels
    epsilon: Output clipping

    Outputs
    -------
    Loss: Loss given by Categorical Cross-Entropy Loss function
    """
    clippedOutput = np.clip(predictedY, epsilon, 1 - epsilon)
    return np.mean(-np.sum(Y * np.log(clippedOutput), axis=0))


### Network ###


def initialiseParameters() -> dict[str, np.ndarray]:
    """
    Generates\n
    Layer 1 16x784 Weight Matrix & 16x1 Bias Matrix\n
    Layer 2 10x16 Weight Matrix & 10x1 Bias Matrix
    """

    # Generate 
    W1 = np.random.uniform(-0.5, 0.5, (16, 784))
    b1 = np.random.uniform(-0.5, 0.5, (16, 1))

    # Generate 
    W2 = np.random.uniform(-0.5, 0.5, (10, 16))
    b2 = np.random.uniform(-0.5, 0.5, (10, 1))

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def forwardPropagation(Z0: np.ndarray, parameters: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Takes first layer and propagates through the network

    Inputs
    ------
    Z0: First layer
    parameters: Weights and biases

    Outputs
    -------
    layers: Returns all layers in a dictionary
    """

    Z1 = parameters['W1'] @ Z0 + parameters['b1']
    A1 = LeakyReLU(Z1, 0.1)
    Z2 = parameters['W2'] @ Z1 + parameters['b2']
    A2 = StableSoftMax(Z2)
    return {'Z0': Z0, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

def backPropagation(layers: dict[str, np.ndarray], parameters: dict[str, np.ndarray], Y: np.ndarray) -> dict[str, np.ndarray]:
    """
    Takes all the layers, classifications and parameters and calculates the gradient matrices

    Inputs
    ------
    layers: All the layers returned by forwardPropagation()
    parameters: Weights and biases
    Y: Labels

    Outputs
    -------
    gradients: Returns all gradient matrices in a dictionary
    """
    
    batchSize = layers['A2'].shape[1]

    dZ2 = (layers['A2'] - Y)
    dW2 = (dZ2 @ layers['A1'].T) / batchSize
    db2 = np.sum(dZ2, axis=1, keepdims=True) / batchSize

    dZ1 = ((parameters['W2'].T @ dZ2) * dLeakyReLU(layers['Z1'], 0.1))
    dW1 = (dZ1 @ layers['Z0'].T) / batchSize
    db1 = np.sum(dZ1, axis=1, keepdims=True) / batchSize

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

def updateParameters(parameters: dict, gradients: dict, learningRate: float):
    """
    Updates all the weights and biases by a factor determined by the learning rate

    Inputs
    ------
    parameters: Weights and biases
    gradients: Gradient matrices returned by backPropagation()
    learningRate: Hyperparameter for step magnitudes towards local minima

    Output
    ------
    parameters: Updated weights and biases
    """
    parameters['W1'] -= learningRate * gradients['dW1']
    parameters['b1'] -= learningRate * gradients['db1']
    parameters['W2'] -= learningRate * gradients['dW2']
    parameters['b2'] -= learningRate * gradients['db2']

    return parameters

def gradientDescent(trainingX: np.ndarray, trainingY: np.ndarray, parameters: dict[str, np.ndarray], hyperParameters: dict[str, np.ndarray], testingX, testingY) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float | int]]:
    """
    Performs gradient descent for network
    
    Inputs
    ------
    trainingX: Batched Image Data
    trainingY: Batched, One Hot Encoded Labels
    parameters: Weights and biases
    
    OPTIONAL
    testingX: Original Image Data
    testingY: One Hot Encoded Labels

    Outputs
    -------
    layers: The layers of the network (Mainly for debugging)
    parameters: Weights and Biases after gradient descent
    netStats: Neural Network Information
    """

    initialLearningRate, bestLoss = hyperParameters['learningRate'], float('inf')
    learningRate = initialLearningRate
    losses = [testModelLoss(parameters, testingX, testingY)]
    patienceCount = 0

    for epoch in range(hyperParameters['epochs']):
        for i in range(trainingY.shape[0]):
            x, y = trainingX[i], trainingY[i]

            layers = forwardPropagation(x, parameters)
            gradients = backPropagation(layers, parameters, y)
            parameters = updateParameters(parameters, gradients, learningRate)

        
        loss = testModelLoss(parameters, testingX, testingY)
        losses.append(loss)

        # Patience
        if loss <= bestLoss: bestLoss, patienceCount = loss, 0
        else: patienceCount += 1
        if patienceCount >= hyperParameters['patience']: break

        # Learning Rate Decay
        learningRate = round(learningRate * hyperParameters['decay'], hyperParameters['learningRateClipping'])

    netStats = {'testAccuracy': testModelAccuracy(parameters, testingX, testingY), 'losses': losses, 'trainingEpochs': epoch + 1}

    return layers, parameters, netStats