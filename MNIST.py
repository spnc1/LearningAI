import pandas as pd
import random
import MNISTdata

def getData(filePath):
    data = pd.read_csv(filePath).T

    # Separating Labels (First Row) From Pixel Data
    labels = data.iloc[0,:]
    data.drop(['label'], axis=0, inplace=True)
    data = data/255

    return labels, data

"""
Get Column ↓
...iloc[:, 0]
"""

def initialiseParameters():
    # Generate 16x784 Weight Matrix & 16x1 Bias Matrix
    W1 = [[random.uniform(-0.5, 0.5) for column in range(784)] for row in range(16)]
    B1 = [random.uniform(-0.5, 0.5) for row in range(16)]

    # Generate 10x16 Weight Matrix & 10x1 Bias Matrix
    W2 = [[random.uniform(-0.5, 0.5) for column in range(16)] for row in range(10)]
    B2 = [random.uniform(-0.5, 0.5) for row in range(10)]

    return W1, B1, W2, B2

def ReLU(DataFrame): return DataFrame * (DataFrame > 0)

def dRelu(DataFrame): return 1 * (DataFrame > 0)

def LeakyReLU(DataFrame): return DataFrame * (DataFrame > 0) + (0.1 * DataFrame * (DataFrame < 0))

def forwardPropagation(Z0, W1, W2, B1, B2):
    # Apply weights and biases with ReLU Activation
    Z1 = pd.DataFrame(W1 @ Z0 + B1)
    A1 = LeakyReLU(Z1)

    # B2 Shape error?
    Z2 = pd.DataFrame(W2 @ A1 + B2)

    return Z1, A1

labels, data = getData(MNISTdata.testingData)

W1, B1, W2, B2 = initialiseParameters()
Z0 = data.iloc[:, 0]
Z1, A1 = forwardPropagation(Z0, W1, W2, B1, B2)