import pandas as pd
import random, math
import MNISTdata

def getData(filePath):
    data = pd.read_csv(filePath).T

    # Separating Labels (First Row) From Pixel Data
    labels = pd.DataFrame(data.iloc[0,:]).T
    data.drop(['label'], axis=0, inplace=True)
    data = data/255

    return labels, data

oneHot = lambda number, listLength: [0 if i != number else 1 for i in range(listLength)]

"""
Get Column ↓
...iloc[:, 0]
"""

def initialiseParameters():
    # Generate 16x784 Weight Matrix & 16x1 Bias Matrix
    W1 = pd.DataFrame([[random.uniform(-0.5, 0.5) for column in range(784)] for row in range(16)])
    B1 = pd.DataFrame([random.uniform(-0.5, 0.5) for row in range(16)], columns=[0])

    # Generate 10x16 Weight Matrix & 10x1 Bias Matrix
    W2 = pd.DataFrame([[random.uniform(-0.5, 0.5) for column in range(16)] for row in range(10)])
    B2 = pd.DataFrame([random.uniform(-0.5, 0.5) for row in range(10)], columns=[0])

    return W1, B1, W2, B2

def ReLU(DataFrame): return DataFrame * (DataFrame > 0)
def dRelu(DataFrame): return 1 * (DataFrame > 0)

def LeakyReLU(DataFrame): return DataFrame * (DataFrame > 0) + (0.1 * DataFrame * (DataFrame < 0))

def forwardPropagation(Z0, W1, W2, B1, B2):
    # Apply weights and biases with ReLU Activation
    Z1 = W1 @ Z0 + B1
    A1 = LeakyReLU(Z1)
    Z2 = W2 @ A1 + B2

    return Z1, A1, Z2

# def backwardPropagation(Z1, W1, B1, A1, Z2, W2, B2, A2, Y):

labels, data = getData(MNISTdata.testingData)

# W1, B1, W2, B2 = initialiseParameters()
# Z0 = pd.DataFrame(data.iloc[:, 0]).reset_index(drop=True)
# Z1, A1, Z2, A2 = forwardPropagation(Z0, W1, W2, B1, B2)



# backwardPropagation(Z1, W1, B1, A1, Z2, W2, B2, A2)