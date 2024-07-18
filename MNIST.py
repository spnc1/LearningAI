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
    b1 = pd.DataFrame([random.uniform(-0.5, 0.5) for row in range(16)], columns=[0])

    # Generate 10x16 Weight Matrix & 10x1 Bias Matrix
    W2 = pd.DataFrame([[random.uniform(-0.5, 0.5) for column in range(16)] for row in range(10)])
    b2 = pd.DataFrame([random.uniform(-0.5, 0.5) for row in range(10)], columns=[0])

    return W1, b1, W2, b2

def ReLU(DataFrame): return DataFrame * (DataFrame > 0)
def LeakyReLU(DataFrame): return DataFrame * (DataFrame > 0) + (0.1 * DataFrame * (DataFrame < 0))
def dRelu(DataFrame): return 1 * (DataFrame > 0)
def dLeakyReLU(DataFrame): return 1 * (DataFrame > 0) + (0.1 * (DataFrame < 0))

def SoftMax(DataFrame):
    dataFrameExp = DataFrame.copy()

    # By column to grab integers not series
    for col in DataFrame.columns: dataFrameExp[col] = DataFrame[col].apply(lambda x: math.exp(x))
    sumExp = dataFrameExp.sum(axis=0)
    return dataFrameExp.div(sumExp, axis=1)
def stableSoftMax(DataFrame):
    # Subtract the max value for numerical stability
    dataFrameMax = DataFrame.max(axis=0)
    dataFrameExp = DataFrame.copy()

    for col in DataFrame.columns: dataFrameExp[col] = DataFrame[col].apply(lambda x: math.exp(x - dataFrameMax[col]))
    sumExp = dataFrameExp.sum(axis=0)
    return dataFrameExp.div(sumExp, axis=1)
def dSoftMax(DataFrame):
    s = SoftMax(DataFrame)
    return s * (1 - s)
def dStableSoftMax(DataFrame):
    s = stableSoftMax(DataFrame)
    return s * (1 - s)

def DataFrameOneHot(labels): return pd.DataFrame([oneHot(label, 10) for label in labels.values[0]]).T

def forwardPropagation(Z0, W1, b1, W2, b2):
    # Apply weights and biases with ReLU Activation
    Z1 = W1 @ Z0 + b1
    A1 = LeakyReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = stableSoftMax(Z2)
    return Z1, A1, Z2, A2

def backwardPropagation(Z0, Z1, A1, Z2, W2, A2, Y):
    # dZ2 10x1 | dW2 10x16
    dZ2 = (2 * (A2 - Y) / len(A2)) * dStableSoftMax(Z2)
    dW2 = dZ2 @ A1.T
    
    # dZ1 16x1 | dW1 16x784
    dZ1 = W2.T @ dZ2 * dLeakyReLU(Z1)
    dW1 = dZ1 @ Z0.T

    return dW1, dZ1, dW2, dZ2

def updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

labels, data = getData(MNISTdata.trainingData)
testLabels, testData = getData(MNISTdata.testingData)
labels = DataFrameOneHot(labels)
testLabels = DataFrameOneHot(testLabels)

def gradientDescent(data, labels, alpha, epochs):
    W1, b1, W2, b2 = initialiseParameters()
    for epoch in range(epochs):
        for i in range(len(labels.columns)):
            Y = pd.DataFrame(labels.iloc[:,i]).reset_index(drop=True)
            Y.columns = [0]
            Z0 = pd.DataFrame(data.iloc[:,i]).reset_index(drop=True)
            Z0.columns = [0]

            Z1, A1, Z2, A2 = forwardPropagation(Z0, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backwardPropagation(Z0, Z1, A1, Z2, W2, A2, Y)
            W1, b1, W2, b2 = updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            
            if i % 1000 == 0:
                print(f'Epoch: {epoch + 1} | Image: {i}\nPredicted {A2.idxmax().iloc[0]} for {Y.idxmax().iloc[0]}\n')
    
    print('Training Complete')
    return W1, b1, W2, b2

def testModel(testLabels, testData, W1, b1, W2, b2):
    correct = 0
    for i in range(len(testLabels.columns)):
        Y = pd.DataFrame(testLabels.iloc[:,i]).reset_index(drop=True)
        Y.columns = [0]
        Z0 = pd.DataFrame(testData.iloc[:,i]).reset_index(drop=True)
        Z0.columns = [0]
        
        Z1, A1, Z2, A2 = forwardPropagation(Z0, W1, b1, W2, b2)
        correct += 1 if A2.idxmax().iloc[0] == Y.idxmax().iloc[0] else 0
    
    return correct/len(testLabels.columns)

W1, b1, W2, b2 = gradientDescent(data, labels, 0.075, epochs=1)
print(f'\nAccuracy on testing data: {testModel(testLabels, testData, W1, b1, W2, b2) * 100}%')