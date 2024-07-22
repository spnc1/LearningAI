import ProjectMNIST.MNIST as nn
import MNISTdata

batchSize = 10

hyperParametersSet = [
    {'learningRate': 0.0017, 'learningRateClipping': 8, 'decay': 0.94, 'patience': 10, 'epochs': 10, 'batchSize': batchSize},
    {'learningRate': 0.0017, 'learningRateClipping': 8, 'decay': 0.95, 'patience': 10, 'epochs': 10, 'batchSize': batchSize},
    {'learningRate': 0.0017, 'learningRateClipping': 8, 'decay': 0.96, 'patience': 10, 'epochs': 10, 'batchSize': batchSize}
]

trainX, trainY, batchedTrainX, batchedTrainY = nn.simpleGetData(MNISTdata.trainingData, batchSize)
testX, testY, batchedTestX, batchedTestY = nn.simpleGetData(MNISTdata.testingData, 1)

for hyperParameters in hyperParametersSet:
    parameters = nn.initialiseParameters()
    layers, parameters, netStats = nn.gradientDescent(batchedTrainX, batchedTrainY, parameters, hyperParameters, testX, testY)

    trainingAccuracy = nn.testModelAccuracy(parameters, trainX, trainY)
    testingAccuracy = nn.testModelAccuracy(parameters, testX, testY)
    print(f'Training Accuracy: {trainingAccuracy}')
    print(f'Testing Accuracy: {testingAccuracy}')

    nn.displayImage(parameters, batchedTestX, batchedTestY)
    nn.displayLossCurve(netStats)

    nn.saveNetwork('C:/Users/coole/Documents/AI/AllPurposeNN/example', parameters, hyperParameters, netStats)

parameters = nn.loadNetwork('C:/Users/coole/Documents/AI/AllPurposeNN/example/exampleModel')

trainingAccuracy = nn.testModelAccuracy(parameters, trainX, trainY)
testingAccuracy = nn.testModelAccuracy(parameters, testX, testY)
print(f'Training Accuracy: {trainingAccuracy}')
print(f'Testing Accuracy: {testingAccuracy}')

nn.displayImage(parameters, batchedTestX, batchedTestY)
nn.displayLossCurve(netStats)