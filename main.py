import ProjectMNIST.MNIST as nn
import MNISTdata, os

trainX, trainY = nn.getData(MNISTdata.trainingData)
testX, testY = nn.getData(MNISTdata.testingData)
testY = nn.oneHot(testY, 10)

hyperParametersSet = [
    {'learningRate': 0.0017, 'learningRateClipping': 8, 'decay': 0.949, 'patience': 10, 'epochs': 100, 'batchSize': 10}
]

os.system('cls')

parameters = nn.loadNetwork('C:/Users/coole/Documents/AI/ModelTesting/BestModels/PB(89.33)/Model')
print(nn.testModelAccuracy(parameters, testX, testY))