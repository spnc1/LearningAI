import csv, random, time, os, math


# Tool Functions
def readCsv(filepath: str, targetArray: list, answersTargetArray: list):
    """
    Takes a file path of training data and saves it to 2 arrays, a data and an answers array.
    IGNORES HEADERS AND MUST BE FLOATS

    Paramaters
    ----------
    filepath            : File path of training/testing data to be read
    targetArray         : Name of arrray for training/testing data
    answersTargetArray  : Name of array for training/testing data answers
    """

    with open(filepath, mode="r") as csv_file:
        reader = csv.reader(csv_file)

        # skips header
        next(reader)

        # QUOTE_NONNUMERIC converts all non-quoted elements to floats
        reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)

        for item in reader:
            targetArray.append(item[1:])
            answersTargetArray.append(item[0])
def MSE(output, expected):
    cost = 0
    for outputValue, expectedValue in zip(output, expected): cost += (outputValue - expectedValue) ** 2
    return cost
oneHot = lambda number, listLength: [0 if i != number else 1 for i in range(listLength)]

learningRate = 0.001

class Layer():
    def __init__(self, nInputs: int, nNeurons: int, activationFunction: str = ''):
        """
        A class to represent a hidden or output layer of a neural network

        Parameters
        ----------
        nInputs             : The amount of neurons in the previous layer (Size of input layer)
        nNeurons            : The amount of Neurons in the current layer (Size of this layer)
        activationFunction  : Activation function applied

        Attributes
        ----------
        nInputs             : The amount of neurons in the previous layer (Size of input layer)
        nNeurons            : The amount of Neurons in the current layer (Size of this layer)
        weights             : The weight matrix
        biases              : The bias vector
        output              : The layer output
        activationFunction  : Activation function applied

        Methods
        -------
        forwardPropagation  : Takes an input layer and applies a forward pass to the layer, saving the output to self.output
        """

        # Generate a matrix of weights with nInputs width and nNeurons height
        weights = [[0.025*random.uniform(-0.5,0.5) for x in range(nInputs)] for y in range(nNeurons)]
        
        # Generate a vector of biases with length nNeurons
        biases = [random.uniform(-0.5,0.5) for i in range(nNeurons)]

        # Make them accessible by the class object
        self.nInputs = nInputs
        self.nNeurons = nNeurons
        self.weights = weights
        self.biases = biases
        self.activationFunction = activationFunction.lower()
    
    def forwardPropagation(self, inputs: list[float]):
        """
        Method for forward propagation of the neural network at the current layer

        Parameters
        ----------
        inputs              : The input layer

        Outputs
        --------
        self.output         : The vector output of the layer, accessible with self.output
        """
        
        # Run data through layer
        output = []
        for n in range(self.nNeurons):
            outputBatch = 0
            for value, weight in zip(inputs, self.weights[n]): outputBatch += value * weight
            output.append(outputBatch + self.biases[n])

        # Activation Function Selection
        match self.activationFunction:
            case '': self.output = output

            case 'relu': self.output = [i if i > 0 else 0 for i in output]

            case 'leaky relu': self.output = [i if i > 0 else 0.01*i for i in output]

            case 'logistic sigmoid': self.output = [1/(1+math.exp(i)) for i in output]

            case 'softmax':
                eSum = 0
                for value in output: eSum += math.exp(value)
                self.output = [math.exp(value)/eSum for value in output]

    # Back Propagation not working

    def backwardPropagation(self, previousLayer, expectedValues):
        dB = []
        for n, output in enumerate(self.output): dB.append(2*(output-expectedValues[n]))
        dW = [neuronDeriv * previousNeuron for neuronDeriv, previousNeuron in zip(dB, previousLayer)]

        # Update Weights
        for i, (weightDeriv, weightSet) in enumerate(zip(dW, self.weights)):
            for j, weight in enumerate(weightSet): self.weights[i][j] = weight - learningRate * weightDeriv
        
        for i, (biasDeriv, bias) in enumerate(zip(dB, self.biases)): self.biases[i] = bias - learningRate * biasDeriv

    def outputLayerBackPropagation(self, previousLayer, expectedValues):
        match self.activationFunction:
            case '':
                dBiases = [2 * (outputValue - expectedValue) for outputValue, expectedValue in zip(self.output, expectedValues)]
                dWeights = [dBias * previousNeuron for dBias, previousNeuron in zip(dBiases, previousLayer)]
            
            case 'relu':
                dBiases = [2 * (outputValue - expectedValue) if 2 * (outputValue - expectedValue) < 0 else 0 for outputValue, expectedValue in zip(self.output, expectedValues)]
                dWeights = [2 * (outputValue - expectedValue) if 2 * (outputValue - expectedValue) * previousNeuron < 0 else 0 for previousNeuron, (outputValue, expectedValue) in zip(previousLayer, zip(self.output, expectedValues))]

            case 'leaky relu':
                dBiases = [1 if 2 * (outputValue - expectedValue) > -0.1 else 0 for outputValue, expectedValue in zip(self.output, expectedValues)]
                dWeights = [1 if 2 * (outputValue - expectedValue) * previousNeuron > 0 else -0.1 for previousNeuron, (outputValue, expectedValue) in zip(previousLayer, zip(self.output, expectedValues))]
            
        # Update Weights & Biases
        for i, (dWeight, weightSet) in enumerate(zip(dWeights, self.weights)):
            for j, weight in enumerate(weightSet): self.weights[i][j] = weight - learningRate * dWeight
        for i, (dBias, bias) in enumerate(zip(dBiases, self.biases)): self.biases[i] = bias - learningRate * dBias

    def getParameters(self): return self.weights, self.biases