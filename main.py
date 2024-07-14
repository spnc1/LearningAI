import csv, random, time, os, math

if os.name == 'nt': os.system('cls')
elif os.name == 'posix': os.system('clear')

data = []
y = []

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
oneHot = lambda number, listLength: [0 if i != number else 1 for i in range(listLength)]

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
        
        output = []
        for n in range(self.nNeurons):
            outputBatch = 0
            for value, weight in zip(inputs, self.weights[n]): outputBatch += value * weight
            output.append(outputBatch + self.biases[n])

        match self.activationFunction:
            case '': self.output = output

            case 'relu': self.output = [i if i > 0 else 0 for i in output]

            case 'leaky relu': self.output = [i if i > 0 else 0.01*i for i in output]

            case 'logistic sigmoid': self.output = [1/(1+math.exp(i)) for i in output]

            case 'softmax':
                eSum = 0
                for value in output: eSum += math.exp(value)
                self.output = [math.exp(value)/eSum for value in output]

    def backwardPropagation(self, previousLayer):
        dB = []
        for n, output in enumerate(self.output): dB.append(2*(output-y[n]))
        dW = [neuronDeriv * previousNeuron for neuronDeriv, previousNeuron in zip(dB, previousLayer)]

        # Update Weights
        for i, (weightDeriv, weightSet) in enumerate(zip(dW, self.weights)):
            for j, weight in enumerate(weightSet): self.weights[i][j] = weight - LR * weightDeriv
        
        for i, (biasDeriv, bias) in enumerate(zip(dB, self.biases)):
            self.biases[i] = bias - LR * biasDeriv


# XOR gate truth table
truthTable = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]
data = [scenario[0:2] for scenario in truthTable]
y = [scenario[1] for scenario in truthTable]
inputLayer = data[1]
y = oneHot(y[1], 2)

LR = 0.05

outputLayer = Layer(len(inputLayer), 2)
for weight, bias in zip(outputLayer.weights, outputLayer.biases): print(f'Weights: {weight} | Bias: {bias}')

for i in range(10000):
    outputLayer.forwardPropagation(inputLayer)
    cost = 0
    for i, output in enumerate(outputLayer.output): cost += (output - y[i]) ** 2
    outputLayer.backwardPropagation(inputLayer)

print()
for weight, bias in zip(outputLayer.weights, outputLayer.biases): print(f'Weights: {weight} | Bias: {bias}')
print()

print(f'Output: {outputLayer.output}, Expected Answer: {y}\n')
print(f'Cost: {cost}')