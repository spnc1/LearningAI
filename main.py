import csv, random, time, os, math

if os.name == 'nt': os.system('cls')
elif os.name == 'posix': os.system('clear')

data = []
answers = []

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

def activationLayer(input: list[int], activationFunction: str) -> list:
    """
    Takes a layer and applies a specified activation function to the entire layer, outputting the activated layer

    Parameters
    ----------
    input               : Unactivated layer needing to be passed through an activation layer
    activationFunction  : Choice of activation function (ReLU, Leaky ReLU, Logistic Sigmoid, Softmax)

    Returns
    -------
    Layer activated with specified function
    """

    if activationFunction == 'ReLU':
        for i in range(len(input)): input[i] = max(0, input[i])
    
    elif activationFunction == 'Leaky ReLU':
        for i in range(len(input)): input[i] = max(0.01*input[i], input[i])
    
    elif activationFunction == 'Logistic Sigmoid':
        for i in range(len(input)): input[i] = 1/(1+math.exp(-input[i]))
    
    elif activationFunction == 'Softmax':
        allExponentials = []
        allExponentialsSum = 0
        for i in range(len(input)):
            ea = math.exp(input[i])
            allExponentials.append(ea)
            allExponentialsSum += ea
        for i in range(len(input)): input[i] = allExponentials[i]/allExponentialsSum

    return input

class Layer():
    """
    A class to represent a hidden or output layer of a neural network

    Attributes
    ----------
    nInputs             : The amount of neurons in the previous layer (Size of input layer)
    nNeurons            : The amount of Neurons in the current layer (Size of this layer)
    weights             : The weight matrix
    biases              : The bias vector
    output              : The layer output

    Methods
    -------
    forwardPropagation  : Takes an input layer and an activation function choice and saves the output to self.output
    """

    def __init__(self, nInputs: int, nNeurons: int):

        # Generate a matrix of weights with nInputs width and nNeurons height
        weights = []
        for i in range(nNeurons):
            weightsBatch = []
            for x in range(nInputs): weightsBatch.append(0.025*random.uniform(-0.5,0.5))
            weights.append(weightsBatch)
        
        # Generate a vector of biases with length nNeurons
        biases = []
        for i in range(nNeurons): biases.append(random.uniform(-0.5,0.5))

        # Make them accessible by the class object
        self.nInputs = nInputs
        self.nNeurons = nNeurons
        self.weights = weights
        self.biases = biases
    
    def forwardPropagation(self, inputs: list[int], activationFunction: str):
        """
        Method for forward propagation of the neural network at the current layer

        Parameters
        ----------
        inputs              : The input layer
        activationFunction  : Choice of activation function (ReLU, Leaky ReLU, Logistic Sigmoid, Softmax)

        Outputs
        --------
        self.output         : The vector output of the layer, accessible with self.output
        """
        output = []
        
        for i in range(self.nNeurons):
            outputBatch = 0
            for x in range(len(inputs)): outputBatch += inputs[x] * self.weights[i][x]
            output.append(outputBatch + self.biases[i])
        
        self.output = activationLayer(output, activationFunction)

inputLayer = [255,84,0,0,15]

h1 = Layer(len(inputLayer),5)
h1.forwardPropagation(inputLayer, 'Logistic Sigmoid')

h2 = Layer(len(h1.output),5)
h2.forwardPropagation(h1.output, 'ReLU')

outputLayer = Layer(len(h2.output), 2)
outputLayer.forwardPropagation(h2.output, 'Softmax')

print('Input Layer')
print(inputLayer)
print('\nHidden Layer 1')
for i in range(len(h1.weights)): print(f'Neuron {i}: {h1.weights[i]}x + {h1.biases[i]}')
print('\nHidden Layer 2')
for i in range(len(h2.weights)): print(f'Neuron {i}: {h2.weights[i]}x + {h2.biases[i]}')
print(f'\nOutput: {outputLayer.output}\n')