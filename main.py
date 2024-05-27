import csv, random, time, os, math

if os.name == 'nt': os.system('cls')
elif os.name == 'posix': os.system('clear')

data = []
answers = []

def readCsv(filename, targetArray, answersTargetArray):
    with open(filename, mode="r") as csv_file: #"r" represents the read mode
        reader = csv.reader(csv_file)
        next(reader)
        reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        for item in reader:
            targetArray.append(item[1:])
            answersTargetArray.append(item[0])

def forwardPropagation(input, activationFunction):
    if activationFunction == 'ReLU':
        for i in range(len(input)):
            input[i] = max(0, input[i])
    
    elif activationFunction == 'Leaky ReLU':
        for i in range(len(input)):
            input[i] = max(0.01*input[i], input[i])
    
    elif activationFunction == 'Logistic Sigmoid':
        for i in range(len(input)):
            input[i] = 1/(1+math.exp(-input[i]))
    
    elif activationFunction == 'Softmax':
        allExponentials = []
        allExponentialsSum = 0
        for i in range(len(input)):
            ea = math.exp(input[i])
            allExponentials.append(ea)
            allExponentialsSum += ea
        for i in range(len(input)):
            input[i] = allExponentials[i]/allExponentialsSum

    return input

class Layer():
    def __init__(self, nInputs, nNeurons):

        # generate matrix of weights
        weights = []
        for i in range(nNeurons):
            weightsBatch = []
            for x in range(nInputs):
                weightsBatch.append(0.025*random.uniform(-0.5,0.5))
            weights.append(weightsBatch)
        
        # generate vector of biases
        biases = []
        for i in range(nNeurons):
            biases.append(random.uniform(-0.5,0.5))

        # store values
        self.nInputs = nInputs
        self.nNeurons = nNeurons
        self.weights = weights
        self.biases = biases
    
    def forwardPropagation(self, inputs, activationFunction):
        output = []
        
        for i in range(self.nNeurons):
            outputBatch = 0
            for x in range(len(inputs)):
                outputBatch += inputs[x] * self.weights[i][x]
            output.append(outputBatch + self.biases[i])

        self.output = forwardPropagation(output, activationFunction)

inputLayer = [255,84,0,0,15]

h1 = Layer(len(inputLayer),10)
h1.forwardPropagation(inputLayer, 'Logistic Sigmoid')

h2 = Layer(len(h1.output),10)
h2.forwardPropagation(h1.output, 'ReLU')

outputLayer = Layer(len(h2.output), 10)
outputLayer.forwardPropagation(h2.output, 'Softmax')

print('Input Layer')
print(inputLayer)
print('\nHidden Layer 1')
for i in range(len(h1.weights)): print(f'Neuron {i}: {h1.weights[i]}x + {h1.biases[i]}')
print('\nHidden Layer 2')
for i in range(len(h2.weights)): print(f'Neuron {i}: {h2.weights[i]}x + {h2.biases[i]}')
print(f'\nOutput: {outputLayer.output}\n')