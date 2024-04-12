import pickle
from preprocessCDR1 import Pair
from bioc.pubtator.datastructure import PubTatorAnn

# A class to represent the input to the model
class Input:
    def __init__(self, pair):
        self.context = pair.context             # Context in which the pair occurs
        self.query = self.createQuery(pair)     # Query that asks which disease is induced

    def createQuery(self, pair):
        return f"what disease does {pair.chemical.text} induce"

# Function to construct the input consisting of the context and query
def queryConstruction(pairs):
    inputs = []                                 # List to store the inputs to the model
    for pairsInDoc in pairs:
        inputsInDoc = []
        for pair in pairsInDoc:
            inputsInDoc.append(Input(pair))
        inputs.append(inputsInDoc)

    return inputs

if __name__ == '__main__':
    # Load the pairs that were produced by the second stage
    with open('pairs2.pkl', 'rb') as input_file:
        pairs = pickle.load(input_file)
    
    # Create the inputs to the model from the pairs
    inputs = queryConstruction(pairs)

    # Open a file to write the output
    with open('input.pkl', 'wb') as outputFile:
        pickle.dump(inputs, outputFile)

    # Close the file after writing
    outputFile.close()

